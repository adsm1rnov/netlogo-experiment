"""
abm_runner.py - запуск модели refinery_shock_abm (NetLogo 7) из Python через pyNetLogo.

Один процесс = одна JVM = одна загруженная модель. Параллельность - через
ProcessPoolExecutor с контекстом "spawn" (JVM нельзя форкать; на Windows spawn и так по умолчанию).

Основные функции:
    make_link(cfg)                     - JVM + загрузка модели в текущем процессе
    run_one(link, job, cfg)            - один прогон, KPI в контрольных точках
    run_timeseries(link, job, cfg)     - один прогон с записью ряда по дням (диагностика)
    run_batch(jobs, cfg, n_workers)    - пакет прогонов, параллельно, с дозаписью и возобновлением
    override_command(unc, nominal)     - команда NetLogo, переопределяющая константы после setup
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Имена элементов интерфейса модели
# ---------------------------------------------------------------------------
SWITCHES = ["L2-export-ban?", "L3-quality-relax?", "L4-import?", "L5-fillup-limit?", "L6-price-cap?"]
SLIDERS = ["L3-share-naphtha", "L4-import-limit", "L4-subsidy", "L5-liters-per-fillup",
           "L6-price-cap", "shock-depth"]
# Константы из set-constants, которые меняются в LHS неопределённости (c3 - только постобработка)
UNCERTAIN = ["panic-sensitivity", "tau-down", "w-own-experience", "price-elasticity", "normal-surplus"]

# Что читаем в каждой контрольной точке (накопленные KPI + состояние)
KPI_REPORTERS = {
    "K0": "K0-total",
    "K1": "K1-total",
    "K2": "K2-total",
    "K3_volume": "K3-volume-total",
    "K4": "K4-total",
    "K5_hours": "K5-total-hours",
    "price": "price",
    "dry_share": "dry-share",
    "perceived": "mean-perceived-shortage",
    "failed": "failed-count",
}
# Что читаем один раз после setup (паспорт прогона)
SETUP_REPORTERS = {
    "seed_used": "seed-used",
    "policy_delay": "policy-delay",
    "capacity_loss": "market-capacity-loss",
    "D0": "D0",
}
# Ряд по дням для диагностики
TS_REPORTERS = [
    "ticks", "price", "dry-share", "production-today", "export-today", "import-today",
    "sales-today", "consumption-today", "K0-today", "K1-today", "desired-today",
    "failed-count", "mean-perceived-shortage",
    "total-refinery-stock", "total-pipeline-stock", "total-station-stock", "total-consumer-stock",
    "K0-total", "K1-total", "K2-total", "K3-volume-total", "K4-total", "K5-total-hours",
]
CONSTANT_REPORTERS = [
    "shock-start", "policy-delay", "L3-ramp-days", "L4-import-delay",
    "n-refineries", "n-stations", "n-consumers", "normal-surplus", "naphtha-ratio",
    "refinery-storage-days", "transport-delay", "liters-per-car-day", "tank-liters",
    "canister-liters", "reorder-threshold", "price-elasticity", "panic-sensitivity",
    "w-own-experience", "tau-up", "tau-down", "base-search-radius", "radius-growth",
    "max-radius", "price-adj-speed", "ref-price", "price-coverage-coef",
    "service-reserve", "normal-visits", "station-service-capacity", "D0", "target-coverage",
]


def nl_value(v) -> str:
    """Python-значение -> литерал NetLogo."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def nl_list(reporters) -> str:
    return "(list " + " ".join(reporters) + ")"


# ---------------------------------------------------------------------------
# Подключение
# ---------------------------------------------------------------------------
def find_jvm(netlogo_home: str | None) -> str | None:
    """JVM, встроенная в NetLogo 7 (папка runtime). Совместимая версия Java гарантирована."""
    if not netlogo_home:
        return None
    names = ["jvm.dll", "libjvm.so", "libjvm.dylib"]
    for name in names:
        hits = glob.glob(os.path.join(netlogo_home, "**", name), recursive=True)
        hits = [h for h in hits if "server" in h.replace("\\", "/")] or hits
        if hits:
            return hits[0]
    return None


def make_link(cfg: dict):
    """Стартует JVM в текущем процессе и загружает модель. Возвращает pynetlogo.NetLogoLink."""
    import pynetlogo
    jvm = cfg.get("jvm_path") or find_jvm(cfg.get("netlogo_home"))
    link = pynetlogo.NetLogoLink(
        gui=False,
        netlogo_home=cfg.get("netlogo_home"),
        jvm_path=jvm,
        jvmargs=cfg.get("jvmargs", []),
    )
    link.load_model(str(cfg["model_path"]))
    return link


# ---------------------------------------------------------------------------
# Переопределение констант после setup
# ---------------------------------------------------------------------------
def override_command(unc: dict | None, nominal: dict) -> str:
    """
    Константы задаются в set-constants внутри setup (после clear-all), поэтому их нельзя
    выставить до setup. Переопределяем после setup, так, чтобы результат совпадал с тем,
    как если бы константа была другой с самого начала:
      * tau-down, w-own-experience, price-elasticity в setup не используются -> просто set;
      * panic-sensitivity: panic-i = max(0, N(ps, 0.5 ps)) = ps * max(0, N(1, 0.5)),
        т. е. линеен по ps -> panic-i := panic-i * ps_new / ps_old (те же случайные числа);
      * normal-surplus: capacity = (1 + ns) * sum(base-sales) -> масштабируем capacity и
        пересчитываем naphtha-capacity, storage-cap и начальный stock НПЗ (stock = 0.5 * storage-cap).
    Всё внутри with-local-randomness: основной генератор не сдвигается, траектория
    случайных чисел та же, что без переопределения (общие случайные числа сохраняются).
    Значения, равные номиналу, не трогаем: тогда прогон побитово совпадает с прогоном без переопределения.
    """
    if not unc:
        return ""
    parts = []
    for k in ("tau-down", "w-own-experience", "price-elasticity"):
        if k in unc and not math.isclose(float(unc[k]), float(nominal[k]), rel_tol=0, abs_tol=1e-15):
            parts.append(f"set {k} {nl_value(unc[k])}")
    if "panic-sensitivity" in unc and unc["panic-sensitivity"] != nominal["panic-sensitivity"]:
        ps = nl_value(unc["panic-sensitivity"])
        parts.append(
            f"if panic-sensitivity > 0 [ ask consumers [ set panic-i panic-i * {ps} / panic-sensitivity ] ] "
            f"set panic-sensitivity {ps}"
        )
    if "normal-surplus" in unc and unc["normal-surplus"] != nominal["normal-surplus"]:
        ns = nl_value(unc["normal-surplus"])
        parts.append(
            f"ask refineries [ set capacity capacity * (1 + {ns}) / (1 + normal-surplus) "
            f"set naphtha-capacity capacity * naphtha-ratio "
            f"set storage-cap refinery-storage-days * capacity "
            f"set stock 0.5 * storage-cap ] "
            f"set normal-surplus {ns}"
        )
    unknown = set(unc) - set(UNCERTAIN)
    if unknown:
        raise ValueError(f"Неизвестные параметры неопределённости: {unknown}")
    if not parts:
        return ""
    return "with-local-randomness [ " + " ".join(parts) + " ]"


# ---------------------------------------------------------------------------
# Один прогон
# ---------------------------------------------------------------------------
def _prepare(link, job: dict, cfg: dict) -> dict:
    """Выставляет ВСЕ элементы интерфейса (clear-all их не сбрасывает), seed, setup, оверрайды."""
    params = {**cfg["interface_defaults"], **job.get("params", {})}
    cmd = [f"set {name} {nl_value(params[name])}" for name in SWITCHES + SLIDERS]
    seed = int(job["seed"])
    if seed == 0:
        raise ValueError("seed = 0 в модели означает случайное зерно - не используем")
    cmd.append(f"set seed {seed}")
    link.command(" ".join(cmd))
    link.command("setup")
    ov = override_command(job.get("uncertain"), cfg["nominal_uncertain"])
    if ov:
        link.command(ov)
    vals = link.report(nl_list(SETUP_REPORTERS.values()))
    return dict(zip(SETUP_REPORTERS.keys(), np.atleast_1d(vals).tolist()))


def run_one(link, job: dict, cfg: dict) -> dict:
    """Прогон до горизонта; KPI читаются в контрольных точках cfg['checkpoints'] (последняя = горизонт)."""
    t0 = time.perf_counter()
    out = {k: v for k, v in job.items() if k not in ("params", "uncertain")}
    out.update({f"p:{k}": v for k, v in job.get("params", {}).items()})
    out.update({f"u:{k}": v for k, v in (job.get("uncertain") or {}).items()})
    out.update(_prepare(link, job, cfg))
    horizon = int(cfg["horizon"])
    cps = sorted(set(int(c) for c in cfg["checkpoints"] if c < horizon)) + [horizon]
    done = 0
    rep = nl_list(KPI_REPORTERS.values())
    for cp in cps:
        link.command(f"repeat {cp - done} [ go ]")
        done = cp
        vals = np.atleast_1d(link.report(rep)).tolist()
        suffix = "" if cp == horizon else f"@{cp}"
        for k, v in zip(KPI_REPORTERS.keys(), vals):
            out[k + suffix] = float(v)
    out["sec"] = time.perf_counter() - t0
    out["pid"] = os.getpid()
    return out


def run_timeseries(link, job: dict, cfg: dict, ticks: int | None = None) -> pd.DataFrame:
    """Прогон с чтением состояния каждый день (строка 0 - сразу после setup). Только для диагностики."""
    _prepare(link, job, cfg)
    ticks = int(ticks or cfg["horizon"])
    rep = nl_list(TS_REPORTERS)
    rows = [np.atleast_1d(link.report(rep)).tolist()]
    for _ in range(ticks):
        link.command("go")
        rows.append(np.atleast_1d(link.report(rep)).tolist())
    return pd.DataFrame(rows, columns=TS_REPORTERS)


# ---------------------------------------------------------------------------
# Пакетный запуск
# ---------------------------------------------------------------------------
_LINK = None
_CFG = None


def _init_worker(cfg: dict):
    global _LINK, _CFG
    _CFG = cfg
    _LINK = make_link(cfg)


def _worker_task(job: dict) -> dict:
    try:
        return run_one(_LINK, job, _CFG)
    except Exception:
        return {**{k: v for k, v in job.items() if k not in ("params", "uncertain")},
                "error": traceback.format_exc()}


def run_batch(jobs: list[dict], cfg: dict, n_workers: int = 1, out_csv: str | Path | None = None,
              resume: bool = True, link=None, flush_every: int = 50, progress: bool = True) -> pd.DataFrame:
    """
    Пакет прогонов. Каждое задание - dict с ключами run_id (уникальный), seed, params, uncertain
    (+ любые метки: stage, design_id, rep ...).
    n_workers = 1 -> прогон в текущем процессе (нужен link); > 1 -> пул процессов, по JVM в каждом.
    out_csv: результаты дописываются каждые flush_every прогонов; при resume=True уже посчитанные
    run_id пропускаются (можно прервать и продолжить).
    """
    out_csv = Path(out_csv) if out_csv else None
    done_df = None
    if out_csv is not None and resume and out_csv.exists():
        done_df = pd.read_csv(out_csv)
        if "error" in done_df:
            done_df = done_df[done_df["error"].isna()]
        done_ids = set(done_df["run_id"])
        jobs = [j for j in jobs if j["run_id"] not in done_ids]
        print(f"[run_batch] уже посчитано {len(done_ids)}, осталось {len(jobs)}")
    ids = [j["run_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("run_id в заданиях не уникальны")

    results, buffer = [], []

    def flush():
        nonlocal buffer
        if out_csv is not None and buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        buffer = []

    try:
        from tqdm.auto import tqdm
    except ImportError:  # без tqdm - без прогресс-бара
        def tqdm(x, **kw):
            return x

    t_start = time.perf_counter()
    if n_workers <= 1:
        if link is None:
            raise ValueError("Для n_workers = 1 передайте link (make_link(cfg))")
        it = tqdm(jobs, total=len(jobs), disable=not progress)
        for job in it:
            try:
                r = run_one(link, job, cfg)
            except Exception:
                r = {**{k: v for k, v in job.items() if k not in ("params", "uncertain")},
                     "error": traceback.format_exc()}
            results.append(r)
            buffer.append(r)
            if len(buffer) >= flush_every:
                flush()
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(_worker_task, j) for j in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), disable=not progress):
                r = f.result()
                results.append(r)
                buffer.append(r)
                if len(buffer) >= flush_every:
                    flush()
    flush()
    wall = time.perf_counter() - t_start
    df = pd.DataFrame(results)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    if len(jobs):
        print(f"[run_batch] {len(jobs)} прогонов за {wall / 60:.1f} мин "
              f"({len(jobs) / max(wall, 1e-9) * 3600:.0f} прогонов/ч), ошибок: {n_err}")
    if n_err:
        print(df.loc[df["error"].notna(), "error"].iloc[0])
    if done_df is not None and len(done_df):
        df = pd.concat([done_df, df], ignore_index=True)
    df.attrs["wall_sec"] = wall
    return df

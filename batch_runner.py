"""
batch_runner.py - пакетный запуск прогонов для этапов 1-5 (поверх abm_runner.run_one).

Отличия от abm_runner.run_batch:
  * живой журнал с временем, скоростью и оценкой окончания (печать раз в log_every секунд);
  * фиксированный порядок столбцов в CSV (дозапись не сдвигает столбцы), ошибки - в отдельный файл;
  * возобновление по run_id после обрыва;
  * на Windows не даёт системе уснуть, пока идут прогоны.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

import abm_runner as ar

_LINK, _CFG = None, None


def _init(cfg):
    global _LINK, _CFG
    _CFG = cfg
    _LINK = ar.make_link(cfg)


def _task(job):
    try:
        return ar.run_one(_LINK, job, _CFG)
    except Exception:
        return {"run_id": job["run_id"], "error": traceback.format_exc()}


def read_results(path) -> pd.DataFrame:
    """CSV результатов -> DataFrame; переживает выравнивание пробелами в редакторе."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            s = df[c].astype(str).str.strip()
            try:
                df[c] = pd.to_numeric(s)
            except (ValueError, TypeError):
                df[c] = s.replace({"nan": None, "True": True, "False": False})
    return df


class _KeepAwake:
    """Windows: запрет сна на время прогонов (SetThreadExecutionState). На других ОС ничего не делает."""
    def __enter__(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return self

    def __exit__(self, *a):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run_jobs(jobs, cfg, out_csv, n_workers=1, link=None, flush_every=10, log_every=60, resume=True):
    """
    Выполняет задания, дописывает результаты в out_csv (ошибки - в out_csv с суффиксом .errors.csv).
    n_workers = 1: в процессе ноутбука (нужен link); > 1: пул процессов, в каждом своя JVM.
    Возвращает DataFrame всех результатов (включая посчитанные ранее).
    """
    out_csv = Path(out_csv)
    err_csv = out_csv.with_suffix(".errors.csv")
    columns = None
    if resume and out_csv.exists():
        prev = read_results(out_csv)
        columns = list(prev.columns)
        done = set(prev["run_id"])
        jobs = [j for j in jobs if j["run_id"] not in done]
        print(f"{time.strftime('%H:%M:%S')} уже посчитано {len(done)}, осталось {len(jobs)}", flush=True)
    total = len(jobs)
    if total == 0:
        return read_results(out_csv)

    buffer, n_done, n_err = [], 0, 0
    t0 = last_log = time.perf_counter()

    def flush():
        nonlocal buffer, columns
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        if columns is None:
            columns = list(df.columns)
        df.reindex(columns=columns).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        buffer = []

    def handle(r):
        nonlocal n_done, n_err, last_log
        n_done += 1
        if "error" in r:
            n_err += 1
            pd.DataFrame([r]).to_csv(err_csv, mode="a", header=not err_csv.exists(), index=False)
            print(f"ОШИБКА в {r['run_id']}:\n{r['error'][-800:]}", flush=True)
        else:
            buffer.append(r)
        if len(buffer) >= flush_every:
            flush()
        now = time.perf_counter()
        if now - last_log >= log_every or n_done == total:
            el = now - t0
            rate = n_done / el * 3600
            eta = (total - n_done) / max(n_done / el, 1e-9) / 60
            print(f"{time.strftime('%H:%M:%S')} {n_done}/{total} | {el / 60:.0f} мин | "
                  f"{rate:.0f} прогонов/ч | осталось ~{eta:.0f} мин | ошибок {n_err}", flush=True)
            last_log = now

    print(f"{time.strftime('%H:%M:%S')} старт: {total} прогонов, процессов: {n_workers}", flush=True)
    with _KeepAwake():
        try:
            if n_workers <= 1:
                if link is None:
                    raise ValueError("для n_workers = 1 нужен link = ar.make_link(CFG)")
                for j in jobs:
                    try:
                        handle(ar.run_one(link, j, cfg))
                    except Exception:
                        handle({"run_id": j["run_id"], "error": traceback.format_exc()})
            else:
                ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"),
                                         initializer=_init, initargs=(cfg,))
                try:
                    pending = {ex.submit(_task, j) for j in jobs}
                    while pending:
                        fin, pending = wait(pending, timeout=log_every, return_when=FIRST_COMPLETED)
                        if not fin:
                            print(f"{time.strftime('%H:%M:%S')} ... работаю, {n_done}/{total}", flush=True)
                        for f in fin:
                            handle(f.result())
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
        finally:
            flush()
    el = time.perf_counter() - t0
    print(f"{time.strftime('%H:%M:%S')} готово: {n_done} за {el / 60:.1f} мин, ошибок {n_err}", flush=True)
    return read_results(out_csv)

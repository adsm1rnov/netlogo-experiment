"""
abm_runner.py - runs the refinery_shock_abm model (NetLogo 7) from Python via pyNetLogo.

One process = one JVM = one loaded model. Parallelism is achieved through
ProcessPoolExecutor with the "spawn" context (the JVM cannot be forked; on Windows spawn is the default anyway).

Main functions:
    make_link(cfg)                     - JVM + load model in the current process
    run_one(link, job, cfg)            - single run, KPIs at checkpoints
    run_timeseries(link, job, cfg)     - single run with a daily time series (diagnostics)
    override_command(unc, nominal)     - NetLogo command that overrides constants after setup
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
# Model interface element names
# ---------------------------------------------------------------------------
SWITCHES = ["L2-export-ban?", "L3-quality-relax?",
            "L4-import?", "L5-fillup-limit?", "L6-price-cap?"]
SLIDERS = ["L3-share-naphtha", "L4-import-limit", "L4-subsidy", "L5-liters-per-fillup",
           "L6-price-cap", "shock-depth"]
# Constants from set-constants that vary in the LHS of uncertainty (c3 is post-processing only)
UNCERTAIN = ["panic-sensitivity", "tau-down",
             "w-own-experience", "price-elasticity", "normal-surplus"]

# What we read at each checkpoint (cumulative KPIs + state)
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
# What we read once after setup (run passport)
SETUP_REPORTERS = {
    "seed_used": "seed-used",
    "policy_delay": "policy-delay",
    "capacity_loss": "market-capacity-loss",
    "D0": "D0",
}
# Daily series for diagnostics
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
    """
    This function converts a Python value to a string representation suitable for NetLogo commands.
    
    Parameters:
        v: The Python value to be converted. It can be a boolean, integer, or float.
    
    Returns:
        A string representation of the input value formatted for NetLogo commands.
    
    """
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def nl_list(reporters) -> str:
    return "(list " + " ".join(reporters) + ")"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def find_jvm(netlogo_home: str | None) -> str | None:
    """
    This function searches for the Java Virtual Machine (JVM) library within the specified NetLogo home directory.
    
    Parameters:
        netlogo_home: The path to the NetLogo home directory.

    Returns:
        The path to the JVM library, or None if it is not found.
    """
    if not netlogo_home:
        return None
    names = ["jvm.dll", "libjvm.so", "libjvm.dylib"]
    for name in names:
        hits = glob.glob(os.path.join(
            netlogo_home, "**", name), recursive=True)
        hits = [h for h in hits if "server" in h.replace("\\", "/")] or hits
        if hits:
            return hits[0]
    return None


def make_link(cfg: dict):
    """
    Starts the JVM in the current process and loads the model. Returns pynetlogo.NetLogoLink.
    
    Parameters:
        cfg: A dictionary containing configuration parameters, including 'jvm_path', 'netlogo_home', and 'model_path'.
        
    Returns:
        An instance of pynetlogo.NetLogoLink connected to the specified NetLogo model.
    
    """
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
# Overriding constants after setup
# ---------------------------------------------------------------------------
def override_command(unc: dict | None, nominal: dict) -> str:
    """
    This function generates a NetLogo command string to
    override certain constants after the setup phase of the model.
    
    Constants are set in set-constants inside setup (after clear-all), so they cannot be
    set before setup. We override them after setup so the result matches what it would be
    if the constant had been different from the start:
      * tau-down, w-own-experience, price-elasticity are not used in setup -> just set;
      * panic-sensitivity: panic-i = max(0, N(ps, 0.5 ps)) = ps * max(0, N(1, 0.5)),
        i.e. linear in ps -> panic-i := panic-i * ps_new / ps_old (same random numbers);
      * normal-surplus: capacity = (1 + ns) * sum(base-sales) -> scale capacity and
        recompute naphtha-capacity, storage-cap, and the initial refinery stock (stock = 0.5 * storage-cap).
    All inside with-local-randomness: the main generator is not shifted, the random
    trajectory is the same as without override (shared random numbers are preserved).
    Values equal to the nominal are left untouched: then the run is bit-for-bit identical to a run without override.
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
        raise ValueError(f"Unknown uncertainty parameters: {unknown}")
    if not parts:
        return ""
    return "with-local-randomness [ " + " ".join(parts) + " ]"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def _prepare(link, job: dict, cfg: dict) -> dict:
    """
    This function prepares the NetLogo model for a single run
    by setting parameters,executing setup,
    and applying any necessary overrides.
    
    Parameters:
        link: The NetLogoLink instance used to communicate with the NetLogo model.
        job: A dictionary containing the parameters for the specific run, including 'params', 'uncertain', and 'seed'.
        cfg: A dictionary containing configuration settings, including 'interface_defaults' and 'nominal_uncertain'.
        
    Returns:
        A dictionary containing the initial state of the model after setup and overrides.

    """
    params = {**cfg["interface_defaults"], **job.get("params", {})}
    cmd = [
        f"set {name} {nl_value(params[name])}" for name in SWITCHES + SLIDERS]
    seed = int(job["seed"])
    if seed == 0:
        raise ValueError(
            "seed = 0 in the model means a random seed - not used")
    cmd.append(f"set seed {seed}")
    link.command(" ".join(cmd))
    link.command("setup")
    ov = override_command(job.get("uncertain"), cfg["nominal_uncertain"])
    if ov:
        link.command(ov)
    vals = link.report(nl_list(SETUP_REPORTERS.values()))
    return dict(zip(SETUP_REPORTERS.keys(), np.atleast_1d(vals).tolist()))


def run_one(link, job: dict, cfg: dict) -> dict:
    """
    This function executes a single run of the NetLogo model,
    collecting key performance indicators (KPIs) at specified checkpoints.
    
    Parameters:
        link: The NetLogoLink instance used to communicate with the NetLogo model.
        job: A dictionary containing the parameters for the specific run, including 'params', 'uncertain', and 'seed'.
        cfg: A dictionary containing configuration
    
    Returns:
        A dictionary containing the results of the run, including KPIs at each checkpoint, execution time, and process ID.
    """
    t0 = time.perf_counter()
    out = {k: v for k, v in job.items() if k not in ("params", "uncertain")}
    out.update({f"p:{k}": v for k, v in job.get("params", {}).items()})
    out.update({f"u:{k}": v for k, v in (job.get("uncertain") or {}).items()})
    out.update(_prepare(link, job, cfg))
    horizon = int(cfg["horizon"])
    cps = sorted(set(int(c)
                 for c in cfg["checkpoints"] if c < horizon)) + [horizon]
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
    """Run with state read each day (row 0 = right after setup). Diagnostics only."""
    _prepare(link, job, cfg)
    ticks = int(ticks or cfg["horizon"])
    rep = nl_list(TS_REPORTERS)
    rows = [np.atleast_1d(link.report(rep)).tolist()]
    for _ in range(ticks):
        link.command("go")
        rows.append(np.atleast_1d(link.report(rep)).tolist())
    return pd.DataFrame(rows, columns=TS_REPORTERS)

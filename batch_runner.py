"""
batch_runner.py - batch execution of runs for stages 1-5

Replaces functionality from abm_runner.run_batch. Improvements include:
  * live log with elapsed time, rate, and ETA (printed every log_every seconds);
  * fixed column order in the CSV (appends don't shift columns), errors go to a separate file;
  * resume by run_id after an interruption;
  * on Windows, prevents the system from sleeping while runs are in progress;
  * on completion, forcibly terminates pool processes (the JVM does not exit on its own);
  * all console output is in English.
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
    """
    This function reads a CSV file and returns a DataFrame with cleaned column names and types.

    Parameters:
        path: The path to the CSV file to be read.

    Returns:
        A pandas DataFrame containing the data from the CSV file, with cleaned column names and types.
    """
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
    """
    This class prevents the system from going to sleep while the context is active.
    """

    def __enter__(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(
                0x80000000 | 0x00000001)
        return self

    def __exit__(self, *a):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run_jobs(jobs, cfg, out_csv, n_workers=1, link=None, flush_every=10, log_every=60, resume=True):
    """
    This function executes a batch of jobs and appends the results to a specified CSV file.
    If any errors occur during the execution of a job, they are logged to a separate error CSV file with a ".errors.csv" suffix.
    
    Parameters:
        jobs: A list of job dictionaries to be executed.
        cfg: A configuration dictionary containing settings for the execution.
        out_csv: The path to the output CSV file where results will be appended.
        n_workers: The number of worker processes to use for parallel execution (default is 1).
        link: An optional link object for communication with the model (required if n_workers=1).
        flush_every: The number of completed jobs after which results are flushed to the output CSV (default is 10).
        log_every: The number of seconds between logging progress updates (default is 60).
        resume: A boolean indicating whether to resume from a previous run by skipping already completed jobs (default is True).
        
    Returns:
        A pandas DataFrame containing the results of all executed jobs, including any errors that occurred.    
    """
    out_csv = Path(out_csv)
    err_csv = out_csv.with_suffix(".errors.csv")
    columns = None
    if resume and out_csv.exists():
        prev = read_results(out_csv)
        columns = list(prev.columns)
        done = set(prev["run_id"])
        jobs = [j for j in jobs if j["run_id"] not in done]
        print(
            f"{time.strftime('%H:%M:%S')} already done: {len(done)}, remaining: {len(jobs)}", flush=True)
    total = len(jobs)
    if total == 0:
        return read_results(out_csv)

    buffer, n_done, n_err = [], 0, 0
    t0 = last_log = time.perf_counter()

    def flush():
        """
        This function flushes the buffer to the output CSV file and resets the buffer.
        
        """
        nonlocal buffer, columns
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        if columns is None:
            columns = list(df.columns)
        df.reindex(columns=columns).to_csv(out_csv, mode="a",
                                           header=not out_csv.exists(), index=False)
        buffer = []

    def handle(r):
        nonlocal n_done, n_err, last_log
        n_done += 1
        if "error" in r:
            n_err += 1
            pd.DataFrame([r]).to_csv(err_csv, mode="a",
                                     header=not err_csv.exists(), index=False)
            print(f"ERROR in {r['run_id']}:\n{r['error'][-800:]}", flush=True)
        else:
            buffer.append(r)
        if len(buffer) >= flush_every:
            flush()
        now = time.perf_counter()
        if now - last_log >= log_every or n_done == total:
            el = now - t0
            rate = n_done / el * 3600
            eta = (total - n_done) / max(n_done / el, 1e-9) / 60
            print(f"{time.strftime('%H:%M:%S')} {n_done}/{total} | {el / 60:.0f} min | "
                  f"{rate:.0f} runs/h | ETA ~{eta:.0f} min | errors {n_err}", flush=True)
            last_log = now

    print(f"{time.strftime('%H:%M:%S')} start: {total} runs, workers: {n_workers}", flush=True)
    with _KeepAwake():
        try:
            if n_workers <= 1:
                if link is None:
                    raise ValueError(
                        "n_workers = 1 requires link = ar.make_link(CFG)")
                for j in jobs:
                    try:
                        handle(ar.run_one(link, j, cfg))
                    except Exception:
                        handle({"run_id": j["run_id"],
                               "error": traceback.format_exc()})
            else:
                ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"),
                                         initializer=_init, initargs=(cfg,))
                try:
                    pending = {ex.submit(_task, j) for j in jobs}
                    while pending:
                        fin, pending = wait(
                            pending, timeout=log_every, return_when=FIRST_COMPLETED)
                        if not fin:
                            print(
                                f"{time.strftime('%H:%M:%S')} ... working, {n_done}/{total}", flush=True)
                        for f in fin:
                            handle(f.result())
                finally:
                    # the JVM process doesn't exit on its own (JPype waits for NetLogo threads) - terminate forcibly
                    for p in list(getattr(ex, "_processes", {}).values()):
                        p.terminate()
                    ex.shutdown(wait=False, cancel_futures=True)
        finally:
            flush()
    el = time.perf_counter() - t0
    print(f"{time.strftime('%H:%M:%S')} done: {n_done} in {el / 60:.1f} min, errors {n_err}", flush=True)
    return read_results(out_csv)

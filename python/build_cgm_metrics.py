"""Extract per-patient CGM metrics in a baseline and a follow-up window for each
of the six Jaeb public RCTs, plus the gap structure needed for the missingness
workstream.

    python python/build_cgm_metrics.py

Writes to output/:
    cgm_windowed.parquet     every CGM reading falling inside a window
    cgm_metrics.csv          one row per (trial, PtID, period): TIR/TAR/TBR/CV/coverage
    cgm_gaps.csv             inter-reading gap distribution per (trial, PtID, period)
    cgm_build_log.md         what was read, what parsed, what was dropped

Windows follow the consensus 14-day CGM analysis window, anchored on each
patient's OWN visit dates (verified to share the CGM clock in every trial):
    baseline  = (randomisation visit - 14d, randomisation visit]
    follow-up = (primary endpoint visit - 14d, primary endpoint visit]

RT-CGM is the exception and is handled explicitly: its control arm wore only
BLINDED CGM at scheduled intervals while the treatment arm wore real-time CGM
continuously, so a 14-day window is not comparable across arms. For that trial
the window is widened to 28 days and per-arm coverage is reported, because the
measurement process itself differs by arm -- a structural asymmetry that must
be visible in the results, not silently averaged away.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)

ART = {
    "CITY": "CITYPublicDataset-344bea7d-8085-4deb-8038-6cb747a744e3.zip",
    "DCLP3": "DCLP3 Public Dataset - Release 3 - 2022-08-04.zip",
    "PEDAP": "PEDAP Public Dataset - Release 5 - 2025-05-12.zip",
    "ReplaceBG": "Replace-BG Dataset.zip",
    "RT_CGM": "RT_CGM_Randomized_Clinical_Trial.zip",
    "WISDM": "WISDMPublicDataset-18f24ae5-b4fb-4e93-bec6-7021086419fa.zip",
}
ZIP_PATHS: dict[str, str] = {}   # filled by main() from a path map written by the caller

CHUNK = 500_000
LOG: list[str] = []


def log(m: str) -> None:
    LOG.append(m)
    print(m, flush=True)


# --------------------------------------------------------------------------
# small-table helpers
# --------------------------------------------------------------------------
def detect(path_or_bytes) -> tuple[str, str]:
    raw = (open(path_or_bytes, "rb").read(4096)
           if isinstance(path_or_bytes, (str, Path)) else path_or_bytes[:4096])
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc, head = "utf-16", raw.decode("utf-16", errors="replace").split("\n")[0]
    else:
        enc, head = "utf-8-sig", raw.decode("utf-8", errors="replace").split("\n")[0]
    sep = max(("|", "\t", ","), key=head.count)
    return enc, (sep if head.count(sep) >= 1 else ",")


def read_member(trial: str, member: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(ZIP_PATHS[trial]) as z:
        b = z.read(member)
    enc, sep = detect(b)
    return pd.read_csv(io.BytesIO(b), sep=sep, encoding=enc, low_memory=False, **kw)


def find_member(trial: str, basename: str) -> str:
    with zipfile.ZipFile(ZIP_PATHS[trial]) as z:
        for n in z.namelist():
            if os.path.basename(n).lower() == basename.lower():
                return n
    raise FileNotFoundError(f"{trial}: {basename}")


def dt(s: pd.Series, fmt: str | None = None) -> pd.Series:
    return pd.to_datetime(s, format=fmt, errors="coerce") if fmt else \
           pd.to_datetime(s, errors="coerce", format="mixed")


# --------------------------------------------------------------------------
# per-trial window tables:  PtID, arm, bl_start, bl_end, fu_start, fu_end
# --------------------------------------------------------------------------
def windows_visitdt(trial: str, roster: str, armcol: str, visitinfo: str,
                    rand_label: str, fu_label: str, days: int = 14) -> pd.DataFrame:
    r = read_member(trial, find_member(trial, roster))
    v = read_member(trial, find_member(trial, visitinfo))
    r = r[["PtID", armcol]].rename(columns={armcol: "arm"}).dropna(subset=["arm"])
    v = v[["PtID", "Visit", "VisitDt"]].copy()
    v["VisitDt"] = dt(v["VisitDt"])
    rd = (v[v["Visit"] == rand_label].groupby("PtID")["VisitDt"].min()
          .rename("rand_dt"))
    fd = (v[v["Visit"] == fu_label].groupby("PtID")["VisitDt"].min()
          .rename("fu_dt"))
    w = r.set_index("PtID").join([rd, fd], how="left").reset_index()
    # visit dates are date-only; extend the end bound to the end of that day
    w["bl_end"] = w["rand_dt"] + pd.Timedelta(days=1)
    w["bl_start"] = w["bl_end"] - pd.Timedelta(days=days)
    w["fu_end"] = w["fu_dt"] + pd.Timedelta(days=1)
    w["fu_start"] = w["fu_end"] - pd.Timedelta(days=days)
    return w


def windows_replacebg(days: int = 14) -> pd.DataFrame:
    r = read_member("ReplaceBG", find_member("ReplaceBG", "HPtRoster.txt"))
    v = read_member("ReplaceBG", find_member("ReplaceBG", "HVisitInfo.txt"))
    r = r[["PtID", "TrtGroup"]].rename(columns={"TrtGroup": "arm"})
    v = v[["PtID", "Visit", "VisitDtDaysFromEnroll"]].copy()
    rd = (v[v["Visit"] == "Randomization"].groupby("PtID")["VisitDtDaysFromEnroll"]
          .min().rename("rand_d"))
    fd = (v[v["Visit"] == "Week 26 Visit"].groupby("PtID")["VisitDtDaysFromEnroll"]
          .min().rename("fu_d"))
    w = r.set_index("PtID").join([rd, fd], how="left").reset_index()
    # day-integer clock: windows are in DAYS, not timestamps
    w["bl_end"] = w["rand_d"]
    w["bl_start"] = w["rand_d"] - days
    w["fu_end"] = w["fu_d"]
    w["fu_start"] = w["fu_d"] - days
    return w


def windows_rtcgm(days: int = 28) -> pd.DataFrame:
    ps = read_member("RT_CGM", find_member("RT_CGM", "tblAPtSummary.csv"))
    lab = read_member("RT_CGM", find_member("RT_CGM", "tblALabHbA1c.csv"))
    ps = ps[["PtID", "TxGroup", "RandDt"]].rename(columns={"TxGroup": "arm"})
    ps["rand_dt"] = dt(ps["RandDt"])
    lab["LabHbA1cDt"] = dt(lab["LabHbA1cDt"])
    fd = (lab[lab["Visit"] == "Phase 1: 26 week"].groupby("PtID")["LabHbA1cDt"]
          .min().rename("fu_dt"))
    w = ps[["PtID", "arm", "rand_dt"]].set_index("PtID").join(fd, how="left").reset_index()
    # fall back to nominal 26 weeks where the 26-week draw date is absent
    w["fu_dt"] = w["fu_dt"].fillna(w["rand_dt"] + pd.Timedelta(days=182))
    w["bl_end"] = w["rand_dt"] + pd.Timedelta(days=1)
    w["bl_start"] = w["bl_end"] - pd.Timedelta(days=days)
    w["fu_end"] = w["fu_dt"] + pd.Timedelta(days=1)
    w["fu_start"] = w["fu_end"] - pd.Timedelta(days=days)
    return w


# --------------------------------------------------------------------------
# CGM sources: (member glob, ptcol, dtcol, glucosecol, encoding, datetime fmt)
# --------------------------------------------------------------------------
CGM_SPEC = {
    "CITY":      [("Data Tables/DeviceCGM.txt", "PtID", "DeviceDtTm", "Value",
                   "utf-16", "%Y-%m-%d %H:%M:%S", "|")],
    "WISDM":     [("Data Tables/DeviceCGM.txt", "PtID", "DeviceDtTm", "Value",
                   "utf-16", "%Y-%m-%d %H:%M:%S", "|")],
    "DCLP3":     [("Data Files/cgm.txt", "PtID", "DataDtTm", "CGM",
                   "utf-8", "%d%b%y:%H:%M:%S", "|")],
    "PEDAP":     [("Data Files/PEDAPDexcomClarityCGM.txt", "PtID", "DeviceDtTm", "CGM",
                   "utf-8", "%m/%d/%Y %I:%M:%S %p", "|"),
                  ("Data Files/PEDAPTandemCGMDATAGXB.txt", "PtID", "DeviceDtTm", "CGMValue",
                   "utf-8", "%m/%d/%Y %I:%M:%S %p", "|")],
    "ReplaceBG": [("Data Tables/HDeviceCGM.txt", "PtID", "DeviceDtTmDaysFromEnroll",
                   "GlucoseValue", "utf-8", None, "|")],
    "RT_CGM":    [("__GLOB__tblADataRTCGM", "PtID", "DeviceDtTm", "Glucose",
                   "utf-8", "%Y-%m-%d %H:%M:%S.%f", ",")],
}


def cgm_members(trial: str, spec_member: str) -> list[str]:
    if not spec_member.startswith("__GLOB__"):
        return [spec_member]
    pat = spec_member.replace("__GLOB__", "")
    with zipfile.ZipFile(ZIP_PATHS[trial]) as z:
        return sorted(n for n in z.namelist() if pat in n and n.endswith(".csv"))


def harvest(trial: str, win: pd.DataFrame) -> pd.DataFrame:
    """Stream the trial's CGM files, keeping only readings inside a window."""
    daybased = trial == "ReplaceBG"
    wl = win.dropna(subset=["bl_start", "fu_start"], how="all").copy()
    bounds = wl.set_index("PtID")[["bl_start", "bl_end", "fu_start", "fu_end"]]
    keep: list[pd.DataFrame] = []
    n_read = n_bad = 0

    for spec_member, ptc, dtc, gc, enc, fmt, sep in CGM_SPEC[trial]:
        for member in cgm_members(trial, spec_member):
            src = os.path.basename(member)
            with zipfile.ZipFile(ZIP_PATHS[trial]) as z, z.open(member) as fh:
                wrapper = io.TextIOWrapper(fh, encoding=enc)
                usecols = [ptc, dtc, gc]
                if daybased:
                    usecols.append("DeviceTm")
                for ch in pd.read_csv(wrapper, sep=sep, usecols=usecols,
                                      chunksize=CHUNK, low_memory=False):
                    n_read += len(ch)
                    ch = ch.rename(columns={ptc: "PtID", gc: "glu"})
                    ch["glu"] = pd.to_numeric(ch["glu"], errors="coerce")
                    if daybased:
                        day = pd.to_numeric(ch[dtc], errors="coerce")
                        tod = pd.to_timedelta(ch["DeviceTm"] + ":00", errors="coerce")
                        ch["t"] = day          # integer day for windowing
                        ch["ts"] = day * 86400 + tod.dt.total_seconds()
                    else:
                        ts = pd.to_datetime(ch[dtc], format=fmt, errors="coerce")
                        n_bad += int(ts.isna().sum())
                        ch["t"] = ts
                        ch["ts"] = ts
                    ch = ch.dropna(subset=["PtID", "t", "glu"])
                    ch = ch.join(bounds, on="PtID", how="inner")
                    if ch.empty:
                        continue
                    in_bl = ch["t"].between(ch["bl_start"], ch["bl_end"])
                    in_fu = ch["t"].between(ch["fu_start"], ch["fu_end"])
                    ch = ch[in_bl | in_fu].copy()
                    if ch.empty:
                        continue
                    ch["period"] = np.where(in_bl[ch.index], "baseline", "followup")
                    ch["source"] = src
                    keep.append(ch[["PtID", "period", "ts", "glu", "source"]])
            log(f"    {trial}/{src}: cumulative kept={sum(len(k) for k in keep):,}")

    out = (pd.concat(keep, ignore_index=True) if keep
           else pd.DataFrame(columns=["PtID", "period", "ts", "glu", "source"]))
    log(f"  {trial}: read {n_read:,} rows, unparsed timestamps {n_bad:,}, "
        f"kept {len(out):,} in-window")
    # PEDAP: two device feeds can both cover a patient -- de-duplicate to the
    # 5-minute grid so a dual-source patient does not get double-counted.
    if trial == "PEDAP" and not out.empty:
        before = len(out)
        key = out["ts"].astype("int64") // (5 * 60 * 10**9) if np.issubdtype(
            out["ts"].dtype, np.datetime64) else (out["ts"] // 300).astype("int64")
        out = out.assign(_k=key).drop_duplicates(["PtID", "period", "_k"]).drop(columns="_k")
        log(f"  PEDAP: de-duplicated dual device feeds {before:,} -> {len(out):,}")
    out["trial"] = trial
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(df: pd.DataFrame, window_days: dict[str, int]) -> pd.DataFrame:
    rows = []
    for (trial, pt, per), g in df.groupby(["trial", "PtID", "period"], sort=False):
        glu = g["glu"].to_numpy(float)
        glu = glu[np.isfinite(glu)]
        n = glu.size
        if n == 0:
            continue
        wd = window_days[trial]
        expected = wd * 288          # 5-minute cadence
        rows.append(dict(
            trial=trial, PtID=pt, period=per, n_readings=n,
            coverage=min(1.0, n / expected),
            window_days=wd,
            gluMean=glu.mean(), gluSD=glu.std(ddof=1) if n > 1 else np.nan,
            gluCV=(glu.std(ddof=1) / glu.mean() * 100) if n > 1 and glu.mean() else np.nan,
            TIR=(glu >= 70).__and__(glu <= 180).mean() * 100,
            TBR70=(glu < 70).mean() * 100,
            TBR54=(glu < 54).mean() * 100,
            TAR180=(glu > 180).mean() * 100,
            TAR250=(glu > 250).mean() * 100,
            frac_at_floor=(glu <= 40).mean() * 100,
            frac_at_ceiling=(glu >= 400).mean() * 100,
            n_sources=g["source"].nunique(),
        ))
    return pd.DataFrame(rows)


def gaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (trial, pt, per), g in df.groupby(["trial", "PtID", "period"], sort=False):
        ts = g["ts"].sort_values()
        if np.issubdtype(ts.dtype, np.datetime64):
            d = ts.diff().dt.total_seconds().to_numpy()[1:] / 60.0
        else:
            d = np.diff(ts.to_numpy(float)) / 60.0
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            continue
        rows.append(dict(
            trial=trial, PtID=pt, period=per, n_intervals=d.size,
            median_interval_min=float(np.median(d)),
            p95_interval_min=float(np.percentile(d, 95)),
            max_gap_min=float(d.max()),
            n_gaps_gt_30min=int((d > 30).sum()),
            n_gaps_gt_2h=int((d > 120).sum()),
            n_gaps_gt_6h=int((d > 360).sum()),
            frac_time_in_gaps_gt_30min=float(d[d > 30].sum() / d.sum()),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    import json
    global ZIP_PATHS
    ZIP_PATHS = json.load(open(OUT / "zip_paths.json"))

    win_specs = {
        "CITY":      lambda: windows_visitdt("CITY", "PtRoster.txt", "TrtGroup",
                                             "VisitInfo.txt", "Randomization", "26 week visit"),
        "WISDM":     lambda: windows_visitdt("WISDM", "PtRoster.txt", "TrtGroup",
                                             "VisitInfo.txt", "Randomization", "26 week"),
        "DCLP3":     lambda: windows_visitdt("DCLP3", "PtRoster_a.txt", "trtGroup",
                                             "VisitInfo_a.txt", "Randomization", "26-Week"),
        "PEDAP":     lambda: windows_visitdt("PEDAP", "PtRoster.txt", "TrtGroup",
                                             "PEDAPVisitInfo.txt", "Randomization", "13 Week"),
        "ReplaceBG": windows_replacebg,
        "RT_CGM":    windows_rtcgm,
    }
    window_days = {"CITY": 14, "WISDM": 14, "DCLP3": 14, "PEDAP": 14,
                   "ReplaceBG": 14, "RT_CGM": 28}

    all_win, all_cgm = [], []
    for trial in ART:
        log(f"\n=== {trial}")
        w = win_specs[trial]()
        w["trial"] = trial
        log(f"  windows: n={len(w)} arms={dict(w['arm'].value_counts())} "
            f"baseline_ok={w['bl_start'].notna().sum()} followup_ok={w['fu_start'].notna().sum()}")
        all_win.append(w)
        all_cgm.append(harvest(trial, w))

    win_all = pd.concat(all_win, ignore_index=True)
    cgm_all = pd.concat(all_cgm, ignore_index=True)
    win_all.to_csv(OUT / "cgm_windows.csv", index=False)

    m = metrics(cgm_all, window_days)
    m = m.merge(win_all[["trial", "PtID", "arm"]], on=["trial", "PtID"], how="left")
    m.to_csv(OUT / "cgm_metrics.csv", index=False)
    gaps(cgm_all).to_csv(OUT / "cgm_gaps.csv", index=False)

    log("\n=== per-trial metric coverage ===")
    for trial, g in m.groupby("trial"):
        piv = g.pivot_table(index="arm", columns="period", values="PtID", aggfunc="count")
        log(f"  {trial}: patients with metrics by arm/period\n{piv.to_string()}")
        log(f"    median coverage: "
            f"{g.groupby('period')['coverage'].median().round(3).to_dict()}")

    (OUT / "cgm_build_log.md").write_text(
        "# CGM metric build log\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()

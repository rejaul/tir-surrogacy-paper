"""RT-CGM needs different handling from the other five trials.

Two problems found in the first pass:

  1. ~6% of its timestamps failed to parse. Its CSVs mix 6-digit and 9-digit
     fractional seconds ("...:14.010000000" vs "...:00.000000"); `%f` accepts at
     most 6, so whole files silently dropped out -- including
     tblADataRTCGM_Blind_Baseline.csv, the blinded baseline for BOTH arms, which
     is why the treatment arm showed 3% baseline coverage.

  2. More fundamentally, the trial's two arms were MEASURED DIFFERENTLY. The
     real-time CGM arm wore CGM continuously; the control arm wore BLINDED CGM
     only in scheduled episodes. A fixed calendar window anchored on the
     26-week visit therefore lands inside a wearing episode for the treatment
     arm and in empty space for the control arm (0.8% coverage). That is not
     missing data to be imputed -- it is a different measurement process, and
     averaging over it would manufacture a treatment effect on TIR out of a
     sampling-design difference.

This script addresses (1) by truncating fractional seconds before parsing, and
(2) by switching RT-CGM to EPISODE-BASED windows: for each patient it finds
contiguous CGM wearing episodes (gap > 24 h starts a new episode), then selects
the episode closest to randomisation as baseline and the episode closest to the
26-week visit as follow-up. Metrics are computed on whichever episode was
actually worn, and the episode's length and distance from the target visit are
recorded so that comparability can be judged rather than assumed.

    python python/rescue_rtcgm.py   ->  output/rtcgm_episode_metrics.csv
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
ZIPS = json.load(open(OUT / "zip_paths.json"))
TRIAL = "RT_CGM"
GAP_H = 24.0          # a gap longer than this starts a new wearing episode
MIN_EPISODE_READINGS = 288 // 2   # >= ~half a day of data to be usable


def parse_ts(s: pd.Series) -> pd.Series:
    """Robust to 6- vs 9-digit fractional seconds."""
    t = s.astype(str).str.slice(0, 19)      # 'YYYY-MM-DD HH:MM:SS'
    return pd.to_datetime(t, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def load_all() -> pd.DataFrame:
    frames, n_read, n_bad = [], 0, 0
    with zipfile.ZipFile(ZIPS[TRIAL]) as z:
        members = sorted(n for n in z.namelist()
                         if "tblADataRTCGM" in n and n.endswith(".csv"))
        for mem in members:
            with z.open(mem) as fh:
                for ch in pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"),
                                      usecols=["PtID", "DeviceDtTm", "Glucose"],
                                      chunksize=500_000, low_memory=False):
                    n_read += len(ch)
                    ts = parse_ts(ch["DeviceDtTm"])
                    n_bad += int(ts.isna().sum())
                    glu = pd.to_numeric(ch["Glucose"], errors="coerce")
                    d = pd.DataFrame({"PtID": ch["PtID"], "ts": ts, "glu": glu,
                                      "src": os.path.basename(mem)}).dropna()
                    if len(d):
                        frames.append(d)
            print(f"  {os.path.basename(mem)}: total kept {sum(map(len, frames)):,}",
                  flush=True)
    print(f"RT_CGM: read {n_read:,}, unparsed {n_bad:,} ({n_bad/max(1,n_read):.2%})")
    return pd.concat(frames, ignore_index=True)


def episodes(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("ts")
    gap_h = g["ts"].diff().dt.total_seconds() / 3600.0
    ep = (gap_h > GAP_H).fillna(False).cumsum()
    return g.assign(ep=ep)


def main() -> None:
    cgm = load_all()
    ps = pd.read_csv(io.BytesIO(
        zipfile.ZipFile(ZIPS[TRIAL]).read(
            [n for n in zipfile.ZipFile(ZIPS[TRIAL]).namelist()
             if n.endswith("tblAPtSummary.csv")][0])))
    lab = pd.read_csv(io.BytesIO(
        zipfile.ZipFile(ZIPS[TRIAL]).read(
            [n for n in zipfile.ZipFile(ZIPS[TRIAL]).namelist()
             if n.endswith("tblALabHbA1c.csv")][0])))

    ps["rand_dt"] = pd.to_datetime(ps["RandDt"], errors="coerce", format="mixed")
    lab["dt"] = pd.to_datetime(lab["LabHbA1cDt"], errors="coerce", format="mixed")
    fu = (lab[lab["Visit"] == "Phase 1: 26 week"].groupby("PtID")["dt"]
          .min().rename("fu_dt"))
    anchors = (ps[["PtID", "TxGroup", "rand_dt"]].rename(columns={"TxGroup": "arm"})
               .set_index("PtID").join(fu).reset_index())
    anchors["fu_dt"] = anchors["fu_dt"].fillna(
        anchors["rand_dt"] + pd.Timedelta(days=182))

    rows = []
    for pt, g in cgm.groupby("PtID", sort=False):
        a = anchors[anchors["PtID"] == pt]
        if a.empty or pd.isna(a["rand_dt"].iloc[0]):
            continue
        rand_dt, fu_dt, arm = a["rand_dt"].iloc[0], a["fu_dt"].iloc[0], a["arm"].iloc[0]
        ge = episodes(g)
        summ = ge.groupby("ep").agg(start=("ts", "min"), end=("ts", "max"),
                                    n=("ts", "size")).reset_index()
        summ = summ[summ["n"] >= MIN_EPISODE_READINGS]
        if summ.empty:
            continue
        summ["mid"] = summ["start"] + (summ["end"] - summ["start"]) / 2
        for period, anchor in (("baseline", rand_dt), ("followup", fu_dt)):
            s = summ.copy()
            s["dist_d"] = (s["mid"] - anchor).dt.total_seconds() / 86400.0
            if period == "baseline":
                s = s[s["dist_d"] <= 7]              # at/before randomisation
            else:
                s = s[s["dist_d"] > 7]               # after randomisation
            if s.empty:
                continue
            pick = s.iloc[s["dist_d"].abs().argmin()]
            gl = ge.loc[ge["ep"] == pick["ep"], "glu"].to_numpy(float)
            gl = gl[np.isfinite(gl)]
            if gl.size < MIN_EPISODE_READINGS:
                continue
            span_d = (pick["end"] - pick["start"]).total_seconds() / 86400.0
            rows.append(dict(
                trial=TRIAL, PtID=pt, arm=arm, period=period,
                n_readings=gl.size,
                episode_days=round(span_d, 2),
                episode_dist_days=round(float(pick["dist_d"]), 1),
                coverage=min(1.0, gl.size / max(1.0, span_d * 288)),
                gluMean=gl.mean(), gluSD=gl.std(ddof=1) if gl.size > 1 else np.nan,
                gluCV=(gl.std(ddof=1) / gl.mean() * 100) if gl.mean() else np.nan,
                TIR=((gl >= 70) & (gl <= 180)).mean() * 100,
                TBR70=(gl < 70).mean() * 100, TBR54=(gl < 54).mean() * 100,
                TAR180=(gl > 180).mean() * 100, TAR250=(gl > 250).mean() * 100,
            ))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rtcgm_episode_metrics.csv", index=False)

    print("\n=== RT-CGM episode-based windows ===")
    print(out.pivot_table(index="arm", columns="period",
                          values="PtID", aggfunc="count").to_string())
    for c in ("episode_days", "episode_dist_days", "coverage", "TIR"):
        print(f"\nmedian {c}:")
        print(out.pivot_table(index="arm", columns="period",
                              values=c, aggfunc="median").round(2).to_string())


if __name__ == "__main__":
    main()

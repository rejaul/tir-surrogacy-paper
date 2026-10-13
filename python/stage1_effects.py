"""Stage 1 of the two-stage surrogacy model: per-trial treatment effects on the
surrogate (TIR) and the true endpoint (central-lab HbA1c), from patient-level
Jaeb RCT data.

    python python/stage1_effects.py

Inputs (output/): cgm_metrics.csv, rtcgm_episode_metrics.csv, a1c_endpoints.csv
Outputs (output/):
    stage1_patient.csv    analysis dataset, one row per patient
    trial_effects.csv     one row per trial: alpha_hat, beta_hat, SEs, and the
                          WITHIN-TRIAL COVARIANCE of the two effect estimates
    stage1_log.md

Model, per trial and per endpoint, is ANCOVA on the follow-up value adjusting
for the baseline value:

    TIR_fu  = mu_S + alpha * arm + gamma_S * TIR_bl  + e_S
    A1c_fu  = mu_T + beta  * arm + gamma_T * A1c_bl  + e_T

ANCOVA rather than change-from-baseline because it is the more efficient and is
what these trials pre-specified; baseline adjustment also absorbs part of the
range-restriction problem that flattens the Stage-2 slope.

Two quantities that aggregate published data cannot give us and patient-level
data can:

  * R2_indiv -- the squared correlation of the two models' residuals within a
    trial, i.e. individual-level surrogacy. (Finding 1 of PIPELINE_AUDIT.md was
    that the first-draft pipeline reported this from simulated residuals.)

  * cov(alpha_hat, beta_hat) -- the two effects are estimated in the SAME
    patients, so their estimation errors are correlated. Published aggregates
    never report this, which is why it had to be a sensitivity parameter. Here
    it is estimated directly by a nonparametric bootstrap over patients, and
    fed to Stage 2 as a known within-trial covariance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = Path(__file__).resolve().parents[1] / "output"
B_BOOT = 1000
RNG = np.random.default_rng(20260928)
LOG: list[str] = []

# Which arm is "treatment" (coded 1) in each trial, and how to label the trial.
ARM_TRT = {
    "CITY": "CGM", "WISDM": "CGM", "DCLP3": "CLC", "PEDAP": "CLC",
    "ReplaceBG": "CGM Only", "RT_CGM": "RT-CGM",
}
POPULATION = {
    "CITY": "T1D_adolescent_youngadult", "WISDM": "T1D_older_adult",
    "DCLP3": "T1D_mixed_age", "PEDAP": "T1D_young_children",
    "ReplaceBG": "T1D_adult", "RT_CGM": "T1D_mixed_age",
}
# Minimum CGM coverage for a patient-window to count as measured.
#
# This is NOT a free choice. CGM coverage is measured AFTER randomisation and is
# itself affected by treatment (an arm handed a real-time CGM wears it more than
# an arm handed a blinded recorder). Screening patients on coverage therefore
# conditions on a post-randomisation variable and breaks the randomisation that
# makes Stage 1 unbiased -- it is the informative-missingness problem, not a fix
# for it. Observed by running both floors on these data: in RT-CGM the 0.70 floor
# retained 79/228 of the treatment arm versus 135/219 of the control arm and
# moved the fitted TIR effect from +0.68 pp (SE 1.46, n=351) at the
# computability floor to -0.16 pp (SE 1.81, n=214); in WISDM it left 0 of 100
# control patients analysable, dropping that trial from the analysis entirely.
#
# Primary analysis therefore uses a bare COMPUTABILITY floor (enough readings to
# form the metric at all, 24 h), with the consensus 0.70 floor run as a
# pre-specified sensitivity analysis. Set via CLI:  stage1_effects.py 0.70
COVERAGE_FLOOR = 24 / (14 * 24)   # 24 h of readings in a 14-day window ~= 0.071


def log(m: str) -> None:
    LOG.append(m)
    print(m, flush=True)


def build() -> pd.DataFrame:
    m = pd.read_csv(OUT / "cgm_metrics.csv")
    m = m[m["trial"] != "RT_CGM"]
    rt = OUT / "rtcgm_episode_metrics.csv"
    if rt.exists():
        r = pd.read_csv(rt)
        m = pd.concat([m, r], ignore_index=True)
        log(f"RT-CGM episode metrics merged: {len(r)} rows")
    else:
        log("RT-CGM episode metrics absent -- trial excluded")

    a1c = pd.read_csv(OUT / "a1c_endpoints.csv")

    wide = m.pivot_table(index=["trial", "PtID", "arm"], columns="period",
                         values=["TIR", "coverage", "gluCV", "TBR70", "TAR180"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(a1c, on=["trial", "PtID"], how="left")

    wide["trt"] = [1 if a == ARM_TRT.get(t) else 0
                   for t, a in zip(wide["trial"], wide["arm"])]
    wide["population"] = wide["trial"].map(POPULATION)

    # measurement-quality gate, applied symmetrically to both arms
    wide["cgm_ok"] = (wide["coverage_baseline"].ge(COVERAGE_FLOOR)
                      & wide["coverage_followup"].ge(COVERAGE_FLOOR))
    wide["analysable"] = (wide["cgm_ok"]
                          & wide[["TIR_baseline", "TIR_followup",
                                  "a1c_bl", "a1c_fu"]].notna().all(axis=1))
    wide.to_csv(OUT / "stage1_patient.csv", index=False)

    log("\nPatients passing each gate, by trial and arm:")
    for t, g in wide.groupby("trial"):
        tab = g.groupby("arm").agg(
            n=("PtID", "size"),
            cgm_cov_ok=("cgm_ok", "sum"),
            a1c_both=("a1c_fu", lambda s: s.notna().sum()),
            analysable=("analysable", "sum"))
        log(f"  {t}\n{tab.to_string()}")
    return wide


def fit_one(d: pd.DataFrame):
    """ANCOVA for both endpoints; returns effects, SEs, residual correlation."""
    Xs = sm.add_constant(d[["trt", "TIR_baseline"]].astype(float))
    Xt = sm.add_constant(d[["trt", "a1c_bl"]].astype(float))
    fs = sm.OLS(d["TIR_followup"].astype(float), Xs).fit()
    ft = sm.OLS(d["a1c_fu"].astype(float), Xt).fit()
    r_indiv = float(np.corrcoef(fs.resid, ft.resid)[0, 1])
    return (float(fs.params["trt"]), float(ft.params["trt"]),
            float(fs.bse["trt"]), float(ft.bse["trt"]), r_indiv)


def stage1(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in wide.groupby("trial"):
        d = g[g["analysable"]].copy()
        n_trt, n_ctl = int(d["trt"].sum()), int((1 - d["trt"]).sum())
        if n_trt < 5 or n_ctl < 5:
            log(f"  {t}: SKIPPED (analysable n per arm = {n_trt}/{n_ctl})")
            continue
        a, b, sa, sb, r_ind = fit_one(d)

        # bootstrap over patients, stratified by arm, for cov(alpha_hat, beta_hat)
        boot = []
        idx_t = d.index[d["trt"] == 1].to_numpy()
        idx_c = d.index[d["trt"] == 0].to_numpy()
        for _ in range(B_BOOT):
            take = np.concatenate([RNG.choice(idx_t, idx_t.size, replace=True),
                                   RNG.choice(idx_c, idx_c.size, replace=True)])
            try:
                aa, bb, *_ = fit_one(d.loc[take])
                boot.append((aa, bb))
            except Exception:
                continue
        boot = np.asarray(boot)
        cov_ab = float(np.cov(boot[:, 0], boot[:, 1])[0, 1]) if len(boot) > 50 else np.nan
        r_ab = (cov_ab / (boot[:, 0].std(ddof=1) * boot[:, 1].std(ddof=1))
                if len(boot) > 50 else np.nan)

        rows.append(dict(
            trial_id=t, population=POPULATION[t], source="published_RCT",
            n_total=len(d), n_trt=n_trt, n_ctl=n_ctl,
            effect_tir=a, se_tir=sa, effect_a1c=b, se_a1c=sb,
            se_tir_boot=float(boot[:, 0].std(ddof=1)),
            se_a1c_boot=float(boot[:, 1].std(ddof=1)),
            cov_tir_a1c=cov_ab, r_within_tir_a1c=float(r_ab),
            r_indiv=r_ind, R2_indiv=r_ind ** 2,
            baseline_tir_mean=float(d["TIR_baseline"].mean()),
            baseline_a1c_mean=float(d["a1c_bl"].mean()),
            cov_cgm_median=float(pd.concat([d["coverage_baseline"],
                                            d["coverage_followup"]]).median()),
            cov_gap_trt_minus_ctl=float(
                d.loc[d.trt == 1, "coverage_followup"].median()
                - d.loc[d.trt == 0, "coverage_followup"].median()),
        ))
        log(f"  {t}: n={len({*d.PtID})} dTIR={a:+.2f} (SE {sa:.2f}) "
            f"dA1c={b:+.3f} (SE {sb:.3f}) r_within={r_ab:+.3f} R2_indiv={r_ind**2:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    import sys as _sys
    global COVERAGE_FLOOR
    tag = "primary"
    if len(_sys.argv) > 1:
        COVERAGE_FLOOR = float(_sys.argv[1])
        tag = f"cov{int(round(COVERAGE_FLOOR * 100)):02d}"
    log(f"COVERAGE_FLOOR = {COVERAGE_FLOOR:.4f}  (tag={tag})")

    wide = build()
    log("\nStage 1 ANCOVA per trial:")
    eff = stage1(wide)
    eff["coverage_floor"] = COVERAGE_FLOOR
    eff.to_csv(OUT / f"trial_effects_{tag}.csv", index=False)
    if tag == "primary":
        eff.to_csv(OUT / "trial_effects.csv", index=False)
    (OUT / f"stage1_log_{tag}.md").write_text(
        "# Stage 1 log\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    print("\n" + eff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

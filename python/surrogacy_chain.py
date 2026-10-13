"""Chain the trial-level TIR -> HbA1c slope with the trial-level HbA1c -> microvascular-risk slope.

Link 1 : from stage2_trial_level.py (21 trials).
Link 2 : weighted regression THROUGH THE ORIGIN of log(risk ratio) on achieved HbA1c difference, across
         four glycaemic-control trials with adjudicated microvascular endpoints
         (results/tables/hba1c_microvascular_link.csv; every value carries a verbatim source quotation).
         Through the origin because a trial with no HbA1c separation has, by construction, no
         glycaemia-mediated effect.
Chain  : RR per +10 pp TIR = exp(10 * link1 * link2); delta-method SE, the two links being estimated from
         disjoint trial sets and therefore independent.
Comparator: the published individual-level DCCT seven-point-profile association (Beck et al. 2019,
         doi:10.2337/dc18-1444), 64% higher hazard per 10 pp LOWER TIR.

Output: results/tables/surrogacy_chain.csv, surrogacy_chain_sensitivity.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

Z = 1.959964
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAB = os.path.join(ROOT, "results", "tables")
BECK = (np.log(1 / 1.64), np.log(1 / 1.78), np.log(1 / 1.51))   # point, lower, upper per +10 pp TIR


def fit_link2(d, random_effects=False):
    """Weighted through-origin slope of log RR on achieved HbA1c difference."""
    if random_effects:
        def nll(lt):
            w = 1 / (d.se_log_rr ** 2 + np.exp(2 * lt))
            b = np.sum(w * d.log_rr * d.d_HbA1c) / np.sum(w * d.d_HbA1c ** 2)
            return 0.5 * np.sum(np.log(d.se_log_rr ** 2 + np.exp(2 * lt))
                                + w * (d.log_rr - b * d.d_HbA1c) ** 2)
        tau2 = np.exp(2 * minimize_scalar(nll, bounds=(-8, 1), method="bounded").x)
        w = 1 / (d.se_log_rr ** 2 + tau2)
    else:
        w = 1 / d.se_log_rr ** 2
    b = float(np.sum(w * d.log_rr * d.d_HbA1c) / np.sum(w * d.d_HbA1c ** 2))
    se = float(np.sqrt(1 / np.sum(w * d.d_HbA1c ** 2)))
    Q = float(np.sum((1 / d.se_log_rr ** 2) * (d.log_rr - b * d.d_HbA1c) ** 2))
    return b, se, Q


def chain(link1, se1, link2, se2):
    """log RR per +10 pp TIR and its delta-method SE."""
    est = link2 * link1 * 10
    se = np.sqrt((link2 * 10) ** 2 * se1 ** 2 + (link1 * 10) ** 2 * se2 ** 2)
    return est, se


def main():
    L2 = pd.read_csv(os.path.join(TAB, "hba1c_microvascular_link.csv"))
    U = L2[L2.use]
    b2, s2, Q = fit_link2(U)
    df = len(U) - 1
    print(f"link 2: {b2:.4f} (SE {s2:.4f}) log-RR per 1% HbA1c  ->  RR {np.exp(-b2):.3f} per 1% lower; "
          f"Q {Q:.2f} on {df} df, p {1 - chi2.cdf(Q, df):.3f}")

    s2tab = pd.read_csv(os.path.join(TAB, "stage2_by_subset.csv")).set_index("set")
    a1 = s2tab.loc["Primary (IPD + published)", "wls_slope_per10pp"] / 10
    sa1 = s2tab.loc["Primary (IPD + published)", "wls_se_per10pp"] / 10

    est, se = chain(a1, sa1, b2, s2)
    pd.DataFrame([
        dict(quantity="link2_logRR_per_1pct_HbA1c", est=b2, se=s2, k=len(U)),
        dict(quantity="link1_HbA1c_pct_per_1pp_TIR", est=a1, se=sa1, k=int(s2tab.loc["Primary (IPD + published)", "k"])),
        dict(quantity="chain_logRR_per_10pp_TIR", est=est, se=se, k=np.nan),
        dict(quantity="beck2019_individual_logHR_per_10pp_TIR", est=BECK[0],
             se=(BECK[2] - BECK[1]) / (2 * Z), k=1440),
    ]).to_csv(os.path.join(TAB, "surrogacy_chain.csv"), index=False)
    print(f"chain: RR {np.exp(est):.3f} ({np.exp(est - Z * se):.3f}-{np.exp(est + Z * se):.3f}) per +10 pp TIR; "
          f"individual-level RR {np.exp(BECK[0]):.3f}; ratio of log effects {BECK[0] / est:.1f}")

    variants2 = {
        "4 trials, primary microvascular outcome": (U, False),
        "DCCT microalbuminuria instead of retinopathy":
            (pd.concat([L2[L2.trial.str.contains("microalbuminuria")], U[~U.trial.str.startswith("DCCT")]]), False),
        "T2D trials only (no DCCT)": (U[~U.trial.str.startswith("DCCT")], False),
        "retinopathy outcomes only (DCCT, ACCORD)":
            (U[U.trial.str.contains("DCCT|ACCORD")], False),
        "4 trials, random effects": (U, True),
    }
    variants1 = {"all 21 trials": (a1, sa1),
                 "HbA1c-focused trials": (s2tab.loc["Primary, HbA1c-focused trials", "wls_slope_per10pp"] / 10,
                                          s2tab.loc["Primary, HbA1c-focused trials", "wls_se_per10pp"] / 10),
                 "IPD only (k=6)": (s2tab.loc["IPD only", "wls_slope_per10pp"] / 10,
                                    s2tab.loc["IPD only", "wls_se_per10pp"] / 10)}
    rows = []
    for n2, (d2, re_) in variants2.items():
        bb, ss, _ = fit_link2(d2, random_effects=re_)
        for n1, (aa, sa) in variants1.items():
            e, s = chain(aa, sa, bb, ss)
            rows.append(dict(link2=n2, k2=len(d2), link1=n1, rr_per_1pct_hba1c=np.exp(-bb),
                             chain_RR_per_10pp_TIR=np.exp(e), chain_lo=np.exp(e - Z * s),
                             chain_hi=np.exp(e + Z * s), pct_risk_reduction=100 * (1 - np.exp(e)),
                             ratio_vs_Beck=BECK[0] / e))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "surrogacy_chain_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()

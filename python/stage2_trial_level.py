"""Stage 2: trial-level surrogacy of time in range (and time above range) for HbA1c.

Input : results/tables/trial_level_dataset.csv
        columns trial, population, d_TIR, se_TIR, d_A1C, se_A1C, d_TAR, se_TAR, r_w, source, include, design
        d_* are treatment effects (intervention minus control); r_w is the within-trial correlation between
        the two effect estimates (median of the IPD trials for published trials, where it is not reported).
Output: results/tables/stage2_by_subset.csv, tar_vs_tir.csv, clump_test.csv, leave_one_out.csv

Model (Buyse & Molenberghs 1998):
    (a_i, b_i)' ~ N( (mu_a, mu_b)', Sigma_i + D ),   D = [[t_a^2, rho t_a t_b], [rho t_a t_b, t_b^2]]
with Sigma_i the known within-trial covariance. R2_trial = rho^2; the CI for rho is the profile likelihood,
which is the right choice because the estimate can sit on the boundary at small k.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import t as tdist

Z = 1.959964
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAB = os.path.join(ROOT, "results", "tables")
HYPO = "IMPACT|IN CONTROL|HypoDE|SMILE|geriatric|WISDM"   # trials whose stated aim was hypoglycaemia


def biv_nll(theta, d, xcol="d_TIR", sxcol="se_TIR"):
    ta, tb, rho = np.exp(theta[0]), np.exp(theta[1]), np.tanh(theta[2])
    D = np.array([[ta ** 2, rho * ta * tb], [rho * ta * tb, tb ** 2]])
    Y = d[[xcol, "d_A1C"]].to_numpy()
    a, b, r = d[sxcol].to_numpy(), d.se_A1C.to_numpy(), d.r_w.to_numpy()
    S = np.stack([np.stack([a ** 2, r * a * b], -1), np.stack([r * a * b, b ** 2], -1)], 1) + D
    Si = np.linalg.inv(S)
    mu = np.linalg.solve(Si.sum(0), np.einsum("kij,kj->i", Si, Y))
    e = Y - mu
    return 0.5 * np.sum(np.linalg.slogdet(S)[1] + np.einsum("ki,kij,kj->k", e, Si, e))


def biv_fit(d, **kw):
    best = None
    for s in [(1, -2, -1), (2, -1, -2), (0.5, -3, 0), (1.5, -1.5, -0.5)]:
        r = minimize(lambda th: biv_nll(th, d, **kw), np.array(s, float), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    return dict(nll=best.fun, tau_x=np.exp(th[0]), tau_a1c=np.exp(th[1]), rho=np.tanh(th[2]), th=th)


def rho_profile(d, ngrid=81, **kw):
    """95% profile-likelihood interval for rho (deviance cut-off 3.84)."""
    f = biv_fit(d, **kw)
    grid = np.linspace(-0.995, 0.995, ngrid)
    dev = []
    for rv in grid:
        r = minimize(lambda t2: biv_nll(np.r_[t2, np.arctanh(rv)], d, **kw), f["th"][:2],
                     method="Nelder-Mead", options=dict(maxiter=800, xatol=1e-4, fatol=1e-6))
        dev.append(2 * (r.fun - f["nll"]))
    ok = grid[np.array(dev) <= 3.84]
    return f, (ok.min(), ok.max())


def wls(d, xcol="d_TIR"):
    """Inverse-variance weighted regression of the HbA1c effect on the surrogate effect."""
    w = (1 / d.se_A1C ** 2).to_numpy()
    x, y = d[xcol].to_numpy(), d.d_A1C.to_numpy()
    X = np.c_[np.ones(len(d)), x]
    A = X.T @ (X * w[:, None])
    b = np.linalg.solve(A, X.T @ (w * y))
    s2 = np.sum(w * (y - X @ b) ** 2) / (len(d) - 2)
    return b[1], float(np.sqrt(s2 * np.linalg.inv(A)[1, 1])), float(np.corrcoef(x, y)[0, 1])


def metareg(d, xcol="d_TIR"):
    """Random-effects meta-regression; returns coefficients, covariance and tau^2."""
    x, y, v = d[xcol].to_numpy(), d.d_A1C.to_numpy(), (d.se_A1C ** 2).to_numpy()
    X = np.c_[np.ones(len(x)), x]

    def nll(lt):
        w = 1 / (v + np.exp(2 * lt))
        A = X.T @ (X * w[:, None])
        b = np.linalg.solve(A, X.T @ (w * y))
        return 0.5 * (np.sum(np.log(v + np.exp(2 * lt))) + np.sum(w * (y - X @ b) ** 2)
                      + np.linalg.slogdet(A)[1])

    t2 = np.exp(2 * minimize_scalar(nll, bounds=(-8, 1), method="bounded").x)
    w = 1 / (v + t2)
    A = X.T @ (X * w[:, None])
    C = np.linalg.inv(A)
    return C @ X.T @ (w * y), C, t2


def surrogate_threshold_effect(d, xcol="d_TIR"):
    b, C, t2 = metareg(d, xcol)
    tq = tdist.ppf(0.975, len(d) - 2)
    xs = np.linspace(0, 40, 4001)
    hi = b[0] + b[1] * xs + tq * np.sqrt(C[0, 0] + 2 * xs * C[0, 1] + xs ** 2 * C[1, 1] + t2)
    return float(xs[np.argmax(hi < 0)]) if np.any(hi < 0) else np.nan


def main():
    d = pd.read_csv(os.path.join(TAB, "trial_level_dataset.csv"))
    d["T2D"] = d.population.fillna("").str.contains("T2D")
    d["hypo"] = d.trial.str.contains(HYPO)
    prim = d[d.include == "primary"].reset_index(drop=True)

    sets = {"Primary (IPD + published)": prim,
            "IPD only": d[d.source == "IPD"],
            "+ CONCEPTT pregnancy": d,
            "Primary, T1D only": prim[~prim.T2D],
            "Primary, T2D only": prim[prim.T2D],
            "Primary, no approximate SEs": prim[~prim.quality.fillna("").str.contains("approx")],
            "Primary, parallel-group only": prim[prim.design != "crossover"],
            "Primary, HbA1c-focused trials": prim[~prim.hypo],
            "Primary, hypoglycaemia-focused trials": prim[prim.hypo]}
    rows = []
    for name, s in sets.items():
        s = s.reset_index(drop=True)
        b, se, r = wls(s)
        if len(s) >= 6:
            f, ci = rho_profile(s)
        else:
            f, ci = biv_fit(s), (np.nan, np.nan)
        rows.append(dict(set=name, k=len(s), k_T2D=int(s.T2D.sum()), wls_slope_per10pp=10 * b,
                         wls_se_per10pp=10 * se, pearson_r=r, rho_trial=f["rho"],
                         rho_ci_lo=ci[0], rho_ci_hi=ci[1], R2_trial=f["rho"] ** 2))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "stage2_by_subset.csv"), index=False)

    # time above range on the trials that report it, sign-flipped so that improvement is positive
    dT = prim.dropna(subset=["d_TAR"]).reset_index(drop=True)
    flip = dT.assign(d_TIR=-dT.d_TAR, se_TIR=dT.se_TAR)
    fT, ciT = rho_profile(flip)
    fI, ciI = rho_profile(dT)
    bT, seT, rT = wls(flip)
    bI, seI, rI = wls(dT)
    pd.DataFrame([
        dict(set=f"TAR as surrogate (k={len(dT)})", k=len(dT), wls_slope_per10pp=-10 * bT,
             wls_se_per10pp=10 * seT, pearson_r=-rT, rho_trial=-fT["rho"],
             rho_ci=f"{-ciT[1]:.2f} to {-ciT[0]:.2f}"),
        dict(set=f"TIR as surrogate, same {len(dT)} trials", k=len(dT), wls_slope_per10pp=10 * bI,
             wls_se_per10pp=10 * seI, pearson_r=rI, rho_trial=fI["rho"],
             rho_ci=f"{ciI[0]:.2f} to {ciI[1]:.2f}")
    ]).to_csv(os.path.join(TAB, "tar_vs_tir.csv"), index=False)

    # is the association carried by the contrast between effective and ineffective trials?
    cl = []
    for lab, s in [("dTIR > 3 pp", prim[prim.d_TIR > 3]), ("dTIR > 5 pp", prim[prim.d_TIR > 5]),
                   ("dTIR <= 5 pp", prim[prim.d_TIR <= 5])]:
        s = s.reset_index(drop=True)
        b, se, r = wls(s)
        cl.append(dict(subset=lab, k=len(s), wls_slope_per10pp=10 * b, wls_se_per10pp=10 * se, pearson_r=r))
    pd.DataFrame(cl).to_csv(os.path.join(TAB, "clump_test.csv"), index=False)

    loo = []
    for i in range(len(prim)):
        s = prim.drop(i).reset_index(drop=True)
        f = biv_fit(s)
        b, _, r = wls(s)
        loo.append(dict(dropped=prim.trial[i], rho=f["rho"], slope_per10pp=10 * b, r=r))
    pd.DataFrame(loo).to_csv(os.path.join(TAB, "leave_one_out.csv"), index=False)

    print(f"k = {len(prim)}; STE = {surrogate_threshold_effect(prim):.1f} pp")


if __name__ == "__main__":
    main()

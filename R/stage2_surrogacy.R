## stage2_surrogacy.R -- trial-level surrogacy from aggregate trial data
##
## Run after python/validate_trial_table.py has written
## output/trial_level_clean.csv.
##
##   Rscript r/stage2_surrogacy.R
##
## Estimators, in the order they should be read:
##
##   (A) Bivariate random-effects meta-analysis  <- PRIMARY
##       Burzykowski, Molenberghs & Buyse (2005), ch. 7, "reduced" bivariate
##       model; equivalently Daniels & Hughes (1997) / van Houwelingen et al.
##       (2002). Models (alpha_i, beta_i) as draws from a bivariate normal with
##       between-trial covariance Omega, given each trial's known within-trial
##       sampling covariance. R2_trial = rho(Omega)^2. This is the estimator
##       that belongs in the abstract: it separates between-trial signal from
##       within-trial sampling noise, so it does NOT attenuate as the naive
##       regression does.
##
##   (B) Naive WLS of beta_hat on alpha_hat  <- for comparison only
##       What the first-draft pipeline reported. Biased toward zero because
##       alpha_hat carries sampling error; reported here alongside its
##       method-of-moments attenuation correction so the size of that bias is
##       visible rather than implicit.
##
##   (C) Rank-based association
##       Weighted Spearman. Invariant to monotone re-scaling of either
##       endpoint, so it is unaffected by whether TIR is modelled raw, logit or
##       arcsine. Measures monotone rather than linear predictiveness.
##
##   (D) Huber M-estimator of the Stage-2 slope
##       Guards against a single small or device-atypical trial dominating.
##
## All intervals are cluster bootstrap over `cluster_id`, so multi-arm trials
## contributing several rows are resampled as a unit.
##
## NOT COMPUTED, deliberately: R2_indiv. It requires the within-trial residual
## covariance of TIR and A1c in the same patients, which aggregate data cannot
## supply. See PIPELINE_AUDIT.md, finding 1.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(metafor); library(boot); library(MASS)
})
select <- dplyr::select

OUT <- "output"
dir.create(OUT, showWarnings = FALSE)
set.seed(20260928)

dat_all <- read_csv(file.path(OUT, "trial_level_clean.csv"), show_col_types = FALSE)
dat  <- dat_all |> filter(usable_weighted)
datu <- dat_all |> filter(usable_unweighted)

msg <- function(...) cat(sprintf(...), "\n")
msg("Trials usable weighted: %d ; unweighted: %d ; clusters: %d",
    nrow(dat), nrow(datu), dplyr::n_distinct(dat$cluster_id))
if (nrow(dat) < 4)
  msg("** N < 4 trials. Every estimate below is descriptive only. **")

## ---------------------------------------------------------------------------
## (A) Bivariate random-effects meta-analysis
## ---------------------------------------------------------------------------
## The within-trial correlation between the TIR effect and the A1c effect is
## NOT identified by published aggregates -- both are estimated in the same
## patients, so it is non-zero, but no paper reports it. It is therefore a
## sensitivity parameter, not a number to assume once. r_w < 0 is the
## physiologically expected direction (a patient-level increase in TIR
## accompanies a decrease in A1c).
## r_within = NULL uses each trial's OWN bootstrap-estimated covariance between
## the two effect estimates (column cov_tir_a1c from Stage 1). A numeric value
## overrides it with a common assumed correlation, for sensitivity analysis.
fit_bivariate <- function(d, r_within = NULL) {
  k <- nrow(d)
  long <- do.call(rbind, lapply(seq_len(k), function(i) data.frame(
    trial   = d$trial_id[i],
    outcome = factor(c("S", "T"), levels = c("S", "T")),
    yi      = c(d$effect_tir[i], d$effect_a1c[i])
  )))
  Vlist <- lapply(seq_len(k), function(i) {
    s1 <- d$se_tir[i]; s2 <- d$se_a1c[i]
    cv <- if (is.null(r_within)) d$cov_tir_a1c[i] else r_within * s1 * s2
    if (!is.finite(cv)) cv <- 0
    ## keep the 2x2 block positive definite
    cv <- sign(cv) * min(abs(cv), 0.99 * s1 * s2)
    matrix(c(s1^2, cv, cv, s2^2), 2, 2)
  })
  V <- metafor::bldiag(Vlist)
  tryCatch({
    f <- rma.mv(yi, V, mods = ~ outcome - 1,
                random = ~ outcome | trial, struct = "UN",
                data = long, method = "REML",
                control = list(optimizer = "nlminb", rel.tol = 1e-8))
    rho <- as.numeric(f$rho)
    tibble(r_within = r_within,
           tau_S = sqrt(f$tau2[1]), tau_T = sqrt(f$tau2[2]),
           rho_between = rho, R2_trial = rho^2,
           ## slope of the trial-level regression implied by Omega
           slope_implied = rho * sqrt(f$tau2[2]) / sqrt(f$tau2[1]),
           boundary = abs(rho) > 0.99 ||
                      min(f$tau2) < 1e-8,
           converged = TRUE)
  }, error = function(e) tibble(
    r_within = r_within, tau_S = NA_real_, tau_T = NA_real_,
    rho_between = NA_real_, R2_trial = NA_real_, slope_implied = NA_real_,
    boundary = NA, converged = FALSE))
}

has_cov <- "cov_tir_a1c" %in% names(dat) && any(is.finite(dat$cov_tir_a1c))
primary <- fit_bivariate(dat, NULL) |> mutate(r_within = NA_real_,
                                              spec = "estimated per trial")
r_grid <- c(0, -0.2, -0.4, -0.6)
biv <- bind_rows(primary,
                 bind_rows(lapply(r_grid, function(r) fit_bivariate(dat, r))) |>
                   mutate(spec = sprintf("assumed r_within = %.1f", r_grid)))
write_csv(biv, file.path(OUT, "stage2_bivariate_sensitivity.csv"))
msg("(A) bivariate RE. Row 1 uses each trial's bootstrap-estimated within-trial")
msg("    covariance; the rest assume a common correlation (sensitivity only).")
print(as.data.frame(biv), row.names = FALSE)
if (!has_cov) {
  msg("** cov_tir_a1c absent: primary row fell back to independence (cv=0).")
}
if (any(biv$boundary, na.rm = TRUE)) {
  msg("** Boundary estimate (|rho|~1 or tau2~0) in at least one fit: with few")
  msg("** trials REML pins the between-trial correlation at the boundary, so")
  msg("** read the bootstrap interval, not the point estimate.")
}

## Individual-level surrogacy: now estimable, because Stage 1 had patient-level
## data. Sample-size weighted mean of the within-trial squared residual
## correlation between the two endpoints.
R2_indiv <- if ("R2_indiv" %in% names(dat))
  weighted.mean(dat$R2_indiv, w = dat$n_total, na.rm = TRUE) else NA_real_
msg("R2_indiv (n-weighted mean of within-trial residual r^2) = %.3f  [range %.3f-%.3f]",
    R2_indiv, min(dat$R2_indiv, na.rm = TRUE), max(dat$R2_indiv, na.rm = TRUE))

## ---------------------------------------------------------------------------
## (B) Naive WLS + attenuation correction
## ---------------------------------------------------------------------------
w_prec <- 1 / dat$se_a1c^2
fit_wls <- lm(effect_a1c ~ effect_tir, data = dat, weights = w_prec)
slope_naive <- unname(coef(fit_wls)[2])

## Method-of-moments reliability of alpha_hat: the share of observed variance in
## the TIR effects that is real between-trial variance rather than sampling
## noise. slope_corrected = slope_naive / lambda.
var_obs    <- var(dat$effect_tir)
mean_err   <- mean(dat$se_tir^2)
lambda_rel <- max(1e-6, (var_obs - mean_err) / var_obs)
slope_corr <- slope_naive / lambda_rel

## Pearson R2 of the weighted regression, and the metafor mixed-model pseudo-R2
## (proportional reduction in between-trial heterogeneity tau2 from adding the
## TIR effect as a moderator). Neither corrects for error in alpha_hat; they are
## upper/lower companions to (A), not replacements for it.
fit_rma0 <- rma(yi = effect_a1c, sei = se_a1c, data = dat, method = "REML")
fit_rma1 <- rma(yi = effect_a1c, sei = se_a1c, mods = ~ effect_tir,
                data = dat, method = "REML")
pseudo_R2 <- max(0, (fit_rma0$tau2 - fit_rma1$tau2) / fit_rma0$tau2)

msg("(B) naive WLS slope = %.4f ; reliability of alpha_hat = %.3f ; attenuation-corrected slope = %.4f",
    slope_naive, lambda_rel, slope_corr)
msg("    weighted-lm R2 = %.3f ; meta-regression pseudo-R2 (tau2 reduction) = %.3f",
    summary(fit_wls)$r.squared, pseudo_R2)

## ---------------------------------------------------------------------------
## (C) Rank-based, and (D) robust slope
## ---------------------------------------------------------------------------
wcor <- function(x, y, w) {
  mx <- weighted.mean(x, w); my <- weighted.mean(y, w)
  sum(w * (x - mx) * (y - my)) /
    sqrt(sum(w * (x - mx)^2) * sum(w * (y - my)^2))
}
w_size <- if (all(is.finite(datu$n_total))) datu$n_total else rep(1, nrow(datu))
rho_spear <- wcor(rank(datu$effect_tir), rank(datu$effect_a1c), w_size)

fit_rlm <- tryCatch(
  MASS::rlm(effect_a1c ~ effect_tir, data = dat, weights = w_prec,
            psi = MASS::psi.huber, k = 1.345, maxit = 100),
  error = function(e) NULL)
slope_huber <- if (is.null(fit_rlm)) NA_real_ else unname(coef(fit_rlm)[2])

msg("(C) weighted Spearman rho = %.3f  (rank R2 = %.3f)", rho_spear, rho_spear^2)
msg("(D) Huber slope = %.4f  (vs naive WLS %.4f)", slope_huber, slope_naive)

## ---------------------------------------------------------------------------
## Cluster bootstrap over trials
## ---------------------------------------------------------------------------
clusters <- unique(dat$cluster_id)
B <- 2000
boot_stat <- function(idx) {
  d <- bind_rows(lapply(clusters[idx], function(cl) dat[dat$cluster_id == cl, ]))
  if (nrow(d) < 3 || sd(d$effect_tir) == 0) return(c(NA, NA, NA))
  ww <- 1 / d$se_a1c^2
  sl <- unname(coef(lm(effect_a1c ~ effect_tir, data = d, weights = ww))[2])
  b  <- fit_bivariate(d, NULL)
  c(b$R2_trial, sl,
    wcor(rank(d$effect_tir), rank(d$effect_a1c), rep(1, nrow(d)))^2)
}
bs <- replicate(B, boot_stat(sample(seq_along(clusters), replace = TRUE)))
qci <- function(v) {
  v <- v[is.finite(v)]
  if (length(v) < 50) return(c(NA, NA))
  unname(quantile(v, c(0.025, 0.975)))
}
ci_R2biv <- qci(bs[1, ]); ci_slope <- qci(bs[2, ]); ci_R2rank <- qci(bs[3, ])
msg("Cluster bootstrap (B=%d, %d clusters): R2_trial(A) [%.3f, %.3f] ; slope [%.4f, %.4f] ; rank R2 [%.3f, %.3f]",
    B, length(clusters), ci_R2biv[1], ci_R2biv[2],
    ci_slope[1], ci_slope[2], ci_R2rank[1], ci_R2rank[2])

## ---------------------------------------------------------------------------
## Surrogate threshold effect, with between-trial heterogeneity included
## ---------------------------------------------------------------------------
## STE = the smallest TIR effect at which the UPPER 95% confidence limit of the
## predicted A1c effect is still below zero, i.e. the TIR gain a future trial
## would have to show before we could conclude it lowered A1c. Computed from
## the random-effects meta-regression so that tau2 enters the prediction, which
## the first-draft `lm`-based STE omitted.
ste <- NA_real_
if (nrow(dat) >= 5 && coef(fit_rma1)[2] < 0) {
  grid <- seq(0, max(dat$effect_tir, na.rm = TRUE) * 3, length.out = 2000)
  pr <- predict(fit_rma1, newmods = grid)
  ok <- which(pr$ci.ub < 0)
  if (length(ok)) ste <- grid[min(ok)]
}
msg("STE (TIR pp needed for upper 95%% CL of predicted dA1c < 0): %s",
    ifelse(is.na(ste), "not reached in range / N too small", sprintf("%.2f", ste)))

## ---------------------------------------------------------------------------
## Pre-specified moderators: data source (the regulatory-discrepancy aim) and
## population stratum
## ---------------------------------------------------------------------------
dat <- dat |> mutate(
  source_grp = if_else(grepl("^FDA", source), "FDA_application", "published_RCT"))
## Pre-specified moderator for the missingness workstream: the arm difference in
## CGM coverage. A trial whose arms were measured with different completeness has
## a TIR effect that is partly a measurement artefact.
if ("cov_gap_trt_minus_ctl" %in% names(dat) && nrow(dat) >= 5) {
  f_cov <- rma(yi = effect_tir, sei = se_tir,
               mods = ~ cov_gap_trt_minus_ctl, data = dat, method = "REML")
  msg("Coverage-asymmetry moderator on the TIR effect: slope = %.2f pp per unit coverage gap (p = %.3f)",
      coef(f_cov)[2], f_cov$pval[2])
  write_csv(as.data.frame(coef(summary(f_cov))),
            file.path(OUT, "stage2_coverage_moderator.csv"))
}
src_test <- tibble(test = "source moderator", statistic = NA_real_,
                   p = NA_real_, note = "not run")
if (all(table(dat$source_grp) >= 3) && dplyr::n_distinct(dat$source_grp) == 2) {
  f_int <- rma(yi = effect_a1c, sei = se_a1c,
               mods = ~ effect_tir * source_grp, data = dat, method = "REML")
  ## Wald test on the interaction: does the TIR->A1c slope differ between
  ## FDA-application trials and published RCTs?
  j <- grep(":", rownames(f_int$beta))
  if (length(j) == 1) {
    at <- anova(f_int, btt = j)
    src_test <- tibble(test = "slope differs FDA vs published",
                       statistic = as.numeric(at$QM), p = as.numeric(at$QMp),
                       note = sprintf("df=%d", at$m))
  }
} else {
  src_test$note <- sprintf("needs >=3 trials per source group; have %s",
                           paste(names(table(dat$source_grp)),
                                 table(dat$source_grp), collapse = ", "))
}
msg("Source moderator: %s (p = %s) -- %s", src_test$test,
    ifelse(is.na(src_test$p), "NA", sprintf("%.3f", src_test$p)), src_test$note)

pop_ok <- dat |> count(population) |> filter(n >= 3) |> pull(population)
by_pop <- if (length(pop_ok) == 0) {
  tibble(population = character(), k = integer(),
         slope_wls = numeric(), rho_spearman = numeric())
} else {
  dat |>
    filter(population %in% pop_ok) |>
    group_by(population) |>
    group_modify(~ tibble(
      k = nrow(.x),
      slope_wls = if (sd(.x$effect_tir) > 0)
        unname(coef(lm(effect_a1c ~ effect_tir, data = .x,
                       weights = 1 / .x$se_a1c^2))[2]) else NA_real_,
      rho_spearman = if (sd(.x$effect_tir) > 0)
        suppressWarnings(cor(.x$effect_tir, .x$effect_a1c,
                             method = "spearman")) else NA_real_)) |>
    ungroup()
}
if (nrow(by_pop) > 0) {
  msg("By population stratum (>=3 trials only):")
  print(as.data.frame(by_pop), row.names = FALSE)
} else msg("No population stratum has >=3 trials; stratified estimates omitted.")

## ---------------------------------------------------------------------------
## Outputs
## ---------------------------------------------------------------------------
results <- tibble(
  k_trials            = nrow(dat),
  k_clusters          = length(clusters),
  R2_trial_bivariate  = primary$R2_trial,
  R2_trial_biv_lo     = ci_R2biv[1],
  R2_trial_biv_hi     = ci_R2biv[2],
  rho_between         = primary$rho_between,
  boundary_fit        = primary$boundary,
  R2_trial_rank       = rho_spear^2,
  R2_rank_lo          = ci_R2rank[1],
  R2_rank_hi          = ci_R2rank[2],
  R2_weighted_lm      = summary(fit_wls)$r.squared,
  pseudo_R2_metareg   = pseudo_R2,
  slope_naive_wls     = slope_naive,
  slope_lo            = ci_slope[1],
  slope_hi            = ci_slope[2],
  reliability_alpha   = lambda_rel,
  slope_attenuation_corrected = slope_corr,
  slope_huber         = slope_huber,
  STE_tir_pp          = ste,
  R2_indiv            = R2_indiv,
  R2_indiv_note       = "n-weighted mean of within-trial residual correlation^2"
)
write_csv(results, file.path(OUT, "stage2_results.csv"))
write_csv(src_test, file.path(OUT, "stage2_source_moderator.csv"))
write_csv(by_pop,   file.path(OUT, "stage2_by_population.csv"))

p <- ggplot(dat, aes(effect_tir, effect_a1c)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey80", linewidth = 0.3) +
  geom_errorbar(aes(ymin = effect_a1c - 1.96 * se_a1c,
                    ymax = effect_a1c + 1.96 * se_a1c),
                width = 0, colour = "grey60", linewidth = 0.3) +
  geom_errorbar(aes(xmin = effect_tir - 1.96 * se_tir,
                    xmax = effect_tir + 1.96 * se_tir),
                orientation = "y", width = 0, colour = "grey60", linewidth = 0.3) +
  geom_abline(intercept = unname(coef(fit_wls)[1]), slope = slope_naive,
              linewidth = 0.5, linetype = "dashed") +
  geom_point(aes(size = 1 / se_a1c^2, fill = trial_id),
             shape = 21, colour = "white", stroke = 0.4, alpha = 0.9) +
  scale_size_area(max_size = 9, guide = "none") +
  labs(x = "Treatment effect on Time in Range (percentage points)",
       y = "Treatment effect on HbA1c (%)",
       fill = "Trial",
       title = sprintf("Trial-level surrogacy of TIR for HbA1c (k = %d trials)", nrow(dat)),
       subtitle = sprintf("Bivariate RE R2_trial = %.2f [%.2f, %.2f]; dashed line = naive WLS (slope %.3f)",
                          primary$R2_trial, ci_R2biv[1], ci_R2biv[2], slope_naive),
       caption = "Bars are 95% CIs on each trial effect. Point area is inverse-variance weight.") +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        plot.title.position = "plot")
ggsave(file.path(OUT, "trial_level_surrogacy.png"), plot = p,
       width = 7.5, height = 5.2, dpi = 200)

msg("Wrote output/stage2_results.csv and output/trial_level_surrogacy.png")
print(as.data.frame(results), row.names = FALSE)

# Time in range as a surrogate for HbA1c and for microvascular outcomes

Analysis code, derived data and manuscript for a two-stage meta-analysis of 21 randomised trials, with a
published-evidence surrogacy chain extending the evaluation to microvascular complications.

**Rejaul Karim**¹ and **Sumaeya Sultana**²
¹ Independent Researcher · ² Cheeloo College of Medicine, Shandong University, Jinan, China

> **Status: not yet published.** This repository has not been pushed to a public host, and the manuscript has
> not been submitted or peer reviewed. See *Publishing this repository* below.

## Headline results

| Result | Value |
|---|---|
| Trial-level surrogacy of TIR for HbA1c (k = 21) | ρ_trial −0.63 (95% CI −0.87 to −0.15), R²_trial 0.40 |
| Slope | −0.34% HbA1c (SE 0.11) per +10 pp TIR |
| Surrogate threshold effect | 9.2 pp |
| Hyperglycaemia-targeting trials (k = 15) | ρ_trial −0.92 |
| Hypoglycaemia-targeting trials (k = 6) | ρ_trial −0.50 |
| Trials achieving > 5 pp TIR gain (k = 12) | slope +0.14 per 10 pp (95% CI −0.41 to +0.69) |
| Time above range, same 13 trials | ρ_trial 0.89 (0.50 to 0.99) vs TIR −0.65 (−0.90 to −0.07) |
| Chain: microvascular risk per +10 pp TIR | RR 0.90 (0.84 to 0.97) |
| Published individual-level association | RR 0.61 (0.56 to 0.66) |
| Ratio of log effects | 4.7 (3.3–5.7 across sensitivity analyses) |

## Layout

```
paper/            main.tex, references.bib, figures/
python/           analysis scripts (see below)
R/                stage2_surrogacy.R (earlier six-trial implementation, retained for reference)
data/raw/         empty; place the Jaeb archives here (not redistributed)
results/tables/   every table the manuscript cites
results/figures/  figures 1 and 2
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
python python/stage2_trial_level.py     # trial-level surrogacy, subsets, TAR, clump test, leave-one-out
python python/surrogacy_chain.py        # link 2 and the chain, with sensitivity analyses
```

Both run in under three minutes on a laptop and regenerate every number in the manuscript from
`results/tables/trial_level_dataset.csv` and `results/tables/hba1c_microvascular_link.csv`. Expected console
output:

```
k = 21; STE = 9.2 pp
link 2: 0.3053 (SE 0.0458) log-RR per 1% HbA1c  ->  RR 0.737 per 1% lower; Q 4.49 on 3 df, p 0.213
chain: RR 0.900 (0.839-0.966) per +10 pp TIR; individual-level RR 0.610; ratio of log effects 4.7
```

The stage-1 scripts (`python/stage1_effects.py`, `build_cgm_metrics.py`, `rescue_rtcgm.py`) regenerate the
six individual-patient-data trial effects from the Jaeb archives; they require those archives in `data/raw/`.

## Data provenance

- **Six trials with individual patient data** — CITY, DCLP3, PEDAP, Replace-BG, RT-CGM, WISDM — from the Jaeb
  Center for Health Research public archive, <https://public.jaeb.org>. Registration required; **not
  redistributed here**.
- **Fifteen published trials** — effect estimates extracted from the primary publications. Every value in
  `results/tables/published_extraction.csv` carries a verbatim quotation from its source, the unit conversion
  applied, the confidence level reported, and a flag where a standard error had to be approximated.
  `published_not_extracted.csv` records trials screened out and why.
- **Four outcome trials** for link 2 — DCCT, UKPDS 33, ACCORD Eye, ADVANCE — in
  `results/tables/hba1c_microvascular_link.csv`, also with verbatim quotations.
- **Seven-point TIR calibration** from an independent public CGM cohort, in
  `results/tables/seven_point_tir_calibration.csv`.

## Building the manuscript

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/main.html` is a self-contained reading copy (MathJax) for anyone without a TeX installation. It is a
convenience rendering, not the submission artefact.

## Publishing this repository

Nothing here has been pushed anywhere. To publish:

```bash
git init && git add . && git commit -m "Initial release"
gh repo create tir-surrogacy-paper --public --source=. --push     # or create the repo on github.com
```

Then (1) add the resulting URL to `CITATION.cff` under `repository-code:`, (2) archive the release on Zenodo
to obtain a DOI, and (3) replace the placeholder sentence in the manuscript's Data sharing section with that
DOI. Lancet journals require the data-sharing statement to describe what is actually available at the time of
submission, so do not cite a URL before the repository exists.

## Licence

- Code (`python/`, `R/`): MIT — see `LICENSE`.
- Manuscript text, figures and derived tables: CC BY 4.0 — see `LICENSE-CONTENT`.
- Source trial data are not covered by either licence and remain subject to their original terms.

## Citation

See `CITATION.cff`.

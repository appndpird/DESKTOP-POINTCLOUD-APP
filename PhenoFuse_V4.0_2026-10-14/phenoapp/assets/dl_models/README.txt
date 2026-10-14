PhenoApp v3.3.3 pretrained two-stream ensembles (2026-10-12): {variant}_{target}_fold{k}.pt, k = 0..4, float16 state dicts.
Trained with folds GROUPED BY PLOT (both flights of a plot always in the same fold) and the common band policy: common_bands.csv lists the 111 bands
(common_95 = 1) used for every plot; the runner applies it automatically when this file is present.
biomass: 760 wheat plots, two trials x anthesis/maturity; log(kg/ha) target, normalisation mean 9.239603 sd 0.227668; dataset embedding order (sorted names):
  0 anthesis trial B, 1 maturity trial B, 2 anthesis trial A, 3 maturity trial A; a new dataset is scored with every training embedding of its stage, averaged.
height: 128 plots of the maturity flight (the only flight with a ruler measurement); cm, mean 84.656250 sd 5.091641; one dataset embedding.
Held-out 10-fold scores (grouped): research/2026-10-09/results_summary/two_stream (dl_*_commonbands_grouped*_comparison.csv).

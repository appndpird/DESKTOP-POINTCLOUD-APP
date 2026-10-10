PhenoApp v3.3.2 pretrained two-stream ensembles (2026-10-10): {variant}_{target}_fold{k}.pt, k = 0..4, float16 state dicts.
Trained with the common band policy: common_bands.csv lists the 111 bands (common_95 = 1) used for every plot; the runner applies it automatically when this file is present.
biomass: log(kg/ha) target, normalisation mean 9.239603 sd 0.227668 over 760 wheat plots (two trials, anthesis + maturity); height: cm, mean 84.656250 sd 5.091641 over 256 plots.
Dataset embedding order (sorted training dataset names): 0 anthesis trial B, 1 maturity trial B, 2 anthesis trial A, 3 maturity trial A; a new dataset is mapped by stage keyword to 2 / 3.
Held-out 10-fold scores of the full 10-fold ensembles are in research/2026-10-09/results_summary/two_stream.

# Saved models of the 9-10 October 2026 rebuild

## height\ and biomass\ (feature models, scikit-learn, joblib)
One file per feature family and flight / dataset: the best learner of that family by nested repeated 10-fold RMSE
(`best_models.csv` and `summary_best_per_family.csv` give R2, RMSE, the learner and the input features of every file;
`pca_components.csv` the components kept by the PCA families). Each file is a dict {model, features, pca_family, target,
dataset, cv, data}; `model` is a fitted scikit-learn estimator or pipeline (the PCA families carry their scaler and PCA
inside), `features` the raw input columns in order. Height targets are cm, biomass kg/ha dry. Apply with:

    import joblib, pandas as pd
    m = joblib.load("biomass/biomass_muresk_anthesis_Fused_PCA_joint_GPR.joblib")
    X = features_all[m["features"]].to_numpy(float)          # features_all.csv of the flight (LiDAR v3 + VNIR v3 columns, NaN rows excluded)
    y_pred = m["model"].predict(X)

or from the PhenoApp Height Models / Biomass ML tab, "Apply a saved model". Models transfer between dates and trials only
with an offset calibrated on a few measured plots (see USER_GUIDE.md).

## two_stream\ (deep network, PyTorch state dicts, float16)
`commonbands\`: the standard ensembles, trained with the fixed common band list (`common_bands.csv`, 111 bands) - all
10 fold models of every variant (two_stream, lidar_only, vnir_only) for biomass and height; the five first folds of
two_stream and lidar_only are the ensembles bundled in PhenoApp v3.3.2. `per_plot\`: the two_stream fold models of the
per-plot band policy, for reference. File names follow the tool's convention `<variant>_<target>_fold<k>.pt`, so either
folder can be given as the weights folder on the Deep Models tab or to the runner:

    python phenoapp/dl/two_stream.py predict --data <dl_data> --weights research/2026-10-09/models/two_stream/commonbands --target biomass --out pred.csv
    python phenoapp/dl/two_stream.py train   --data <dl_data> --target biomass --out <dir> --init research/2026-10-09/models/two_stream/commonbands   (fine-tune)

Target normalisation inside the runner: biomass log(kg/ha) mean 9.239603 sd 0.227668; height cm mean 84.656250 sd 5.091641.
Dataset embedding order of the training set (sorted names): 0 = anthesis trial B, 1 = maturity trial B, 2 = anthesis
trial A, 3 = maturity trial A; a new dataset is scored with every training embedding of its stage and averaged.
Held-out 10-fold scores: results_summary/two_stream/*.csv. Training code: dl_prepare.py (tensors), dl_twostream.py
(research run), phenoapp/dl/two_stream.py (tool runner, predict / train / fine-tune).

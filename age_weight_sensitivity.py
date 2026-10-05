#!/usr/bin/env python3
"""
Sensitivity test for the weight of the ordinal age-band feature.
Check whether age is too strong, and whether a lower weight gives a better segmentation.
Steps: prepare data, log_standard baseline, age weight 1 / 0.5 / 0, then K-Means.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import segmentation as seg

AGE_WEIGHTS = [1.0, 0.5, 0.0]
K = 3
N_INIT = 20

def prepare_data(cfg):
    raw = seg.load_data(cfg)
    df, _ = seg.audit_and_clean(raw)

    # drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Baseline imputation and derived variables
    df, _ = seg.apply_imputation(df, "knn", cfg)
    df = seg.add_derived(df)

    # Baseline feature space
    X, names = seg.build_space(df, "log_standard", cfg)
    if names[0] != "AGE_ORD":
        raise ValueError("AGE_ORD is expected to be column 0.")
    return df, X


def run_model(df, X, age_weight, cfg, sil_idx, stab_idx):
    # k=3, change only the age-band weight
    Xw = X.copy()
    # Apply weight after standardization
    Xw[:, 0] *= age_weight

    labels = KMeans(
        n_clusters=K,
        n_init=N_INIT,
        random_state=cfg.random_state,
    ).fit_predict(Xw)

    sil = silhouette_score(Xw[sil_idx], labels[sil_idx])
    stab_mean, stab_sd, _ = seg.bootstrap_stability(
        Xw[stab_idx],
        K,
        cfg.n_bootstrap,
        cfg.random_state,
    )
    profile, _, _ = seg.build_profiles(df, labels)
    return labels, sil, stab_mean, stab_sd, profile


def print_profile(profile):
    cols = [
        "n",
        "share",
        "income_median",
        "trb_median",
        "age_18-24",
        "age_25-29",
        "age_30-34",
    ]
    # Cluster IDs can change, so display from highest to lowest TRB
    table = profile[cols].sort_values("trb_median", ascending=False)
    print(table.round(3))


def main():
    cfg = seg.Config.from_env()
    df, X = prepare_data(cfg)

    # Match the baseline sampling procedure
    sil_rng = np.random.default_rng(cfg.random_state)
    sil_idx = sil_rng.choice(len(X), min(cfg.sample_size, len(X)), replace=False)
    stab_rng = np.random.default_rng(cfg.random_state)
    stab_idx = stab_rng.choice(len(X), min(cfg.stability_sample, len(X)), replace=False)

    baseline_labels = None
    for weight in AGE_WEIGHTS:
        labels, sil, stab_mean, stab_sd, profile = run_model(
            df, X, weight, cfg, sil_idx, stab_idx
        )
        if baseline_labels is None:
            baseline_labels = labels
            ari = 1.0
        else:
            ari = adjusted_rand_score(baseline_labels, labels)

        print(f"\nAGE WEIGHT = {weight}")
        print(f"Silhouette:   {sil:.3f}")
        print(f"Stability ARI: {stab_mean:.3f} ± {stab_sd:.3f}")
        print(f"ARI vs 1.0:   {ari:.3f}")
        print_profile(profile)


if __name__ == "__main__":
    main()

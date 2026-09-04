"""Graphes de comparaison des couleurs des affiches par genre."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES_CSV = "features_par_affiche.csv"
SCALAR_FIG = "comparaison_couleurs_par_genre.png"
HISTOGRAM_FIG = "comparaison_couleurs_par_genre_histogrammes.png"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}


def plot_scalar_features(dataframe):
    scalar_features = [
        "brightness_mean", "saturation_mean", "saturation_std",
        "colorfulness", "lab_l_mean", "lab_a_mean", "lab_b_mean",
        "lab_l_std", "lab_a_std", "lab_b_std",
    ]
    missing = [feature for feature in scalar_features if feature not in dataframe]
    if missing:
        raise ValueError(f"Caractéristiques absentes : {missing}")

    plot_df = dataframe.copy()
    plot_df["genre"] = plot_df["label"].map(CLASS_NAMES)
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    for axis, feature in zip(axes.flat, scalar_features):
        sns.boxplot(data=plot_df, x="genre", y=feature, order=genre_order, ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
    figure.suptitle("Comparaison HSV, LAB, saturation et colorfulness par genre")
    figure.tight_layout()
    figure.savefig(SCALAR_FIG, dpi=150)
    plt.close(figure)


def plot_histograms(dataframe):
    plot_df = dataframe.copy()
    plot_df["genre"] = plot_df["label"].map(CLASS_NAMES)
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    families = ("hue_hist_", "saturation_hist_", "value_hist_")
    titles = (
        "Histogramme de teinte HSV",
        "Histogramme de saturation HSV",
        "Histogramme de luminosité HSV",
    )
    for axis, prefix, title in zip(axes, families, titles):
        columns = [column for column in plot_df if column.startswith(prefix)]
        if not columns:
            raise ValueError(f"Histogramme absent : {prefix}")
        means = plot_df.groupby("genre", sort=False)[columns].mean().reindex(genre_order)
        for genre in genre_order:
            axis.plot(range(len(columns)), means.loc[genre], marker="o", label=genre)
        axis.set_title(title)
        axis.set_xlabel("Numéro de bin")
        axis.set_ylabel("Proportion moyenne de pixels")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Genre", bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.suptitle("Histogrammes de couleurs moyens par genre")
    figure.tight_layout()
    figure.savefig(HISTOGRAM_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    dataframe = pd.read_csv(FEATURES_CSV)
    if "label" not in dataframe:
        raise ValueError("La colonne label est absente du CSV")
    plot_scalar_features(dataframe)
    plot_histograms(dataframe)
    print(f"Graphes créés : {SCALAR_FIG} et {HISTOGRAM_FIG}")


if __name__ == "__main__":
    main()

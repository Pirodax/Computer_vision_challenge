"""Graphes de comparaison de la luminosite des affiches par genre."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES_CSV = "features_par_affiche.csv"
BOXPLOT_FIG = "comparaison_luminosite_par_genre.png"
DISTRIBUTION_FIG = "distributions_luminosite_par_genre.png"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}
BRIGHTNESS_FEATURES = [
    "brightness_mean",
    "dark_ratio",
    "bright_ratio",
    "contrast",
]


def prepare_data():
    dataframe = pd.read_csv(FEATURES_CSV)
    missing = [feature for feature in BRIGHTNESS_FEATURES if feature not in dataframe]
    if "label" not in dataframe or missing:
        raise ValueError(f"Colonnes absentes : {missing + ([] if 'label' in dataframe else ['label'])}")
    dataframe["genre"] = dataframe["label"].map(CLASS_NAMES)
    return dataframe


def plot_boxplots(dataframe):
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    for axis, feature in zip(axes.flat, BRIGHTNESS_FEATURES):
        sns.boxplot(data=dataframe, x="genre", y=feature, order=genre_order, ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Comparaison des caractéristiques de luminosité par genre", fontsize=16)
    figure.tight_layout()
    figure.savefig(BOXPLOT_FIG, dpi=150)
    plt.close(figure)


def plot_distributions(dataframe):
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    for axis, feature in zip(axes.flat, BRIGHTNESS_FEATURES):
        for genre in genre_order:
            values = dataframe.loc[dataframe["genre"] == genre, feature]
            sns.kdeplot(values, ax=axis, label=genre, fill=False, warn_singular=False)
        axis.set_title(f"Distribution de {feature}")
        axis.set_xlabel(feature)
        axis.set_ylabel("Densité")
        axis.grid(alpha=0.25)
    axes[0, 1].legend(title="Genre")
    figure.suptitle("Distribution des caractéristiques de luminosité par genre", fontsize=16)
    figure.tight_layout()
    figure.savefig(DISTRIBUTION_FIG, dpi=150)
    plt.close(figure)


def main():
    dataframe = prepare_data()
    plot_boxplots(dataframe)
    plot_distributions(dataframe)
    print(f"Graphiques créés : {BOXPLOT_FIG} et {DISTRIBUTION_FIG}")


if __name__ == "__main__":
    main()

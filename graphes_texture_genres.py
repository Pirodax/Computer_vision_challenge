"""Graphes de comparaison des textures des affiches par genre."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES_CSV = "features_par_affiche.csv"
GLCM_FIG = "comparaison_texture_glcm_par_genre.png"
LBP_FIG = "comparaison_texture_lbp_par_genre.png"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}
GLCM_FEATURES = [
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
]


def prepare_data():
    dataframe = pd.read_csv(FEATURES_CSV)
    missing = [feature for feature in GLCM_FEATURES if feature not in dataframe]
    if "label" not in dataframe or missing:
        raise ValueError(f"Colonnes absentes : {missing + ([] if 'label' in dataframe else ['label'])}")
    dataframe["genre"] = dataframe["label"].map(CLASS_NAMES)
    return dataframe


def plot_glcm(dataframe):
    """Compare les trois indicateurs calculés à partir de la matrice GLCM."""
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    for axis, feature in zip(axes, GLCM_FEATURES):
        sns.boxplot(data=dataframe, x="genre", y=feature, order=genre_order, ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Comparaison des caractéristiques GLCM par genre", fontsize=16)
    figure.tight_layout()
    figure.savefig(GLCM_FIG, dpi=150)
    plt.close(figure)


def plot_lbp(dataframe):
    """Compare les histogrammes LBP moyens, indicateurs de texture locale."""
    genre_order = list(CLASS_NAMES.values())
    columns = [column for column in dataframe if column.startswith("lbp_hist_")]
    if not columns:
        raise ValueError("Aucune caractéristique LBP n'est présente dans le CSV")

    means = dataframe.groupby("genre", sort=False)[columns].mean().reindex(genre_order)
    figure, axis = plt.subplots(figsize=(11, 6))
    for genre in genre_order:
        axis.plot(
            range(len(columns)),
            means.loc[genre],
            marker="o",
            linewidth=2,
            label=genre,
        )
    axis.set_title("Histogrammes LBP moyens par genre")
    axis.set_xlabel("Motif LBP (bin)")
    axis.set_ylabel("Proportion moyenne de pixels")
    axis.set_xticks(range(len(columns)))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(title="Genre")
    figure.tight_layout()
    figure.savefig(LBP_FIG, dpi=150)
    plt.close(figure)


def main():
    dataframe = prepare_data()
    plot_glcm(dataframe)
    plot_lbp(dataframe)
    print(f"Graphiques créés : {GLCM_FIG} et {LBP_FIG}")


if __name__ == "__main__":
    main()

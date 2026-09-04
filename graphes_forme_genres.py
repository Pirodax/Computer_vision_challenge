"""Graphes de comparaison des caractéristiques de forme par genre."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES_CSV = "features_par_affiche.csv"
EDGES_FIG = "comparaison_contours_par_genre.png"
HOG_FIG = "comparaison_hog_par_genre.png"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}


def prepare_data():
    dataframe = pd.read_csv(FEATURES_CSV)
    edge_features = ["edge_density"] + [
        f"zone_{row}_{column}_edge_density"
        for row in range(2)
        for column in range(2)
    ]
    hog_features = [column for column in dataframe if column.startswith("hog_")]
    missing = [feature for feature in edge_features if feature not in dataframe]
    if "label" not in dataframe or missing or not hog_features:
        raise ValueError(
            f"Colonnes absentes : {missing + ([] if 'label' in dataframe else ['label'])}; "
            f"HOG présent : {bool(hog_features)}"
        )
    dataframe["genre"] = dataframe["label"].map(CLASS_NAMES)
    return dataframe, edge_features, sorted(hog_features, key=lambda name: int(name.split("_")[1]))


def plot_edge_density(dataframe, edge_features):
    genre_order = list(CLASS_NAMES.values())
    long_dataframe = dataframe.melt(
        id_vars="genre",
        value_vars=edge_features,
        var_name="zone",
        value_name="edge_value",
    )
    long_dataframe["zone"] = long_dataframe["zone"].replace({
        "edge_density": "globale",
        "zone_0_0_edge_density": "haut-gauche",
        "zone_0_1_edge_density": "haut-droite",
        "zone_1_0_edge_density": "bas-gauche",
        "zone_1_1_edge_density": "bas-droite",
    })
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.boxplot(
        data=dataframe,
        x="genre",
        y="edge_density",
        order=genre_order,
        ax=axes[0],
    )
    axes[0].set_title("Densité globale des contours")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    sns.barplot(
        data=long_dataframe,
        x="zone",
        y="edge_value",
        hue="genre",
        hue_order=genre_order,
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("Densité moyenne des contours par zone")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Densité de contours")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Comparaison des contours par genre", fontsize=16)
    figure.tight_layout()
    figure.savefig(EDGES_FIG, dpi=150)
    plt.close(figure)


def plot_hog(dataframe, hog_features):
    genre_order = list(CLASS_NAMES.values())
    means = dataframe.groupby("genre", sort=False)[hog_features].mean().reindex(genre_order)
    figure, axis = plt.subplots(figsize=(12, 7))
    for genre in genre_order:
        axis.plot(
            range(len(hog_features)),
            means.loc[genre],
            marker="o",
            linewidth=2,
            label=genre,
        )
    axis.set_title("Histogrammes HOG moyens par genre")
    axis.set_xlabel("Bin HOG")
    axis.set_ylabel("Proportion des valeurs HOG")
    axis.set_xticks(range(len(hog_features)))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(title="Genre")
    figure.tight_layout()
    figure.savefig(HOG_FIG, dpi=150)
    plt.close(figure)


def main():
    dataframe, edge_features, hog_features = prepare_data()
    plot_edge_density(dataframe, edge_features)
    plot_hog(dataframe, hog_features)
    print(f"Graphiques créés : {EDGES_FIG} et {HOG_FIG}")


if __name__ == "__main__":
    main()

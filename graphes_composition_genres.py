"""Graphes de comparaison de la composition spatiale des affiches."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES_CSV = "features_par_affiche.csv"
BOXPLOT_FIG = "comparaison_composition_par_genre.png"
HEATMAP_FIG = "cartes_chaleur_composition_par_genre.png"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}
ZONE_METRICS = ["brightness", "saturation", "colorfulness", "edge_density"]
ZONE_LABELS = {
    "zone_0_0": "haut-gauche",
    "zone_0_1": "haut-droite",
    "zone_1_0": "bas-gauche",
    "zone_1_1": "bas-droite",
}


def prepare_data():
    dataframe = pd.read_csv(FEATURES_CSV)
    zone_features = [
        f"{zone}_{metric}"
        for zone in ZONE_LABELS
        for metric in ZONE_METRICS
    ]
    missing = [feature for feature in zone_features if feature not in dataframe]
    if "label" not in dataframe or missing:
        raise ValueError(
            f"Colonnes absentes : {missing + ([] if 'label' in dataframe else ['label'])}"
        )
    dataframe["genre"] = dataframe["label"].map(CLASS_NAMES)
    return dataframe


def plot_boxplots(dataframe):
    """Compare les distributions de chaque mesure dans les quatre zones."""
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    genre_order = list(CLASS_NAMES.values())
    for axis, metric in zip(axes.flat, ZONE_METRICS):
        columns = [f"{zone}_{metric}" for zone in ZONE_LABELS]
        long_dataframe = dataframe.melt(
            id_vars="genre",
            value_vars=columns,
            var_name="zone",
            value_name="value",
        )
        long_dataframe["zone"] = long_dataframe["zone"].str.replace(
            f"_{metric}", "", regex=False
        ).map(ZONE_LABELS)
        sns.boxplot(
            data=long_dataframe,
            x="zone",
            y="value",
            hue="genre",
            hue_order=genre_order,
            ax=axis,
        )
        axis.set_title(f"{metric} par zone")
        axis.set_xlabel("")
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        axis.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Genre", loc="upper center", ncol=5)
    figure.suptitle("Comparaison de la composition spatiale par genre", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(BOXPLOT_FIG, dpi=150)
    plt.close(figure)


def plot_heatmaps(dataframe):
    """Affiche le profil spatial moyen de chaque genre sous forme de cartes."""
    genre_order = list(CLASS_NAMES.values())
    figure, axes = plt.subplots(5, 4, figsize=(13, 15))
    for row, genre in enumerate(genre_order):
        genre_dataframe = dataframe[dataframe["genre"] == genre]
        for column, metric in enumerate(ZONE_METRICS):
            values = [
                genre_dataframe[f"{zone}_{metric}"].mean()
                for zone in ZONE_LABELS
            ]
            heatmap = pd.DataFrame(
                [[values[0], values[1]], [values[2], values[3]]],
                index=["haut", "bas"],
                columns=["gauche", "droite"],
            )
            sns.heatmap(
                heatmap,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar=False,
                ax=axes[row, column],
            )
            axes[row, column].set_title(f"{genre} - {metric}")
            axes[row, column].set_xlabel("")
            axes[row, column].set_ylabel("")
    figure.suptitle("Profils moyens de composition par genre", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(HEATMAP_FIG, dpi=150)
    plt.close(figure)


def main():
    dataframe = prepare_data()
    plot_boxplots(dataframe)
    plot_heatmaps(dataframe)
    print(f"Graphiques créés : {BOXPLOT_FIG} et {HEATMAP_FIG}")


if __name__ == "__main__":
    main()

"""Entraînement de modèles classiques pour classer les genres d'affiches.

Aucun modèle pré-entraîné n'est utilisé : les entrées proviennent uniquement
 des caractéristiques calculées par analyse_caracteristiques_affiches.py.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


FEATURES_CSV = "features_par_affiche.csv"
RESULTS_CSV = "resultats_modeles_genres.csv"
CONFUSION_FIG = "matrices_confusion_genres.png"
MODEL_FILE = "modele_genres_visuel.joblib"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai",
}

BASE_FEATURES = [
    "brightness_mean", "brightness_std", "saturation_mean", "saturation_std",
    "r_mean", "g_mean", "b_mean", "colorfulness", "color_entropy",
    "contrast", "dark_ratio", "bright_ratio", "edge_density",
    "texture_contrast", "texture_homogeneity", "texture_energy",
]
DIMENSION_FEATURES = ["orig_width", "orig_height", "aspect_ratio", "resolution_mp"]


def get_feature_groups(dataframe):
    """Construit les variantes à comparer sans utiliser le chemin ni le label."""
    missing = [column for column in BASE_FEATURES if column not in dataframe]
    if missing:
        raise ValueError(f"Caractéristiques absentes du CSV : {missing}")

    enriched = [
        column for column in dataframe.columns
        if column.startswith(("hue_hist_", "saturation_hist_", "value_hist_",
                              "zone_", "hog_", "lbp_hist_"))
        or column == "symmetry_score"
    ]
    visual_features = BASE_FEATURES + enriched
    return {
        "base_visuelles": BASE_FEATURES,
        "visuelles_enrichies": visual_features,
        "avec_dimensions": visual_features + DIMENSION_FEATURES,
    }


def make_models():
    """Modèles sans apprentissage externe, avec pondération des classes."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", max_iter=3000, random_state=42),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def plot_confusion_matrices(matrices):
    figure, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]
    labels = list(CLASS_NAMES)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Prédit")
        axis.set_ylabel("Réel")
    figure.tight_layout()
    figure.savefig(CONFUSION_FIG, dpi=150)
    plt.close(figure)


def train_and_evaluate(dataframe, feature_groups):
    labels = dataframe["label"].astype(int)
    train_indices, test_indices = train_test_split(
        dataframe.index,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    y_train = labels.loc[train_indices]
    y_test = labels.loc[test_indices]
    results = []
    matrices = {}
    best_visual_score = -1.0
    best_visual_model = None
    best_visual_features = None

    for group_name, columns in feature_groups.items():
        x_train = dataframe.loc[train_indices, columns].fillna(0)
        x_test = dataframe.loc[test_indices, columns].fillna(0)
        for model_name, model in make_models().items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            macro_f1 = f1_score(y_test, predictions, average="macro")
            results.append({
                "feature_set": group_name,
                "model": model_name,
                "macro_f1": macro_f1,
                "n_features": len(columns),
            })
            if group_name == "visuelles_enrichies" and macro_f1 > best_visual_score:
                best_visual_score = macro_f1
                best_visual_model = model
                best_visual_features = columns
            matrix_name = f"{group_name} + {model_name}"
            matrices[matrix_name] = confusion_matrix(
                y_test, predictions, labels=list(CLASS_NAMES)
            )
            print(f"\n--- {matrix_name} ({len(columns)} features) ---")
            print(classification_report(
                y_test,
                predictions,
                labels=list(CLASS_NAMES),
                target_names=list(CLASS_NAMES.values()),
                zero_division=0,
            ))

    results_dataframe = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_dataframe.to_csv(RESULTS_CSV, index=False)
    joblib.dump(
        {
            "model": best_visual_model,
            "features": best_visual_features,
            "class_names": CLASS_NAMES,
        },
        MODEL_FILE,
    )
    plot_confusion_matrices(matrices)
    print("\n--- Comparaison des modèles ---")
    print(results_dataframe.to_string(index=False))
    print(f"\nRésultats sauvegardés dans {RESULTS_CSV}")
    print(f"Matrices de confusion sauvegardées dans {CONFUSION_FIG}")
    print(f"Modèle visuel sauvegardé dans {MODEL_FILE}")
    return results_dataframe


def main():
    dataframe = pd.read_csv(FEATURES_CSV)
    required = {"label", "path"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du CSV : {sorted(missing)}")
    if not any(column.startswith("hog_") for column in dataframe.columns):
        raise ValueError(
            "Le CSV ne contient pas les features enrichies. "
            "Relance d'abord analyse_caracteristiques_affiches.py."
        )
    feature_groups = get_feature_groups(dataframe)
    train_and_evaluate(dataframe, feature_groups)


if __name__ == "__main__":
    main()

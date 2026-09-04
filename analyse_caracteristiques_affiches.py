"""
Analyse exploratoire des caractéristiques visuelles des affiches par label.

Objectif : mesurer un ensemble de caractéristiques simples (couleur, contraste,
texture) sur chaque image, les comparer entre les 5 catégories, et identifier
lesquelles sont les plus discriminantes AVANT même d'entraîner un modèle.
C'est une étape utile pour :
  - comprendre si le problème est "raisonnable" visuellement
  - éventuellement nourrir un modèle classique (SVM/Random Forest) avec ces
    features en plus/à la place des pixels bruts
  - orienter le choix d'augmentation de données (si la couleur est très
    discriminante, éviter les augmentations de couleur trop agressives par ex.)

Structure de fichiers attendue (dossier plat + CSV de correspondance) :
    train/*.jpg                  <- toutes les images, sans sous-dossier par classe
    labels.csv                   <- colonnes : nom de fichier + label
        exemple :
            filename,label
            poster_0001.jpg,comedie
            poster_0002.jpg,horreur
            ...

Si les noms de colonnes de votre CSV diffèrent de "filename"/"label", ajustez
CSV_FILENAME_COL et CSV_LABEL_COL dans la section CONFIGURATION ci-dessous.

Les images du dataset n'étant PAS toutes de la même taille, deux précautions
sont prises :
  1. les dimensions d'origine (largeur, hauteur, ratio d'aspect, résolution)
     sont elles-mêmes enregistrées comme features — potentiellement
     discriminantes (les affiches "art et essai" ont parfois un format
     différent des affiches commerciales)
  2. les calculs sensibles à la résolution (contours, texture GLCM) sont
     effectués sur une version redimensionnée à une taille fixe (STANDARD_SIZE),
     pour que les valeurs restent comparables d'une image à l'autre. Les
     features de couleur/luminosité, elles, sont des moyennes/écarts-types
     et restent valides quelle que soit la résolution d'origine.

Dépendances : opencv-python, scikit-image, pandas, seaborn, matplotlib, scipy, scikit-learn
    pip install opencv-python scikit-image pandas seaborn matplotlib scipy scikit-learn --break-system-packages
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from scipy.stats import f_oneway

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

DATA_DIR = "train"                  # dossier contenant toutes les .jpg
LABELS_CSV = "train_labels.csv"           # CSV avec la correspondance fichier -> label
CSV_FILENAME_COL = "filename"
CSV_LABEL_COL = "label"
CLASS_NAMES = {
    0: "animation",
    1: "blockbuster",
    2: "horreur",
    3: "comédie",
    4: "art_et_essai"
}

OUTPUT_CSV = "features_par_affiche.csv"
OUTPUT_MEANS_CSV = "moyennes_caracteristiques_par_genre.csv"
OUTPUT_FIG = "boxplots_features_par_label.png"
OUTPUT_COLOR_FIG = "comparaison_couleurs_par_genre.png"
STANDARD_SIZE = 256  # taille de redimensionnement pour les calculs sensibles
                      # à la résolution (contours, texture)
HISTOGRAM_BINS = 16


# ----------------------------------------------------------------------------
# EXTRACTION DES CARACTÉRISTIQUES
# ----------------------------------------------------------------------------

def colorfulness(image_bgr):
    """Métrique de Hasler & Süsstrunk (2003) : combine saturation et diversité
    chromatique en une seule valeur. Plus la valeur est haute, plus l'image
    est visuellement "colorée/spectaculaire"."""
    b, g, r = cv2.split(image_bgr.astype("float"))
    rg = np.absolute(r - g)
    yb = np.absolute(0.5 * (r + g) - b)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    return std_root + (0.3 * mean_root)


def warm_cool_ratio(hue_channel):
    """Proportion de pixels dans les teintes chaudes (rouge/orange/jaune,
    hue OpenCV 0-30 et 150-180) par rapport aux teintes froides (bleu/vert,
    hue 60-140)."""
    warm_mask = ((hue_channel <= 30) | (hue_channel >= 150))
    cool_mask = (hue_channel >= 60) & (hue_channel <= 140)
    n_warm = np.sum(warm_mask)
    n_cool = np.sum(cool_mask)
    if n_cool == 0:
        return float(n_warm)  # évite division par zéro
    return n_warm / n_cool


def normalized_histogram(channel, bins, value_range):
    histogram, _ = np.histogram(channel, bins=bins, range=value_range)
    return histogram.astype(float) / max(histogram.sum(), 1)


def add_histogram_features(features, img_hsv):
    """Ajoute des histogrammes HSV normalisés, indépendants de la résolution."""
    channels = (("hue", img_hsv[:, :, 0], (0, 180)),
                ("saturation", img_hsv[:, :, 1], (0, 256)),
                ("value", img_hsv[:, :, 2], (0, 256)))
    for name, channel, value_range in channels:
        histogram = normalized_histogram(channel, HISTOGRAM_BINS, value_range)
        for index, value in enumerate(histogram):
            features[f"{name}_hist_{index}"] = float(value)


def add_lab_features(features, image_bgr):
    """Ajoute les moyennes et écarts-types dans l'espace couleur LAB."""
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channel_names = ("lab_l", "lab_a", "lab_b")
    for index, name in enumerate(channel_names):
        channel = image_lab[:, :, index]
        features[f"{name}_mean"] = float(np.mean(channel))
        features[f"{name}_std"] = float(np.std(channel))


def add_zone_features(features, img_gray, img_hsv):
    """Mesure couleur, luminosité et contours séparément dans quatre zones."""
    height, width = img_gray.shape
    edges = cv2.Canny(img_gray, 100, 200)
    for row in range(2):
        for column in range(2):
            y_start, y_end = row * height // 2, (row + 1) * height // 2
            x_start, x_end = column * width // 2, (column + 1) * width // 2
            zone_name = f"zone_{row}_{column}"
            gray_zone = img_gray[y_start:y_end, x_start:x_end]
            hsv_zone = img_hsv[y_start:y_end, x_start:x_end]
            edge_zone = edges[y_start:y_end, x_start:x_end]
            features[f"{zone_name}_brightness"] = float(np.mean(gray_zone))
            features[f"{zone_name}_saturation"] = float(np.mean(hsv_zone[:, :, 1]))
            features[f"{zone_name}_colorfulness"] = float(
                colorfulness(cv2.cvtColor(hsv_zone, cv2.COLOR_HSV2BGR)))
            features[f"{zone_name}_edge_density"] = float(np.mean(edge_zone > 0))


def add_shape_features(features, img_gray):
    """Ajoute des descripteurs classiques de forme et de texture, sans modèle appris."""
    shape_image = cv2.resize(img_gray, (64, 64))
    hog_values = hog(
        shape_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    # Un histogramme compact évite de créer des milliers de colonnes HOG.
    hog_histogram, _ = np.histogram(hog_values, bins=16, range=(0, 1))
    hog_histogram = hog_histogram.astype(float) / max(hog_histogram.sum(), 1)
    for index, value in enumerate(hog_histogram):
        features[f"hog_{index}"] = float(value)

    lbp = local_binary_pattern(shape_image, P=8, R=1, method="uniform")
    lbp_histogram, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10))
    lbp_histogram = lbp_histogram.astype(float) / max(lbp_histogram.sum(), 1)
    for index, value in enumerate(lbp_histogram):
        features[f"lbp_hist_{index}"] = float(value)

    half = img_gray.shape[1] // 2
    left = img_gray[:, :half]
    right = cv2.flip(img_gray[:, -half:], 1)
    features["symmetry_score"] = float(np.mean(np.abs(left.astype(float) - right)))



def extract_features(image_path):
    try:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            image_rgb.load()
        img_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError):
        return None

    orig_height, orig_width = img_bgr.shape[:2]
    aspect_ratio = orig_width / orig_height
    resolution_mp = (orig_width * orig_height) / 1_000_000  # mégapixels

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)
    b, g, r = cv2.split(img_bgr)

    # --- Dimensions d'origine (dataset non uniforme) ---
    # Ces features capturent le format brut de l'affiche, potentiellement
    # révélateur du type de production/édition.


    # --- Couleur ---
    brightness_mean = float(np.mean(v))
    brightness_std = float(np.std(v))
    saturation_mean = float(np.mean(s))
    saturation_std = float(np.std(s))
    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    colorfulness_score = float(colorfulness(img_bgr))
    wc_ratio = float(warm_cool_ratio(h))

    # Entropie de l'histogramme de couleur (sur la teinte, 32 bins)
    hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
    hist_h = hist_h / (hist_h.sum() + 1e-8)
    color_entropy = float(-np.sum(hist_h * np.log2(hist_h + 1e-8)))

    # --- Contraste / luminosité ---
    # Note : contrast, dark_ratio, bright_ratio sont des proportions/écarts-types
    # normalisés (0-255, ou ratios de pixels) donc valides quelle que soit la
    # résolution — pas besoin de standardiser la taille pour ceux-là.
    contrast = float(np.std(img_gray))
    dark_ratio = float(np.mean(img_gray < 60))     # proportion de pixels sombres
    bright_ratio = float(np.mean(img_gray > 200))  # proportion de pixels clairs

    # --- Contours / texture : calculés sur une taille FIXE (STANDARD_SIZE) ---
    # Important car le dataset n'est pas uniformément dimensionné : la densité
    # de contours et la texture GLCM dépendent de la résolution (une image
    # plus grande a mécaniquement plus de pixels de contour). Redimensionner
    # d'abord rend ces features comparables entre toutes les images.
    img_gray_std = cv2.resize(img_gray, (STANDARD_SIZE, STANDARD_SIZE))
    img_hsv_std = cv2.resize(img_hsv, (STANDARD_SIZE, STANDARD_SIZE))

    edges = cv2.Canny(img_gray_std, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Sous-échantillonnage des niveaux de gris à 32 pour accélérer le calcul
    img_gray_small = (img_gray_std / 8).astype(np.uint8)  # 256 -> 32 niveaux
    glcm = graycomatrix(img_gray_small, distances=[1], angles=[0],
                         levels=32, symmetric=True, normed=True)
    texture_contrast = float(graycoprops(glcm, "contrast")[0, 0])
    texture_homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    texture_energy = float(graycoprops(glcm, "energy")[0, 0])

    features = {
        # Dimensions d'origine
        "orig_width": orig_width,
        "orig_height": orig_height,
        "aspect_ratio": aspect_ratio,
        "resolution_mp": resolution_mp,
        
        # Couleur
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "colorfulness": colorfulness_score,
        "warm_cool_ratio": wc_ratio,
        "color_entropy": color_entropy,
        # Contraste / texture (calculés à résolution standardisée)
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
        "texture_contrast": texture_contrast,
        "texture_homogeneity": texture_homogeneity,
        "texture_energy": texture_energy,
    }
    add_histogram_features(features, img_hsv_std)
    add_lab_features(features, img_bgr)
    add_zone_features(features, img_gray_std, img_hsv_std)
    add_shape_features(features, img_gray_std)
    return features


def build_feature_dataframe():
    labels_df = pd.read_csv(LABELS_CSV,
    header=None,
    names=["filename", "label"])

    # Vérification que les colonnes attendues existent bien
    if CSV_FILENAME_COL not in labels_df.columns or CSV_LABEL_COL not in labels_df.columns:
        raise ValueError(
            f"Colonnes attendues '{CSV_FILENAME_COL}'/'{CSV_LABEL_COL}' introuvables dans "
            f"{LABELS_CSV}. Colonnes disponibles : {list(labels_df.columns)}. "
            f"Ajustez CSV_FILENAME_COL / CSV_LABEL_COL en haut du script."
        )

    print(f"{len(labels_df)} entrées trouvées dans {LABELS_CSV}")
    print("Répartition des labels :")
    print(labels_df[CSV_LABEL_COL].value_counts().to_string())

    rows = []
    n_missing = 0
    for _, row in labels_df.iterrows():
        label = int(row[CSV_LABEL_COL])

        filename = row[CSV_FILENAME_COL]


        if label not in CLASS_NAMES:
            # Label présent dans le CSV mais absent de CLASS_NAMES : on l'ignore
            # plutôt que de planter, mais on prévient.
            continue

        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            n_missing += 1
            continue

        feats = extract_features(path)
        if feats is None:
            print(f"  ! Impossible de lire {path}, ignorée")
            continue
        feats["label"] = label
        feats["path"] = path
        rows.append(feats)

    if n_missing:
        print(f"! {n_missing} fichiers listés dans le CSV mais introuvables dans {DATA_DIR}/")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{len(df)} images analysées au total. Features sauvegardées dans {OUTPUT_CSV}")
    return df


# ----------------------------------------------------------------------------
# ANALYSE : quelles features séparent le mieux les classes ?
# ----------------------------------------------------------------------------

def rank_features_by_discriminative_power(df):
    """Test ANOVA (F-test) par feature : mesure si la moyenne de la feature
    diffère significativement entre les 5 classes. Un F-score élevé et une
    p-value faible indiquent une feature discriminante."""
    feature_cols = [c for c in df.columns if c not in ("label", "path")]
    results = []
    for col in feature_cols:
        groups = [df[df["label"] == c][col].values for c in CLASS_NAMES]
        f_stat, p_val = f_oneway(*groups)
        results.append({"feature": col, "F_score": f_stat, "p_value": p_val})

    ranking = pd.DataFrame(results).sort_values("F_score", ascending=False)
    ranking.reset_index(drop=True, inplace=True)

    print("\n--- Classement des features par pouvoir discriminant (ANOVA) ---")
    print("F_score élevé + p_value < 0.05 = la feature distingue bien les classes\n")
    print(ranking.to_string(index=False))
    return ranking


def mean_features_by_genre(df):
    """Calcule les moyennes de toutes les caractéristiques pour chaque genre.

    Les lignes sont conservées dans l'ordre des labels 0 à 4, avec le nom du
    genre correspondant. Les genres sans image analysée auront des valeurs
    manquantes plutôt que de disparaître du résultat.
    """
    feature_cols = [c for c in df.columns if c not in ("label", "path")]
    means = (
        df.groupby("label")[feature_cols]
        .mean()
        .reindex(list(CLASS_NAMES))
        .reset_index()
    )
    means.insert(1, "genre", means["label"].map(CLASS_NAMES))
    means.to_csv(OUTPUT_MEANS_CSV, index=False)

    print("\n--- Moyennes des caractéristiques par genre ---")
    print(means.round(2).to_string(index=False))
    print(f"\nMoyennes sauvegardées dans {OUTPUT_MEANS_CSV}")
    return means


def plot_features_by_label(df, ranking, top_n=9):
    """Boxplots des top_n features les plus discriminantes, une par classe."""
    top_features = ranking["feature"].head(top_n).tolist()
    n_cols = 3
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        sns.boxplot(data=df, x="label", y=feat, ax=axes[i], order=CLASS_NAMES)
        axes[i].set_title(feat)
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].set_xlabel("")

    for j in range(len(top_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    print(f"\nGraphique sauvegardé dans {OUTPUT_FIG}")


def plot_color_features_by_genre(df):
    """Crée des graphes lisibles pour HSV, LAB, saturation et histogrammes."""
    plot_df = df.copy()
    plot_df["genre"] = plot_df["label"].map(CLASS_NAMES)
    genre_order = list(CLASS_NAMES.values())

    scalar_features = [
        "brightness_mean", "saturation_mean", "saturation_std",
        "colorfulness", "lab_l_mean", "lab_a_mean", "lab_b_mean",
        "lab_l_std", "lab_a_std", "lab_b_std",
    ]
    figure, axes = plt.subplots(2, 5, figsize=(22, 9))
    for axis, feature in zip(axes.flat, scalar_features):
        sns.boxplot(data=plot_df, x="genre", y=feature, order=genre_order, ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
    figure.suptitle("Comparaison des caractéristiques HSV/LAB par genre", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT_COLOR_FIG, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    histogram_families = ("hue_hist_", "saturation_hist_", "value_hist_")
    titles = ("Histogramme de teinte (HSV)", "Histogramme de saturation (HSV)",
              "Histogramme de luminosité (HSV)")
    for axis, prefix, title in zip(axes, histogram_families, titles):
        columns = [column for column in plot_df.columns if column.startswith(prefix)]
        means = plot_df.groupby("genre", sort=False)[columns].mean().reindex(genre_order)
        for genre in genre_order:
            axis.plot(range(len(columns)), means.loc[genre], marker="o", label=genre)
        axis.set_title(title)
        axis.set_xlabel("Numéro de bin")
        axis.set_ylabel("Proportion moyenne de pixels")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Genre", bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.suptitle("Histogrammes de couleurs moyens par genre", fontsize=16)
    figure.tight_layout()
    histogram_fig = OUTPUT_COLOR_FIG.replace(".png", "_histogrammes.png")
    figure.savefig(histogram_fig, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Graphiques couleur sauvegardés dans {OUTPUT_COLOR_FIG} et {histogram_fig}")


if __name__ == "__main__":
    df = build_feature_dataframe()
    ranking = rank_features_by_discriminative_power(df)
    plot_features_by_label(df, ranking, top_n=9)
    plot_color_features_by_genre(df)
    mean_features_by_genre(df)

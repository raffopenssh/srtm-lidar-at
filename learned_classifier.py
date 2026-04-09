"""Learned classifier: Random Forest trained on cadastre ground truth.

Replaces the hand-tuned decision tree in classify_object() with a
data-driven Random Forest that learns from cadastre land-use labels.

Architecture:
  1. Feature extraction: same per-segment features as before
     (height, slope, roughness, shape, NDVI, texture, harmonics, SAR)
  2. Training: cadastre parcel codes provide noisy labels
  3. Prediction: RF predicts type + probability per segment
  4. Disagreement detection: segments where RF disagrees with cadastre
     are flagged as likely cadastre errors

The classifier falls back to the rule-based system when no trained
model is available, so there's zero risk of regression.

Usage::

    from learned_classifier import LearnedClassifier
    clf = LearnedClassifier.load()       # load trained model
    pred = clf.predict(feature_dict)     # returns (type, conf)
    clf.train(features, labels)          # retrain from new data
    disagreements = clf.find_disagreements(features, cadastre_labels)
"""
from __future__ import annotations

import logging
import pathlib
import json
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = pathlib.Path("/tmp/learned_classifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"
META_PATH = MODEL_DIR / "rf_meta.json"

# Features used by the RF.  Order matters (must match training).
# These are the per-segment features from extract_object_features().
FEATURE_KEYS = [
    # Height & terrain (DTM/DSM 1m)
    "h_mean", "h_max", "h_std", "h_p90",
    "slope_mean", "slope_std",
    "dsm_roughness", "dtm_roughness",
    # Shape
    "compactness", "elongation", "solidity", "extent", "area",
    # DSM edges
    "dsm_edge_strength",
    # Spectral (BEV ortho 1m)
    "ndvi_mean", "ndvi_std", "brightness_mean", "nir_mean",
    # Copernicus spectral
    "cop_ndvi_mean", "fused_ndvi_mean", "fused_ndvi_std",
    # ESA WorldCover
    "esa_built_frac", "esa_tree_frac", "esa_crop_frac",
    "esa_grass_frac", "esa_water_frac",
    # Temporal
    "h_change", "dtm_change", "dtm_change_abs",
    "temporal_h_std", "stability",
    # GLCM texture
    "glcm_contrast", "glcm_homogeneity", "glcm_entropy",
    "glcm_dissimilarity", "glcm_energy", "texture_complexity",
    # SAR
    "sar_vv", "sar_vh", "sar_ratio",
    # NDVI harmonics
    "harm_mean", "harm_amplitude", "harm_phase", "harm_rmse",
    # Hansen Global Forest Change
    "hansen_treecover2000", "hansen_loss_frac", "hansen_recent_loss_frac",
    "hansen_gain_frac", "hansen_current_forest_frac",
    # Additional discriminators
    "ndvi_max", "slope_max", "h_p10", "perimeter", "esa_dominant_lc",
]

# Mapping from cadastre land-use codes (Benützungsart) to our types.
# These are the Austrian BEV cadastre "BA" codes.
CADASTRE_TO_TYPE = {
    # Agricultural
    51: "crop",       # Acker (arable)
    62: "crop",       # Acker (field)
    52: "grass",      # Wiese (meadow)
    53: "grass",      # Hutweide (pasture)
    54: "grass",      # Alpe (alpine pasture)
    55: "grass",      # Weide (grazing)
    58: "grass",      # Alpe (alpine meadow)
    61: "grass",      # Grünland (grassland)
    # Forest
    56: "tree",       # Wald (forest)
    57: "shrub",      # Krummholz/Strauchwald
    # Orchards/vineyards
    63: "vineyard",   # Weingarten
    64: "garden",     # Hausgarten
    65: "orchard",    # Obstgarten
    # Buildings
    40: "garden",     # Baufläche begrünt (green built-up)
    42: "roof",       # Gebäude
    43: "roof",       # Gebäude (other)
    44: "roof",       # Gebäude (farm)
    45: "roof",       # Gewächshaus (greenhouse)
    46: "roof",       # Gebäude
    47: "roof",       # Gebäude
    41: "parking",    # Baufläche (paved)
    # Transport
    48: "road",       # Straße (road)
    73: "road",       # Straße (road)
    74: "path",       # Weg (path)
    75: "road",       # Brücke (bridge)
    # Water
    70: "water",      # Gewässer
    71: "water",      # Stehende Gewässer
    96: "water",      # Feuchtgebiet
    # Terrain
    59: "bare_soil",  # Ödland
    60: "water",      # Sumpf/Moor (wetland)
    72: "water",      # Quelle/Brunnen (spring)
    80: "excavation", # Abbaufläche (quarry)
    81: "fill",       # Deponie
    83: "rock",       # Felsen (rock)
    84: "rock",       # Geröll (scree)
    90: "bare_soil",  # sonstige Fläche
    93: "excavation", # Abbaufläche
}

# Simplified target classes for the RF (merge rare types)
TYPE_CLASSES = [
    "tree", "shrub", "grass", "crop", "road", "path", "parking",
    "roof", "water", "bare_soil", "rock", "excavation", "fill",
    "garden", "orchard", "vineyard", "hedge", "fence", "wall",
    "tree_loss", "construction",
]


def feature_vector(feat: dict) -> np.ndarray:
    """Extract fixed-length feature vector from a segment feature dict."""
    return np.array([feat.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)


class LearnedClassifier:
    """Random Forest classifier trained on cadastre ground truth."""

    def __init__(self):
        self.model = None
        self.classes: list[str] = []
        self.feature_importances: dict[str, float] = {}
        self.n_train: int = 0
        self.oob_score: float = 0.0
        self.trained_at: str = ""
        self.n_kgs: int = 0

    def train(
        self,
        features: list[dict],
        labels: list[str],
        *,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 5,
        n_kgs: int = 0,
    ) -> dict:
        """Train RF on feature dicts + string labels.

        Returns training stats dict.
        """
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        # Filter valid
        X_list, y_list = [], []
        for feat, label in zip(features, labels):
            if label not in TYPE_CLASSES:
                continue
            X_list.append(feature_vector(feat))
            y_list.append(label)

        if len(X_list) < 20:
            raise ValueError(f"Need >= 20 training samples, got {len(X_list)}")

        X = np.stack(X_list)
        y = np.array(y_list)

        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        log.info("Training RF: %d samples, %d features, %d classes",
                 len(X), X.shape[1], len(set(y)))

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",  # handle imbalanced classes
        )
        rf.fit(X, y)

        self.model = rf
        self.classes = list(rf.classes_)
        self.n_train = len(X)
        self.oob_score = rf.oob_score_
        self.feature_importances = {
            k: float(v) for k, v in zip(FEATURE_KEYS, rf.feature_importances_)
        }

        import datetime
        self.trained_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        self.n_kgs = n_kgs

        # Save
        joblib.dump(rf, MODEL_PATH)
        meta = {
            "classes": self.classes,
            "n_train": self.n_train,
            "oob_score": self.oob_score,
            "feature_importances": self.feature_importances,
            "feature_keys": FEATURE_KEYS,
            "trained_at": self.trained_at,
            "n_kgs": self.n_kgs,
        }
        META_PATH.write_text(json.dumps(meta, indent=2))

        # Top features
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:10]
        log.info("RF trained: OOB=%.3f, top features: %s",
                 self.oob_score,
                 ", ".join(f"{k}={v:.3f}" for k, v in top))

        return {
            "n_train": self.n_train,
            "n_classes": len(self.classes),
            "oob_score": self.oob_score,
            "top_features": dict(top),
        }

    def predict(self, feat: dict) -> tuple[str, float]:
        """Predict type and confidence for a single segment.

        Returns (type_name, confidence).
        """
        if self.model is None:
            return "", 0.0

        x = feature_vector(feat).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        pred = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        conf = float(proba.max())

        return pred, conf

    def predict_batch(self, features: list[dict]) -> list[tuple[str, float]]:
        """Predict for multiple segments."""
        if self.model is None:
            return [("", 0.0)] * len(features)

        X = np.stack([feature_vector(f) for f in features])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)
        confs = probas.max(axis=1)

        return [(str(p), float(c)) for p, c in zip(preds, confs)]

    def find_disagreements(
        self,
        features: list[dict],
        cadastre_labels: list[str],
    ) -> list[dict]:
        """Find segments where RF disagrees with cadastre.

        These are candidate cadastre errors.

        Returns list of {segment_features, cadastre_label, rf_prediction,
        rf_confidence, likely_correct}.
        """
        if self.model is None:
            return []

        results = []
        for feat, cad_label in zip(features, cadastre_labels):
            pred, conf = self.predict(feat)
            if pred != cad_label and conf > 0.6:
                results.append({
                    "label": feat.get("label", 0),
                    "cadastre_type": cad_label,
                    "rf_prediction": pred,
                    "rf_confidence": round(conf, 3),
                    "likely_correct": "rf" if conf > 0.75 else "uncertain",
                    "area": feat.get("area", 0),
                    "centroid_e": feat.get("centroid_e", 0),
                    "centroid_n": feat.get("centroid_n", 0),
                })

        results.sort(key=lambda x: -x["rf_confidence"])
        log.info("Found %d disagreements (RF vs cadastre)", len(results))
        return results

    @classmethod
    def load(cls) -> "LearnedClassifier":
        """Load a previously trained model, or return empty classifier."""
        inst = cls()
        if MODEL_PATH.exists() and META_PATH.exists():
            try:
                import joblib
                inst.model = joblib.load(MODEL_PATH)
                meta = json.loads(META_PATH.read_text())
                inst.classes = meta.get("classes", [])
                inst.n_train = meta.get("n_train", 0)
                inst.oob_score = meta.get("oob_score", 0)
                inst.feature_importances = meta.get("feature_importances", {})
                inst.trained_at = meta.get("trained_at", "")
                inst.n_kgs = meta.get("n_kgs", 0)
                log.info("Loaded RF model: %d classes, OOB=%.3f, n=%d, kgs=%d, trained=%s",
                         len(inst.classes), inst.oob_score, inst.n_train,
                         inst.n_kgs, inst.trained_at)
            except Exception as e:
                log.warning("Failed to load RF model: %s", e)
        return inst

    @property
    def is_trained(self) -> bool:
        return self.model is not None


# ===================================================================
# Integration with object_segmentation pipeline
# ===================================================================

_cached_classifier: Optional[LearnedClassifier] = None
_cached_model_mtime: float = 0.0


def get_classifier() -> LearnedClassifier:
    """Get or load the singleton classifier.  Reloads when model file changes."""
    global _cached_classifier, _cached_model_mtime
    try:
        mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    except OSError:
        mtime = 0.0
    if _cached_classifier is None or mtime != _cached_model_mtime:
        _cached_classifier = LearnedClassifier.load()
        _cached_model_mtime = mtime
    return _cached_classifier


def classify_with_rf(
    feat: dict,
    *,
    fallback_fn=None,
    has_spectral: bool = False,
    min_confidence: float = 0.4,
) -> tuple[str, int, float, bool]:
    """Classify segment: try RF first, fall back to rules.

    Parameters
    ----------
    feat : feature dict from extract_object_features
    fallback_fn : callable(feat, has_spectral=bool) -> (name, code, conf, is_mm)
    has_spectral : whether ortho was available
    min_confidence : minimum RF confidence to trust

    Returns
    -------
    (type_name, type_code, confidence, is_manmade)
    """
    from object_segmentation import OBJECT_TYPES, classify_object

    clf = get_classifier()
    if clf.is_trained:
        pred, conf = clf.predict(feat)
        if pred and conf >= min_confidence and pred in OBJECT_TYPES:
            code = OBJECT_TYPES[pred]
            is_mm = pred in {
                "road", "path", "parking", "roof", "wall", "fence",
                "mast", "greenhouse", "solar_panel", "bridge",
                "excavation", "fill", "construction",
            }
            return pred, code, conf, is_mm

    # Fallback to rule-based
    if fallback_fn:
        return fallback_fn(feat, has_spectral=has_spectral)
    return classify_object(feat, has_spectral=has_spectral)

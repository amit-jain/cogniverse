"""
Profile Performance Optimizer

Learns which backend profile works best for different query types using XGBoost.
Analyzes Phoenix evaluation data to build (query_features, profile, ndcg) training dataset.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cogniverse_foundation.telemetry.manager import get_telemetry_manager

logger = logging.getLogger(__name__)


@dataclass
class QueryFeatures:
    """Extracted features from a query"""

    query_text: str
    query_length: int
    word_count: int
    has_temporal_keywords: bool
    has_spatial_keywords: bool
    has_object_keywords: bool
    avg_word_length: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for XGBoost"""
        return np.array(
            [
                self.query_length,
                self.word_count,
                float(self.has_temporal_keywords),
                float(self.has_spatial_keywords),
                float(self.has_object_keywords),
                self.avg_word_length,
            ]
        )


class ProfilePerformanceOptimizer:
    """
    XGBoost-based optimizer that learns which backend profile works best for different queries.

    Uses Phoenix evaluation data to extract (query_features, profile, ndcg) tuples and trains
    a classifier to predict the best profile for new queries.
    """

    # Temporal keywords for query classification
    TEMPORAL_KEYWORDS = {
        "when",
        "before",
        "after",
        "during",
        "while",
        "timeline",
        "sequence",
        "first",
        "last",
        "beginning",
        "end",
        "start",
        "finish",
        "recent",
        "old",
    }

    # Spatial keywords for query classification
    SPATIAL_KEYWORDS = {
        "where",
        "location",
        "place",
        "here",
        "there",
        "near",
        "far",
        "around",
        "inside",
        "outside",
        "left",
        "right",
        "top",
        "bottom",
        "scene",
    }

    # Object keywords for query classification
    OBJECT_KEYWORDS = {
        "object",
        "thing",
        "item",
        "person",
        "people",
        "face",
        "animal",
        "car",
        "building",
        "furniture",
        "device",
        "tool",
        "what",
        "who",
        "which",
    }

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize profile performance optimizer

        Args:
            model_dir: Directory to save/load models (defaults to outputs/models/profile_performance)
        """
        self.model_dir = model_dir or Path("outputs/models/profile_performance")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_trained = False

        logger.info(
            f"Initialized ProfilePerformanceOptimizer (model_dir: {self.model_dir})"
        )

    def extract_query_features(self, query_text: str) -> QueryFeatures:
        """
        Extract features from query text

        Args:
            query_text: Query string

        Returns:
            QueryFeatures object with extracted features
        """
        query_lower = query_text.lower()
        words = query_text.split()

        return QueryFeatures(
            query_text=query_text,
            query_length=len(query_text),
            word_count=len(words),
            has_temporal_keywords=any(
                kw in query_lower for kw in self.TEMPORAL_KEYWORDS
            ),
            has_spatial_keywords=any(kw in query_lower for kw in self.SPATIAL_KEYWORDS),
            has_object_keywords=any(kw in query_lower for kw in self.OBJECT_KEYWORDS),
            avg_word_length=np.mean([len(w) for w in words]) if words else 0.0,
        )

    async def extract_training_data_from_phoenix(
        self,
        tenant_id: str,
        project_name: str,
        start_time=None,
        end_time=None,
        min_samples: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract training data from telemetry provider evaluation spans

        Args:
            tenant_id: Tenant identifier
            project_name: Project name for span query
            start_time: Start time for span query
            end_time: End time for span query
            min_samples: Minimum samples required

        Returns:
            Tuple of (features_array, labels_array, profile_names)

        Raises:
            ValueError: If insufficient training data found
        """
        logger.info("Querying telemetry provider for evaluation spans...")

        # Get provider from telemetry manager
        telemetry_manager = get_telemetry_manager()
        provider = telemetry_manager.get_provider(tenant_id=tenant_id)

        # Get spans from provider
        spans_df = await provider.traces.get_spans(
            project=project_name, start_time=start_time, end_time=end_time
        )

        if spans_df is None or spans_df.empty:
            raise ValueError("No spans found in telemetry backend")

        # Filter for search/evaluation spans with profile and quality info
        search_spans = (
            spans_df[spans_df["name"].str.contains("search|eval", case=False, na=False)]
            if "name" in spans_df.columns
            else pd.DataFrame()
        )

        if search_spans.empty:
            raise ValueError("No search/evaluation spans found")

        logger.info(f"Found {len(search_spans)} search spans")

        # Extract profile and quality columns
        profile_cols = [col for col in search_spans.columns if "profile" in col.lower()]
        quality_cols = [
            col
            for col in search_spans.columns
            if any(metric in col.lower() for metric in ["ndcg", "score", "quality"])
        ]
        query_cols = [
            col
            for col in search_spans.columns
            if "query" in col.lower() or "input" in col.lower()
        ]

        if not profile_cols:
            raise ValueError("No profile information found in spans")
        if not quality_cols:
            raise ValueError("No quality metrics (NDCG) found in spans")
        if not query_cols:
            raise ValueError("No query text found in spans")

        profile_col = profile_cols[0]
        quality_col = quality_cols[0]
        query_col = query_cols[0]

        # Build training dataset
        training_data = search_spans[[query_col, profile_col, quality_col]].dropna()

        if len(training_data) < min_samples:
            raise ValueError(
                f"Insufficient training data: {len(training_data)} samples (need {min_samples})"
            )

        logger.info(f"Extracted {len(training_data)} training samples")

        # Extract features from queries
        features_list = []
        for query in training_data[query_col]:
            query_features = self.extract_query_features(str(query))
            features_list.append(query_features.to_array())

        X = np.array(features_list)

        # Encode profile labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(training_data[profile_col])

        profile_names = self.label_encoder.classes_.tolist()

        logger.info(f"Features shape: {X.shape}, Labels: {len(np.unique(y))} profiles")

        return X, y, profile_names

    def train(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, **xgb_params
    ) -> Dict[str, float]:
        """
        Train XGBoost classifier

        Args:
            X: Feature matrix
            y: Label array
            test_size: Test set proportion
            **xgb_params: Additional XGBoost parameters

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training XGBoost classifier on {len(X)} samples...")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Default XGBoost parameters
        params = {
            "max_depth": 5,
            "n_estimators": 200,
            "learning_rate": 0.1,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "mlogloss",
        }
        params.update(xgb_params)

        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)

        self.is_trained = True

        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_profiles": len(np.unique(y)),
        }

        logger.info(f"Training complete - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

        return metrics

    def predict_best_profile(self, query_text: str) -> Tuple[str, float]:
        """
        Predict best profile for a query

        Args:
            query_text: Query string

        Returns:
            Tuple of (profile_name, confidence)

        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Extract features
        query_features = self.extract_query_features(query_text)
        X = query_features.to_array().reshape(1, -1)

        # Predict
        pred_label = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]

        profile_name = self.label_encoder.inverse_transform([pred_label])[0]
        confidence = pred_proba[pred_label]

        return profile_name, float(confidence)

    def save(self) -> None:
        """Save model and encoder to disk"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_path = self.model_dir / "xgboost_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        with open(encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        logger.info(f"Model saved to {self.model_dir}")

    def load(self) -> bool:
        """
        Load model and encoder from disk

        Returns:
            True if successful, False otherwise
        """
        model_path = self.model_dir / "xgboost_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"

        if not model_path.exists() or not encoder_path.exists():
            logger.warning(f"Model files not found in {self.model_dir}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

            self.is_trained = True
            logger.info(f"Model loaded from {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

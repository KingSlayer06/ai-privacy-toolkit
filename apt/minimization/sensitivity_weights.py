"""
Sensitivity-Weighted Generalization Module

This module implements sensitivity-weighted NCP calculation and feature prioritization
for privacy-aware generalization. Features are assigned sensitivity scores, and the
generalization process prioritizes protecting more sensitive features.

Security Purpose:
- Protects sensitive features more aggressively than non-sensitive ones
- Enables domain-specific privacy requirements (e.g., medical data, financial data)
- Improves privacy-utility trade-offs by focusing generalization effort on high-value targets
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from collections import Counter


class SensitivityWeightCalculator:
    """
    Calculates and manages feature sensitivity scores for weighted generalization.
    
    Feature sensitivity indicates how important it is to protect a particular feature.
    Higher sensitivity scores (> 0.5) lead to more aggressive generalization of those features,
    while lower sensitivity scores (< 0.5) may be left less generalized or removed from
    generalization entirely.
    
    Security Mechanism:
    By weighting NCP calculations and feature removal decisions by sensitivity, the
    algorithm ensures that sensitive features (e.g., age, income, medical conditions)
    receive stronger privacy protection than less sensitive features (e.g., public
    information, aggregated statistics).
    
    :param feature_sensitivity_scores: Dictionary mapping feature names to sensitivity
                                       scores (0.0 to 1.0). Higher values indicate more
                                       sensitive features. If None, sensitivity will be
                                       calculated automatically based on data characteristics.
    :type feature_sensitivity_scores: Dict[str, float], optional
    :param auto_calculate: If True and feature_sensitivity_scores is None, automatically
                          calculate sensitivity based on entropy and value distribution.
                          Default is True.
    :type auto_calculate: bool, optional
    """
    
    def __init__(self, feature_sensitivity_scores: Optional[Dict[str, float]] = None, auto_calculate: bool = True):
        """
        Initialize the sensitivity weight calculator.
        
        :param feature_sensitivity_scores: Manual sensitivity scores for features
        :type feature_sensitivity_scores: Dict[str, float], optional
        :param auto_calculate: Whether to auto-calculate sensitivity if not provided
        :type auto_calculate: bool
        """
        self.feature_sensitivity_scores = feature_sensitivity_scores or {}
        self.auto_calculate = auto_calculate
        self._calculated_scores = {}
        
        if self.feature_sensitivity_scores:
            print(f"Initialized with {len(self.feature_sensitivity_scores)} "
                  f"manual sensitivity scores")
            # Validate scores
            for feature, score in self.feature_sensitivity_scores.items():
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"Sensitivity score for {feature} must be between 0.0 and 1.0")
        else:
            print(f"Initialized with auto-calculation enabled")
    
    def calculate_sensitivity_scores(self, samples: pd.DataFrame, categorical_features: Optional[List] = None) -> Dict[str, float]:
        """
        Calculate sensitivity scores for features based on data characteristics.
        
        Sensitivity is calculated using:
        - Entropy: Higher entropy (more diverse values) indicates higher sensitivity
        - Value distribution: Features with rare values are more sensitive
        - Cardinality: Features with many unique values are more sensitive
        
        Security Rationale:
        Features with high entropy or many unique values contain more information
        and thus pose higher privacy risks if disclosed. These features should
        receive stronger protection.
        
        :param samples: DataFrame containing feature data
        :type samples: pd.DataFrame
        :param categorical_features: List of categorical feature names/indices
        :type categorical_features: List, optional
        :return: Dictionary mapping feature names to sensitivity scores (0.0 to 1.0)
        """
        if not self.auto_calculate and not self.feature_sensitivity_scores:
            print("Auto-calculation disabled and no manual scores provided")
            return {}
        
        print("Calculating sensitivity scores...")
        
        sensitivity_scores = {}
        categorical_set = set(categorical_features or [])
        
        for feature in samples.columns:
            if feature in self.feature_sensitivity_scores:
                # If sensitivity score provided, use them
                sensitivity_scores[feature] = self.feature_sensitivity_scores[feature]
                continue
            
            if feature in categorical_set or samples[feature].dtype == 'object':
                # If Categorical feature, use entropy-based sensitivity
                score = self._calculate_categorical_sensitivity(samples[feature])
            else:
                # If Numerical feature, use distribution-based sensitivity
                score = self._calculate_numerical_sensitivity(samples[feature])
            
            sensitivity_scores[feature] = score
        
        self._calculated_scores = sensitivity_scores
        
        # Calculate sensitivity
        high_sensitivity = {f: s for f, s in sensitivity_scores.items() if s >= 0.7}
        print(f"Calculated sensitivity for {len(sensitivity_scores)} features")
        if high_sensitivity:
            print(f"  High sensitivity features (>=0.7): {list(high_sensitivity.keys())}")
        
        return sensitivity_scores
    
    def _calculate_categorical_sensitivity(self, feature_values: pd.Series) -> float:
        """
        Calculate sensitivity for a categorical feature using entropy.
        
        Higher entropy (more diverse, less predictable values) indicates higher sensitivity.
        
        :param feature_values: Series of categorical values
        :type feature_values: pd.Series
        :return: Sensitivity score between 0.0 and 1.0
        """
        # Remove NaN values
        values = feature_values.dropna()
        
        if len(values) == 0:
            return 0.0
        
        # Calculate entropy
        value_counts = Counter(values)
        total = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        # Normalize entropy 
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Calculate cardinality score
        cardinality_score = min(len(value_counts) / 100.0, 1.0)  # Cap at 100 unique values
        
        # Combine entropy and cardinality
        sensitivity = 0.7 * normalized_entropy + 0.3 * cardinality_score
        
        return min(sensitivity, 1.0)
    
    def _calculate_numerical_sensitivity(self, feature_values: pd.Series) -> float:
        """
        Calculate sensitivity for a numerical feature using distribution characteristics.
        
        Features with high variance, wide ranges, or many unique values are more sensitive.
        
        :param feature_values: Series of numerical values
        :type feature_values: pd.Series
        :return: Sensitivity score between 0.0 and 1.0
        """
        # Remove NaN values
        values = feature_values.dropna()
        
        if len(values) == 0:
            return 0.0
        
        # Calculate coefficient of variation (CV) as measure of dispersion
        mean_val = values.mean()
        std_val = values.std()
        
        if mean_val != 0:
            cv = abs(std_val / mean_val)
        else:
            cv = std_val
        
        # Normalize CV (typical range is 0-2, but can be higher)
        cv_score = min(cv / 2.0, 1.0)
        
        # Calculate range relative to mean
        value_range = values.max() - values.min()
        if mean_val != 0:
            range_score = min(abs(value_range / mean_val) / 10.0, 1.0)
        else:
            range_score = min(value_range / 100.0, 1.0) if value_range > 0 else 0.0
        
        # Calculate cardinality
        unique_count = values.nunique()
        cardinality_score = min(unique_count / 1000.0, 1.0)  # Cap at 1000 unique values
        
        # Combine metrics
        sensitivity = 0.4 * cv_score + 0.3 * range_score + 0.3 * cardinality_score
        
        return min(sensitivity, 1.0)
    
    def get_sensitivity_score(self, feature: str) -> float:
        """
        Get the sensitivity score for a specific feature.
        
        :param feature: Feature name
        :type feature: str
        :return: Sensitivity score (0.0 to 1.0), default 0.5 if not found
        """
        if feature in self.feature_sensitivity_scores:
            return self.feature_sensitivity_scores[feature]
        elif feature in self._calculated_scores:
            return self._calculated_scores[feature]
        else:
            # Default sensitivity for unknown features
            return 0.5
    
    def apply_sensitivity_weight(self, ncp_score: float, feature: str) -> float:
        """
        Apply sensitivity weighting to an NCP score.
        
        Higher sensitivity features get higher weighted NCP scores, making them
        more likely to be generalized and less likely to be removed from generalization.
        
        Security Purpose:
        Ensures that sensitive features receive stronger privacy protection by
        prioritizing their generalization over less sensitive features.
        
        :param ncp_score: Original NCP score for the feature
        :type ncp_score: float
        :param feature: Feature name
        :type feature: str
        :return: Weighted NCP score
        """
        sensitivity = self.get_sensitivity_score(feature)
        
        # Weight formula: weighted_ncp = ncp * (1 + sensitivity)
        # This increases the effective NCP for sensitive features
        weighted_ncp = ncp_score * (1.0 + sensitivity)
        
        return min(weighted_ncp, 1.0)  # Cap at 1.0
    
    def prioritize_features_for_removal(self, features: List[str], feature_ncp_scores: Dict[str, float]) -> List[str]:
        """
        Prioritize features for removal from generalization based on sensitivity.
        
        Lower sensitivity features should be removed first, as they pose less
        privacy risk. This function sorts features by their removal priority.
        
        Security Purpose:
        Ensures that when accuracy needs to be improved by removing features from
        generalization, less sensitive features are removed first, preserving
        privacy protection for sensitive features.
        
        :param features: List of feature names to prioritize
        :type features: List[str]
        :param feature_ncp_scores: Dictionary mapping features to their NCP scores
        :type feature_ncp_scores: Dict[str, float]
        :return: List of features sorted by removal priority (lowest sensitivity first)
        """
        # Calculate priority score: lower sensitivity + lower NCP = higher priority for removal
        priority_scores = []
        
        for feature in features:
            sensitivity = self.get_sensitivity_score(feature)
            ncp = feature_ncp_scores.get(feature, 0.0)
            
            # Priority = (1 - sensitivity) * (1 - ncp)
            # Lower sensitivity and lower NCP = higher priority for removal
            priority = (1.0 - sensitivity) * (1.0 - ncp)
            priority_scores.append((priority, feature))
        
        # Sort by priority (descending) - highest priority = should be removed first
        priority_scores.sort(reverse=True)
        
        return [feature for _, feature in priority_scores]
    
    def get_all_sensitivity_scores(self) -> Dict[str, float]:
        """
        Get all calculated sensitivity scores.
        
        :return: Dictionary mapping feature names to sensitivity scores
        """
        # Combine manual and calculated scores
        all_scores = self._calculated_scores.copy()
        all_scores.update(self.feature_sensitivity_scores)
        return all_scores

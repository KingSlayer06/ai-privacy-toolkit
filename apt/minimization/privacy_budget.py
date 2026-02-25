"""
Privacy Budget Management Module

This module implements privacy budget tracking and enforcement to ensure that privacy
levels (measured via NCP scores) never fall below a specified minimum threshold.
This addresses GDPR compliance requirements by providing a mechanism to guarantee
that data minimization maintains a minimum level of privacy protection.

Security Purpose:
- Prevents privacy degradation below acceptable thresholds
- Provides audit trail of privacy budget consumption
- Enables compliance verification for GDPR data minimization requirements
"""

from typing import Optional, List, Dict
import numpy as np


class PrivacyBudgetTracker:
    """
    Tracks and enforces minimum privacy thresholds during the generalization process.
    
    This class monitors the Normalized Certainty Penalty (NCP) scores throughout
    the data minimization process and ensures that privacy never falls below a
    specified minimum threshold. This is critical for GDPR compliance, as it provides
    a mechanism to demonstrate that privacy protections are maintained.
    
    Security Mechanism:
    The tracker maintains a history of NCP scores and prevents operations that would
    cause privacy to drop below the minimum threshold. This ensures that even when
    optimizing for accuracy, privacy protections are never compromised.
    
    :param min_privacy_threshold: Minimum acceptable NCP score (0.0 to 1.0).
                                  Higher values indicate better privacy protection.
                                  Default is 0.0.
    :type min_privacy_threshold: float, optional
    """
    
    def __init__(self, min_privacy_threshold: float = 0.0):
        """
        Initialize the privacy budget tracker.
        
        :param min_privacy_threshold: Minimum acceptable NCP score
        :type min_privacy_threshold: float
        """
        if not 0.0 <= min_privacy_threshold <= 1.0:
            raise ValueError("min_privacy_threshold must be between 0.0 and 1.0")
        
        self.min_privacy_threshold = min_privacy_threshold
        self.privacy_history: List[float] = []
        self.operation_history: List[str] = []
        self._current_ncp: Optional[float] = None
        
        print(f"Initialized with minimum privacy threshold: {min_privacy_threshold:.4f}")
    
    def check_privacy_threshold(self, current_ncp: float) -> bool:
        """
        Check if the current NCP score meets the minimum privacy threshold.
        
        This function is called before any generalization operation to ensure
        that privacy levels remain acceptable. If privacy would fall below
        the threshold, the operation should be rejected or rolled back.
        
        Security Purpose:
        Prevents privacy degradation by enforcing a hard lower bound on NCP scores.
        This ensures GDPR compliance by guaranteeing minimum privacy protection.
        
        :param current_ncp: Current Normalized Certainty Penalty score
        :type current_ncp: float
        :return: True if privacy threshold is met, False otherwise
        """
        self._current_ncp = current_ncp
        self.privacy_history.append(current_ncp)
        
        threshold_met = current_ncp >= self.min_privacy_threshold
        
        if not threshold_met:
            print(f"WARNING: Privacy threshold violation detected!")
            print(f"  Current NCP: {current_ncp:.4f}, Minimum required: {self.min_privacy_threshold:.4f}")
            print(f"  Privacy degradation: {self.min_privacy_threshold - current_ncp:.4f}")
        else:
            privacy_margin = current_ncp - self.min_privacy_threshold
            print(f"Privacy threshold check passed. NCP: {current_ncp:.4f}, "
                  f"Margin: {privacy_margin:.4f}")
        
        return threshold_met
    
    def record_operation(self, operation: str, ncp_before: float, ncp_after: float):
        """
        Record a generalization operation and its impact on privacy.
        
        This creates an audit trail of all privacy-affecting operations, which
        is essential for compliance verification and debugging.
        
        :param operation: Description of the operation
        :type operation: str
        :param ncp_before: NCP score before the operation
        :type ncp_before: float
        :param ncp_after: NCP score after the operation
        :type ncp_after: float
        """
        privacy_change = ncp_after - ncp_before
        self.operation_history.append({
            'operation': operation,
            'ncp_before': ncp_before,
            'ncp_after': ncp_after,
            'privacy_change': privacy_change
        })
        
        print(f"Operation recorded: {operation}")
        print(f"  NCP before: {ncp_before:.4f}, NCP after: {ncp_after:.4f}, "
              f"Change: {privacy_change:+.4f}")
    
    def get_current_privacy(self) -> Optional[float]:
        """
        Get the current privacy level (NCP score).
        
        :return: Current NCP score, or None if no operations have been performed
        """
        return self._current_ncp
    
    def get_privacy_margin(self) -> Optional[float]:
        """
        Calculate the margin between current privacy and minimum threshold.
        
        This indicates how much "privacy budget" remains before hitting the threshold.
        Negative values indicate a threshold violation.
        
        :return: Privacy margin, or None if no operations performed
        """
        if self._current_ncp is None:
            return None
        return self._current_ncp - self.min_privacy_threshold
    
    def get_privacy_history(self) -> List[float]:
        """
        Get the complete history of NCP scores.
        
        :return: List of NCP scores in chronological order
        """
        return self.privacy_history.copy()
    
    def get_operation_history(self) -> List[Dict]:
        """
        Get the complete audit trail of operations.
        
        :return: List of operation records
        """
        return self.operation_history.copy()
    
    def reset(self):
        """
        Reset the privacy budget tracker.
        
        Useful when restarting the generalization process.
        """
        self.privacy_history = []
        self.operation_history = []
        self._current_ncp = None
        print("Reset: History cleared, threshold maintained")

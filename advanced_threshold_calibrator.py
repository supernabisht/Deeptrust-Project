# advanced_threshold_calibrator.py
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

class AdvancedThresholdCalibrator:
    def __init__(self, method='isotonic', n_bins=10):
        """
        Initialize the threshold calibrator.
        
        Args:
            method: 'isotonic' for isotonic regression or 'sigmoid' for Platt scaling
            n_bins: Number of bins for calibration curve
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrator = None
        self.threshold = 0.5  # Default threshold
        
    def calibrate(self, y_true, y_prob, sample_weight=None):
        """
        Calibrate the threshold based on true labels and predicted probabilities.
        
        Args:
            y_true: Array of true binary labels (0 or 1)
            y_prob: Array of predicted probabilities
            sample_weight: Optional array of sample weights
            
        Returns:
            float: Optimal threshold
        """
        if len(np.unique(y_true)) < 2:
            raise ValueError("Need samples of at least 2 classes for calibration")
            
        if self.method == 'isotonic':
            # Use isotonic regression for calibration
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, 
                n_bins=self.n_bins,
                strategy='quantile'
            )
            # Simple threshold finding - could be enhanced
            self.threshold = np.median(prob_pred)
        elif self.method == 'sigmoid':
            # Platt scaling
            lr = LogisticRegression(C=1e5, solver='lbfgs')
            lr.fit(y_prob.reshape(-1, 1), y_true)
            self.calibrator = lr
            # Set threshold where P(y=1) = 0.5
            self.threshold = (0.5 - lr.intercept_[0]) / lr.coef_[0][0]
        else:
            raise ValueError("Unsupported calibration method. Use 'isotonic' or 'sigmoid'")
            
        return self.threshold
    
    def predict(self, y_prob):
        """
        Predict binary labels using calibrated threshold.
        
        Args:
            y_prob: Array of predicted probabilities
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if self.calibrator is not None and self.method == 'sigmoid':
            # Apply Platt scaling if using sigmoid calibration
            y_prob = self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
            
        return (y_prob >= self.threshold).astype(int)
    
    def get_calibration_curve(self, y_true, y_prob):
        """
        Get calibration curve data.
        
        Args:
            y_true: Array of true binary labels (0 or 1)
            y_prob: Array of predicted probabilities
            
        Returns:
            Tuple of (prob_true, prob_pred)
        """
        return calibration_curve(y_true, y_prob, n_bins=self.n_bins)
    
    def get_brier_score(self, y_true, y_prob):
        """
        Calculate Brier score loss.
        
        Args:
            y_true: Array of true binary labels (0 or 1)
            y_prob: Array of predicted probabilities
            
        Returns:
            float: Brier score
        """
        return brier_score_loss(y_true, y_prob)
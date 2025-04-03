import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
)

class MetricsTracker:
    """Class for tracking and computing various metrics.
    
    Handles computation of classification and regression metrics.
    """
    
    def __init__(self, task_type='classification', n_classes=None, average='macro'):
        """Initialize the metrics tracker.
        
        Args:
            task_type: Type of task ('classification' or 'regression')
            n_classes: Number of classes for classification tasks
            average: Method for averaging scores in multi-class classification
        """
        self.task_type = task_type.lower()
        self.n_classes = n_classes
        self.average = average
        
        # Initialize metric values
        self.reset()
        
        # Validate parameters
        if self.task_type == 'classification' and self.n_classes is None:
            raise ValueError("n_classes must be specified for classification tasks")
    
    def reset(self):
        """Reset all metrics."""
        self.true_labels = []
        self.predictions = []
        self.probabilities = [] if self.task_type == 'classification' else None
        
        # Initialize metric dictionaries
        if self.task_type == 'classification':
            self.metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'confusion_matrix': None,
                'auroc': 0.0 if self.n_classes == 2 else None
            }
        else:  # regression
            self.metrics = {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2': 0.0
            }
    
    def update(self, y_true, y_pred, probabilities=None):
        """Update metrics with batch predictions.
        
        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted labels/values
            probabilities: Predicted probabilities for classification (optional)
        """
        # Convert tensors to numpy if needed
        y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        
        # Flatten arrays if needed
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        # Update stored values
        self.true_labels.extend(y_true)
        self.predictions.extend(y_pred)
        
        # Update probabilities for classification
        if self.task_type == 'classification' and probabilities is not None:
            probabilities = probabilities.cpu().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(-1, 1)
                # For binary classification, add the complement probability
                probabilities = np.hstack([1 - probabilities, probabilities])
            
            self.probabilities.extend(probabilities)
    
    def compute(self):
        """Compute all metrics based on stored predictions.
        
        Returns:
            Dictionary of metrics
        """
        # Convert lists to numpy arrays
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        if self.task_type == 'classification':
            # Compute classification metrics
            self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle case where some classes may not be present in the batch
            try:
                self.metrics['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
                self.metrics['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
                self.metrics['f1'] = f1_score(y_true, y_pred, average=self.average, zero_division=0)
            except Exception as e:
                print(f"Warning: Error computing precision/recall/f1: {e}")
                self.metrics['precision'] = 0.0
                self.metrics['recall'] = 0.0
                self.metrics['f1'] = 0.0
            
            # Compute confusion matrix
            self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, 
                                                               labels=range(self.n_classes))
            
            # Compute AUROC for binary classification if probabilities are available
            if self.n_classes == 2 and self.probabilities is not None and len(self.probabilities) > 0:
                try:
                    probs = np.array(self.probabilities)[:, 1]  # Probability of positive class
                    self.metrics['auroc'] = roc_auc_score(y_true, probs)
                except Exception as e:
                    print(f"Warning: Error computing AUROC: {e}")
                    self.metrics['auroc'] = 0.0
            
        else:  # regression metrics
            self.metrics['mse'] = mean_squared_error(y_true, y_pred)
            self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
            self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
            
            # Compute R² (coefficient of determination)
            y_mean = np.mean(y_true)
            ss_total = np.sum((y_true - y_mean) ** 2)
            ss_residual = np.sum((y_true - y_pred) ** 2)
            self.metrics['r2'] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        return self.metrics
    
    def print_metrics(self):
        """Print all computed metrics in a formatted way."""
        metrics = self.compute()
        
        print("\n------- Metrics Summary -------")
        
        for metric_name, value in metrics.items():
            if metric_name == 'confusion_matrix':
                print(f"\nConfusion Matrix:")
                print(value)
            elif value is not None:
                print(f"{metric_name.capitalize():>10}: {value:.4f}")
        
        print("------------------------------\n")

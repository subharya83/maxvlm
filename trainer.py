#!/usr/bin/env python3
"""
Train models for materials property prediction

This module trains LightGBM models for predicting:
1. Magnetic ordering (FM vs FiM) - Classification
2. Magnetic moment per atom - Regression  
3. Formation energy per atom - Regression

The script trains models with two different train/test splits:
- 90:10 split: More training data, better for generalization
- 70:30 split: More testing data, useful for model validation

Training Data Format:
    Expected pickle file with pandas DataFrame containing:
    - 'outer_product': 518-dimensional feature vectors
    - 'ordering': Binary labels (0=FM, 1=FiM)
    - 'total_magnetization_normalized_atoms': Magnetic moments
    - 'formation_energy_per_atom': Formation energies

Model Hyperparameters:
    Optimized for materials science datasets with:
    - Regularization to prevent overfitting
    - Class weights for imbalanced datasets
    - Conservative learning rates for stability

Author: Materials Prediction Pipeline
Version: 2.0
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr
from lightgbm import LGBMClassifier, LGBMRegressor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when training encounters errors"""
    pass


def prepare_element_features(element_csv: str) -> pd.DataFrame:
    """
    Prepare element feature vectors from CSV file.
    
    This function creates 16-dimensional feature vectors for each element
    by combining 8 atomic properties and their squared values.
    
    Args:
        element_csv: Path to element properties CSV file
        
    Returns:
        DataFrame with columns ['Element', 'Vector']
        
            Raises:
            TrainingError: If training fails
    """
    logger.info("Training magnetic ordering classifier...")
    
    # Validate and prepare data
    X, y = validate_training_data(features_df, 'ordering')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}, "
                f"Test: {y_test.value_counts().to_dict()}")
    
    # Set default class weights if not provided (balance FiM minority class)
    if class_weight is None:
        class_weight = {0: 1, 1: 3}
    
    # Train model
    try:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight=class_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1  # Suppress training output
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_report': classification_report(y_train, y_train_pred),
        'test_report': classification_report(y_test, y_test_pred)
    }
    
    return model, metrics


def train_moment_regressor(
    features_df: pd.DataFrame,
    split_ratio: float = 0.1,
    n_estimators: int = 200,
    learning_rate: float = 0.08,
    max_depth: int = 8,
    num_leaves: int = 18,
    subsample: float = 0.5,
    reg_alpha: float = 1.0,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 0.25,
    min_child_samples: int = 15,
    random_state: int = 42
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """
    Train magnetic moment regressor.
    
    Args:
        features_df: DataFrame with features and moment labels
        split_ratio: Test set ratio (default: 0.1 for 90:10 split)
        n_estimators: Number of boosting iterations
        learning_rate: Boosting learning rate
        max_depth: Maximum tree depth
        num_leaves: Maximum number of leaves per tree
        subsample: Subsample ratio of training instances
        reg_alpha: L1 regularization term
        reg_lambda: L2 regularization term
        colsample_bytree: Subsample ratio of columns
        min_child_samples: Minimum samples per leaf
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Raises:
        TrainingError: If training fails
    """
    logger.info("Training magnetic moment regressor...")
    
    # Validate and prepare data
    X, y = validate_training_data(features_df, 'moment')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state
    )
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Moment range - Train: [{y_train.min():.3f}, {y_train.max():.3f}], "
                f"Test: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Train model
    try:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            random_state=random_state,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate metrics
    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Pearson correlation
    train_corr, _ = pearsonr(y_train, y_train_pred)
    test_corr, _ = pearsonr(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, metrics


def train_formation_regressor(
    features_df: pd.DataFrame,
    split_ratio: float = 0.1,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 12,
    num_leaves: int = 30,
    subsample: float = 0.75,
    reg_alpha: float = 1.0,
    reg_lambda: float = 1.0,
    colsample_bytree: float = 0.6,
    min_child_samples: int = 60,
    random_state: int = 42
) -> Tuple[LGBMRegressor, Dict[str, Any]]:
    """
    Train formation energy regressor.
    
    Args:
        features_df: DataFrame with features and formation energy labels
        split_ratio: Test set ratio (default: 0.1 for 90:10 split)
        n_estimators: Number of boosting iterations
        learning_rate: Boosting learning rate
        max_depth: Maximum tree depth
        num_leaves: Maximum number of leaves per tree
        subsample: Subsample ratio of training instances
        reg_alpha: L1 regularization term
        reg_lambda: L2 regularization term
        colsample_bytree: Subsample ratio of columns
        min_child_samples: Minimum samples per leaf
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Raises:
        TrainingError: If training fails
    """
    logger.info("Training formation energy regressor...")
    
    # Validate and prepare data
    X, y = validate_training_data(features_df, 'formation')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state
    )
    
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Energy range - Train: [{y_train.min():.3f}, {y_train.max():.3f}], "
                f"Test: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    # Train model
    try:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            random_state=random_state,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    except Exception as e:
        raise TrainingError(f"Model training failed: {e}")
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate metrics
    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Pearson correlation
    train_corr, _ = pearsonr(y_train, y_train_pred)
    test_corr, _ = pearsonr(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, metrics


def save_model(model: Any, filepath: Path, metadata: Optional[Dict] = None) -> None:
    """
    Save model to disk with optional metadata.
    
    Args:
        model: Trained model to save
        filepath: Path to save model
        metadata: Optional metadata to include
    """
    try:
        save_obj = {'model': model}
        if metadata:
            save_obj['metadata'] = metadata
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved model to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description='Train materials property prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python trainer.py --features training_data.pkl --elements elements.csv
  
  # Train only ordering classifier
  python trainer.py --features training_data.pkl --elements elements.csv --tasks ordering
  
  # Train multiple specific tasks
  python trainer.py --features training_data.pkl --elements elements.csv --tasks ordering moment
  
  # Save to custom directory
  python trainer.py --features training_data.pkl --elements elements.csv --weights-dir my_models

Notes:
  - Training data must be prepared using pipeline.py
  - Element features are saved for reference but not used in training
  - Models are saved with standardized naming convention
  - Two splits are trained: 90:10 (generalization) and 70:30 (validation)
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to features pickle file with compound data')
    parser.add_argument('--elements', required=True,
                       help='Path to element properties CSV file')
    parser.add_argument('--weights-dir', default='weights',
                       help='Directory to save model weights (default: weights)')
    parser.add_argument('--tasks', nargs='+', 
                       choices=['ordering', 'moment', 'formation', 'all'],
                       default=['all'],
                       help='Tasks to train (default: all)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.features).exists():
        logger.error(f"Features file not found: {args.features}")
        return 1
    
    if not Path(args.elements).exists():
        logger.error(f"Elements file not found: {args.elements}")
        return 1
    
    # Create weights directory
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(exist_ok=True)
    logger.info(f"Model weights will be saved to: {weights_dir.absolute()}")
    
    print(f"\n{'='*80}")
    print("MATERIALS PROPERTY PREDICTION - MODEL TRAINING")
    print(f"{'='*80}\n")
    
    # Prepare element features
    try:
        logger.info("Preparing element features...")
        element_df = prepare_element_features(args.elements)
        logger.info(f"Loaded {len(element_df)} elements")
        
        # Save element features
        with open(weights_dir / 'element_features.pkl', 'wb') as f:
            pickle.dump(element_df, f)
        logger.info(f"Saved element features to {weights_dir / 'element_features.pkl'}\n")
        
    except Exception as e:
        logger.error(f"Error preparing element features: {e}")
        return 1
    
    # Load compound features
    try:
        logger.info(f"Loading compound features from {args.features}...")
        with open(args.features, 'rb') as f:
            features_data = pickle.load(f)
        
        if not isinstance(features_data, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(features_data)}")
            return 1
        
        logger.info(f"Loaded features for {len(features_data)} compounds\n")
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return 1
    
    # Determine tasks to train
    tasks = args.tasks if 'all' not in args.tasks else ['ordering', 'moment', 'formation']
    
    # Train models
    success_count = 0
    total_models = len(tasks) * 2  # Two splits per task
    
    if 'ordering' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC ORDERING CLASSIFIER")
        print(f"{'='*80}\n")
        
        try:
            # 90:10 split
            logger.info("Training with 90:10 train/test split...")
            model_90_10, metrics_90_10 = train_ordering_classifier(
                features_data, split_ratio=0.1
            )
            
            print(f"Train Accuracy: {metrics_90_10['train_accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_90_10['test_accuracy']:.4f}")
            print(f"Train AUC: {metrics_90_10['train_auc']:.4f}")
            print(f"Test AUC: {metrics_90_10['test_auc']:.4f}")
            print(f"\nTest Classification Report:\n{metrics_90_10['test_report']}")
            
            # Save model
            save_model(model_90_10, weights_dir / 'magnetic_ordering_model_90_10.pkl')
            success_count += 1
            
            # 70:30 split
            logger.info("\nTraining with 70:30 train/test split...")
            model_70_30, metrics_70_30 = train_ordering_classifier(
                features_data, split_ratio=0.3,
                class_weight={0: 2, 1: 7},
                n_estimators=25,
                max_depth=8,
                num_leaves=9,
                min_child_samples=12,
                subsample=0.4
            )
            
            print(f"Train Accuracy: {metrics_70_30['train_accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_70_30['test_accuracy']:.4f}")
            print(f"Train AUC: {metrics_70_30['train_auc']:.4f}")
            print(f"Test AUC: {metrics_70_30['test_auc']:.4f}\n")
            
            # Save model
            save_model(model_70_30, weights_dir / 'magnetic_ordering_model_70_30.pkl')
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error training ordering classifier: {e}")
    
    if 'moment' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC MOMENT REGRESSOR")
        print(f"{'='*80}\n")
        
        try:
            # 90:10 split
            logger.info("Training with 90:10 train/test split...")
            model_90_10, metrics_90_10 = train_moment_regressor(
                features_data, split_ratio=0.1
            )
            
            print(f"Train RMSE: {metrics_90_10['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
            print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
            print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}\n")
            
            # Save model
            save_model(model_90_10, weights_dir / 'magnetic_moment_model_90_10.pkl')
            success_count += 1
            
            # 70:30 split
            logger.info("Training with 70:30 train/test split...")
            model_70_30, metrics_70_30 = train_moment_regressor(
                features_data, split_ratio=0.3,
                n_estimators=75,
                num_leaves=12,
                subsample=0.30,
                reg_alpha=0.5,
                reg_lambda=0.5,
                colsample_bytree=0.3
            )
            
            print(f"Train RMSE: {metrics_70_30['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_70_30['test_rmse']:.4f}")
            print(f"Train R²: {metrics_70_30['train_r2']:.4f}")
            print(f"Test R²: {metrics_70_30['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_70_30['test_corr']:.4f}\n")
            
            # Save model
            save_model(model_70_30, weights_dir / 'magnetic_moment_model_70_30.pkl')
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error training moment regressor: {e}")
    
    if 'formation' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING FORMATION ENERGY REGRESSOR")
        print(f"{'='*80}\n")
        
        try:
            # 90:10 split
            logger.info("Training with 90:10 train/test split...")
            model_90_10, metrics_90_10 = train_formation_regressor(
                features_data, split_ratio=0.1
            )
            
            print(f"Train RMSE: {metrics_90_10['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
            print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
            print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}\n")
            
            # Save model
            save_model(model_90_10, weights_dir / 'formation_energy_model_90_10.pkl')
            success_count += 1
            
            # 70:30 split
            logger.info("Training with 70:30 train/test split...")
            model_70_30, metrics_70_30 = train_formation_regressor(
                features_data, split_ratio=0.3
            )
            
            print(f"Train RMSE: {metrics_70_30['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_70_30['test_rmse']:.4f}")
            print(f"Train R²: {metrics_70_30['train_r2']:.4f}")
            print(f"Test R²: {metrics_70_30['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_70_30['test_corr']:.4f}\n")
            
            # Save model
            save_model(model_70_30, weights_dir / 'formation_energy_model_70_30.pkl')
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error training formation regressor: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Successfully trained {success_count}/{total_models} models")
    print(f"All models saved to: {weights_dir.absolute()}\n")
    
    return 0 if success_count == total_models else 1


if __name__ == '__main__':
    exit(main())
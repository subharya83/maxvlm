#!/usr/bin/env python3
"""
Train models for materials property prediction

Supports:
- Default: 518-dimensional feature vector
- With -x: Extended feature vector (>1367 dimensions) with tuning and feature selection
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingError(Exception):
    pass


def validate_training_data(features_df: pd.DataFrame, task: str, extended: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(features_df, pd.DataFrame):
        raise TrainingError(f"Expected DataFrame, got {type(features_df)}")
    required_columns = ['outer_product']
    target_columns = {
        'ordering': 'ordering',
        'moment': 'total_magnetization_normalized_atoms',
        'formation': 'formation_energy_per_atom'
    }
    target_col = target_columns.get(task)
    if not target_col or target_col not in features_df.columns:
        raise TrainingError(f"Missing column: {target_col}")
    initial_count = len(features_df)
    features_df = features_df.dropna(subset=[target_col])
    if len(features_df) < initial_count:
        logger.warning(f"Dropped {initial_count - len(features_df)} rows with missing {target_col}")
    if len(features_df) < 10:
        raise TrainingError(f"Too few samples: {len(features_df)}")
    X = np.stack(features_df['outer_product'].values)
    y = features_df[target_col].values
    if not extended and X.shape[1] != 518:
        raise TrainingError(f"Expected 518 features, got {X.shape[1]}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features for {task}")
    return X, y


def select_features(model, X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    importances = model.feature_importances_
    feature_indices = np.argsort(importances)[::-1][:min(n_features, X.shape[1])]
    logger.info(f"Selected {len(feature_indices)} features")
    return feature_indices


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, task: str, split: str, weights_dir: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel(f"Actual {task}")
    plt.ylabel(f"Predicted {task}")
    plt.title(f"{task.capitalize()} Prediction ({split} Split)")
    plt.savefig(weights_dir / f"{task}_prediction_{split}.png")
    plt.close()
    logger.info(f"Saved prediction plot to {weights_dir / f'{task}_prediction_{split}.png'}")


def plot_feature_importance(model, task: str, split: str, weights_dir: Path, top_n: int = 20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [f"Feature {i}" for i in indices], rotation=45)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title(f"Top {top_n} Feature Importances for {task.capitalize()} ({split} Split)")
    plt.tight_layout()
    plt.savefig(weights_dir / f"{task}_importance_{split}.png")
    plt.close()
    logger.info(f"Saved feature importance plot to {weights_dir / f'{task}_importance_{split}.png'}")


def train_ordering_classifier(
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
    random_state: int = 42,
    class_weight: dict = None,
    extended: bool = False,
    n_features: int = 100
) -> tuple[LGBMClassifier, dict]:
    logger.info("Training magnetic ordering classifier...")
    X, y = validate_training_data(features_df, 'ordering', extended)
    class_weight = {0: len(y) / (2 * pd.Series(y).value_counts().get(0, 1)),
                    1: len(y) / (2 * pd.Series(y).value_counts().get(1, 1))} if extended else {0: 1, 1: 3}
    logger.info(f"Class weights: {class_weight}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Class distribution - Train: {pd.Series(y_train).value_counts().to_dict()}")
    if extended:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [6, 8, 10],
            'num_leaves': [15, 18, 21]
        }
        base_model = LGBMClassifier(
            subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            random_state=random_state, verbose=-1
        )
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True
        )
        try:
            grid_search.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc',
                early_stopping_rounds=10, verbose=False
            )
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            selected_features = select_features(model, X_train, y_train, n_features)
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
            model.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc',
                early_stopping_rounds=10, verbose=False
            )
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
    else:
        try:
            model = LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                num_leaves=num_leaves, subsample=subsample, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, colsample_bytree=colsample_bytree,
                min_child_samples=min_child_samples, random_state=random_state,
                class_weight=class_weight, verbose=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
        selected_features = None
    logger.info("Model training completed")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    metrics = {
        'train_accuracy': train_acc, 'test_accuracy': test_acc,
        'train_auc': train_auc, 'test_auc': test_auc,
        'train_size': len(X_train), 'test_size': len(X_test),
        'train_report': classification_report(y_train, y_train_pred),
        'test_report': classification_report(y_test, y_test_pred)
    }
    if extended:
        metrics['best_params'] = grid_search.best_params_ if 'grid_search' in locals() else {}
        metrics['selected_features'] = selected_features
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
    random_state: int = 42,
    extended: bool = False,
    n_features: int = 100
) -> tuple[LGBMRegressor, dict]:
    logger.info("Training magnetic moment regressor...")
    X, y = validate_training_data(features_df, 'moment', extended)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state
    )
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Moment range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    if extended:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [6, 8, 10],
            'num_leaves': [15, 18, 21]
        }
        base_model = LGBMRegressor(
            subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            random_state=random_state, verbose=-1
        )
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True
        )
        try:
            grid_search.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                early_stopping_rounds=10, verbose=False
            )
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            selected_features = select_features(model, X_train, y_train, n_features)
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
            model.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                early_stopping_rounds=10, verbose=False
            )
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
    else:
        try:
            model = LGBMRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                num_leaves=num_leaves, subsample=subsample, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, colsample_bytree=colsample_bytree,
                min_child_samples=min_child_samples, random_state=random_state, verbose=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
        selected_features = None
    logger.info("Model training completed")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_corr, _ = pearsonr(y_train, y_train_pred)
    test_corr, _ = pearsonr(y_test, y_test_pred)
    metrics = {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_corr': train_corr, 'test_corr': test_corr,
        'train_size': len(X_train), 'test_size': len(X_test)
    }
    if extended:
        metrics['best_params'] = grid_search.best_params_ if 'grid_search' in locals() else {}
        metrics['selected_features'] = selected_features
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
    random_state: int = 42,
    extended: bool = False,
    n_features: int = 100
) -> tuple[LGBMRegressor, dict]:
    logger.info("Training formation energy regressor...")
    X, y = validate_training_data(features_df, 'formation', extended)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state
    )
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    logger.info(f"Energy range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    if extended:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [10, 12, 14],
            'num_leaves': [25, 30, 35]
        }
        base_model = LGBMRegressor(
            subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree, min_child_samples=min_child_samples,
            random_state=random_state, verbose=-1
        )
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True
        )
        try:
            grid_search.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                early_stopping_rounds=10, verbose=False
            )
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            selected_features = select_features(model, X_train, y_train, n_features)
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
            model.fit(
                X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                early_stopping_rounds=10, verbose=False
            )
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
    else:
        try:
            model = LGBMRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                num_leaves=num_leaves, subsample=subsample, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, colsample_bytree=colsample_bytree,
                min_child_samples=min_child_samples, random_state=random_state, verbose=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            raise TrainingError(f"Model training failed: {e}")
        selected_features = None
    logger.info("Model training completed")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    train_rmse = rmse(y_train, y_train_pred)
    test_rmse = rmse(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_corr, _ = pearsonr(y_train, y_train_pred)
    test_corr, _ = pearsonr(y_test, y_test_pred)
    metrics = {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_corr': train_corr, 'test_corr': test_corr,
        'train_size': len(X_train), 'test_size': len(X_test)
    }
    if extended:
        metrics['best_params'] = grid_search.best_params_ if 'grid_search' in locals() else {}
        metrics['selected_features'] = selected_features
    return model, metrics


def save_model(model, filepath: Path, metadata: dict = None):
    try:
        save_obj = {'model': model, 'metadata': metadata or {}}
        with open(filepath, 'wb') as f:
            pickle.dump(save_obj, f)
        logger.info(f"Saved model to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train Materials Property Models')
    parser.add_argument('-f', required=True, help='Features pickle file')
    parser.add_argument('-e', required=True, help='Element properties CSV')
    parser.add_argument('-w', default='weights', help='Model weights directory (default: weights)')
    parser.add_argument('-t', nargs='+', choices=['ordering', 'moment', 'formation', 'all'],
                       default=['all'], help='Tasks to train (default: all)')
    parser.add_argument('-n', type=int, default=100, help='Number of features to select with -x (default: 100)')
    parser.add_argument('-x', action='store_true', help='Use extended feature processing')
    args = parser.parse_args()

    if not Path(args.f).exists():
        logger.error(f"Features file not found: {args.f}")
        return 1
    if not Path(args.e).exists():
        logger.error(f"Elements file not found: {args.e}")
        return 1

    weights_dir = Path(args.w)
    weights_dir.mkdir(exist_ok=True)
    logger.info(f"Model weights will be saved to: {weights_dir.absolute()}")

    print(f"\n{'='*80}")
    print(f"MATERIALS PROPERTY PREDICTION - MODEL TRAINING ({'EXTENDED' if args.x else 'ORIGINAL'})")
    print(f"{'='*80}\n")

    logger.info(f"Loading features from {args.f}...")
    with open(args.f, 'rb') as f:
        features_data = pickle.load(f)
    if not isinstance(features_data, pd.DataFrame):
        logger.error(f"Expected DataFrame, got {type(features_data)}")
        return 1
    logger.info(f"Loaded features for {len(features_data)} compounds with {len(features_data['outer_product'].iloc[0])} features\n")

    tasks = args.t if 'all' not in args.t else ['ordering', 'moment', 'formation']
    total_models = len(tasks) * 2
    success_count = 0

    if 'ordering' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC ORDERING CLASSIFIER")
        print(f"{'='*80}\n")
        try:
            model_90_10, metrics_90_10 = train_ordering_classifier(
                features_data, split_ratio=0.1, extended=args.x, n_features=args.n
            )
            print(f"90:10 Split - Train Accuracy: {metrics_90_10['train_accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_90_10['test_accuracy']:.4f}")
            print(f"Train AUC: {metrics_90_10['train_auc']:.4f}")
            print(f"Test AUC: {metrics_90_10['test_auc']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_90_10['best_params']}")
            print(f"\nTest Classification Report:\n{metrics_90_10['test_report']}")
            save_model(model_90_10, weights_dir / 'magnetic_ordering_model_90_10.pkl', metadata=metrics_90_10)
            if args.x:
                plot_feature_importance(model_90_10, 'ordering', '90_10', weights_dir)
            success_count += 1

            model_70_30, metrics_70_30 = train_ordering_classifier(
                features_data, split_ratio=0.3, extended=args.x, n_features=args.n,
                class_weight={0: 2, 1: 7} if not args.x else None,
                n_estimators=25, max_depth=8, num_leaves=9, min_child_samples=12, subsample=0.4
            )
            print(f"\n70:30 Split - Train Accuracy: {metrics_70_30['train_accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_70_30['test_accuracy']:.4f}")
            print(f"Train AUC: {metrics_70_30['train_auc']:.4f}")
            print(f"Test AUC: {metrics_70_30['test_auc']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_70_30['best_params']}")
            print(f"\nTest Classification Report:\n{metrics_70_30['test_report']}")
            save_model(model_70_30, weights_dir / 'magnetic_ordering_model_70_30.pkl', metadata=metrics_70_30)
            if args.x:
                plot_feature_importance(model_70_30, 'ordering', '70_30', weights_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Error training ordering classifier: {e}")

    if 'moment' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC MOMENT REGRESSOR")
        print(f"{'='*80}\n")
        try:
            model_90_10, metrics_90_10 = train_moment_regressor(
                features_data, split_ratio=0.1, extended=args.x, n_features=args.n
            )
            print(f"90:10 Split - Train RMSE: {metrics_90_10['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
            print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
            print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_90_10['best_params']}")
            save_model(model_90_10, weights_dir / 'magnetic_moment_model_90_10.pkl', metadata=metrics_90_10)
            if args.x:
                plot_feature_importance(model_90_10, 'moment', '90_10', weights_dir)
                X_full = np.stack(features_data['outer_product'].values)[:, metrics_90_10['selected_features']] if metrics_90_10.get('selected_features') is not None else np.stack(features_data['outer_product'].values)
                plot_predictions(
                    features_data['total_magnetization_normalized_atoms'].values,
                    model_90_10.predict(X_full),
                    'moment', '90_10', weights_dir
                )
            success_count += 1

            model_70_30, metrics_70_30 = train_moment_regressor(
                features_data, split_ratio=0.3, extended=args.x, n_features=args.n,
                n_estimators=75, num_leaves=12, subsample=0.30, reg_alpha=0.5, reg_lambda=0.5, colsample_bytree=0.3
            )
            print(f"\n70:30 Split - Train RMSE: {metrics_70_30['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_70_30['test_rmse']:.4f}")
            print(f"Train R²: {metrics_70_30['train_r2']:.4f}")
            print(f"Test R²: {metrics_70_30['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_70_30['test_corr']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_70_30['best_params']}")
            save_model(model_70_30, weights_dir / 'magnetic_moment_model_70_30.pkl', metadata=metrics_70_30)
            if args.x:
                plot_feature_importance(model_70_30, 'moment', '70_30', weights_dir)
                X_full = np.stack(features_data['outer_product'].values)[:, metrics_70_30['selected_features']] if metrics_70_30.get('selected_features') is not None else np.stack(features_data['outer_product'].values)
                plot_predictions(
                    features_data['total_magnetization_normalized_atoms'].values,
                    model_70_30.predict(X_full),
                    'moment', '70_30', weights_dir
                )
            success_count += 1
        except Exception as e:
            logger.error(f"Error training moment regressor: {e}")

    if 'formation' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING FORMATION ENERGY REGRESSOR")
        print(f"{'='*80}\n")
        try:
            model_90_10, metrics_90_10 = train_formation_regressor(
                features_data, split_ratio=0.1, extended=args.x, n_features=args.n
            )
            print(f"90:10 Split - Train RMSE: {metrics_90_10['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
            print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
            print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_90_10['best_params']}")
            save_model(model_90_10, weights_dir / 'formation_energy_model_90_10.pkl', metadata=metrics_90_10)
            if args.x:
                plot_feature_importance(model_90_10, 'formation', '90_10', weights_dir)
                X_full = np.stack(features_data['outer_product'].values)[:, metrics_90_10['selected_features']] if metrics_90_10.get('selected_features') is not None else np.stack(features_data['outer_product'].values)
                plot_predictions(
                    features_data['formation_energy_per_atom'].values,
                    model_90_10.predict(X_full),
                    'formation', '90_10', weights_dir
                )
            success_count += 1

            model_70_30, metrics_70_30 = train_formation_regressor(
                features_data, split_ratio=0.3, extended=args.x, n_features=args.n
            )
            print(f"\n70:30 Split - Train RMSE: {metrics_70_30['train_rmse']:.4f}")
            print(f"Test RMSE: {metrics_70_30['test_rmse']:.4f}")
            print(f"Train R²: {metrics_70_30['train_r2']:.4f}")
            print(f"Test R²: {metrics_70_30['test_r2']:.4f}")
            print(f"Test Correlation: {metrics_70_30['test_corr']:.4f}")
            if args.x:
                print(f"Best Parameters: {metrics_70_30['best_params']}")
            save_model(model_70_30, weights_dir / 'formation_energy_model_70_30.pkl', metadata=metrics_70_30)
            if args.x:
                plot_feature_importance(model_70_30, 'formation', '70_30', weights_dir)
                X_full = np.stack(features_data['outer_product'].values)[:, metrics_70_30['selected_features']] if metrics_70_30.get('selected_features') is not None else np.stack(features_data['outer_product'].values)
                plot_predictions(
                    features_data['formation_energy_per_atom'].values,
                    model_70_30.predict(X_full),
                    'formation', '70_30', weights_dir
                )
            success_count += 1
        except Exception as e:
            logger.error(f"Error training formation regressor: {e}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Successfully trained {success_count}/{total_models} models")
    print(f"All models saved to: {weights_dir.absolute()}\n")
    return 0 if success_count == total_models else 1


if __name__ == '__main__':
    exit(main())
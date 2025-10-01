#!/usr/bin/env python3
"""
Train models for materials property prediction
This script trains all models and saves them to the weights directory
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from lightgbm import LGBMClassifier, LGBMRegressor


def prepare_element_features(element_csv):
    """Prepare element feature vectors"""
    element_df = pd.read_csv(element_csv)
    
    features = element_df[['Atomic number', 'Group', 'Period', 'Density',
                           'Electronegativity', 'UE', 'Ionisation Energy', 'Atomic radius']]
    
    scaled_features = StandardScaler().fit_transform(features)
    squared_features = np.square(scaled_features)
    extended_features = np.hstack((scaled_features, squared_features))
    
    element_df['vector'] = list(extended_features)
    element_df = element_df.filter(items=['Symbol', 'vector'])
    element_df.rename(columns={'Symbol': 'Element', 'vector': 'Vector'}, inplace=True)
    
    return element_df


def train_ordering_classifier(features_df, split_ratio=0.1, class_weight=None, 
                              n_estimators=50, max_depth=6, learning_rate=0.1,
                              num_leaves=10, min_child_samples=10, subsample=0.5,
                              reg_alpha=0.1, reg_lambda=0.3, colsample_bytree=0.3):
    """Train magnetic ordering classifier"""
    
    # Prepare features
    outer_product_df = pd.DataFrame(features_df['outer_product'].tolist(), 
                                   columns=[f'feature_{i+1}' for i in range(518)])
    df_expanded = pd.concat([outer_product_df, features_df['ordering']], axis=1)
    
    X = df_expanded.drop('ordering', axis=1)
    y = df_expanded['ordering']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )
    
    # Train model
    if class_weight is None:
        class_weight = {0: 1, 1: 3}
    
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
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
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
        'train_report': classification_report(y_train, y_train_pred),
        'test_report': classification_report(y_test, y_test_pred)
    }
    
    return model, metrics


def train_moment_regressor(features_df, split_ratio=0.1,
                           n_estimators=200, learning_rate=0.08, max_depth=8,
                           num_leaves=18, subsample=0.5, reg_alpha=1.0,
                           reg_lambda=1.0, colsample_bytree=0.25, min_child_samples=15):
    """Train magnetic moment regressor"""
    
    # Prepare features
    outer_product_df = pd.DataFrame(features_df['outer_product'].tolist(),
                                   columns=[f'feature_{i+1}' for i in range(518)])
    df_expanded = pd.concat([outer_product_df, features_df['total_magnetization_normalized_atoms']], 
                           axis=1)
    
    X = df_expanded.drop('total_magnetization_normalized_atoms', axis=1)
    y = df_expanded['total_magnetization_normalized_atoms']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )
    
    # Train model
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
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        'train_rmse': rmse(y_train, y_train_pred),
        'test_rmse': rmse(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_corr': pearsonr(y_train, y_train_pred)[0],
        'test_corr': pearsonr(y_test, y_test_pred)[0]
    }
    
    return model, metrics


def train_formation_regressor(features_df, split_ratio=0.1,
                              n_estimators=100, learning_rate=0.05, max_depth=12,
                              num_leaves=30, subsample=0.75, reg_alpha=1.0,
                              reg_lambda=1.0, colsample_bytree=0.6, min_child_samples=60):
    """Train formation energy regressor"""
    
    # Prepare features
    outer_product_df = pd.DataFrame(features_df['outer_product'].tolist(),
                                   columns=[f'feature_{i+1}' for i in range(518)])
    df_expanded = pd.concat([outer_product_df, features_df['formation_energy_per_atom']], 
                           axis=1)
    
    X = df_expanded.drop('formation_energy_per_atom', axis=1)
    y = df_expanded['formation_energy_per_atom']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )
    
    # Train model
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
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        'train_rmse': rmse(y_train, y_train_pred),
        'test_rmse': rmse(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_corr': pearsonr(y_train, y_train_pred)[0],
        'test_corr': pearsonr(y_test, y_test_pred)[0]
    }
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train materials property prediction models'
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
    
    # Create weights directory
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("MATERIALS PROPERTY PREDICTION - MODEL TRAINING")
    print(f"{'='*80}\n")
    
    # Prepare element features
    print("Preparing element features...")
    element_df = prepare_element_features(args.elements)
    print(f"Loaded {len(element_df)} elements\n")
    
    # Save element features
    with open(weights_dir / 'element_features.pkl', 'wb') as f:
        pickle.dump(element_df, f)
    print(f"Saved element features to {weights_dir / 'element_features.pkl'}\n")
    
    # Load compound features
    print(f"Loading compound features from {args.features}...")
    with open(args.features, 'rb') as f:
        features_data = pickle.load(f)
    
    print(f"Loaded features for {len(features_data)} compounds\n")
    
    tasks = args.tasks if 'all' not in args.tasks else ['ordering', 'moment', 'formation']
    
    # Train models
    if 'ordering' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC ORDERING CLASSIFIER")
        print(f"{'='*80}\n")
        
        # 90:10 split
        print("Training with 90:10 train/test split...")
        model_90_10, metrics_90_10 = train_ordering_classifier(
            features_data, split_ratio=0.1
        )
        
        print(f"Train Accuracy: {metrics_90_10['train_accuracy']:.4f}")
        print(f"Test Accuracy: {metrics_90_10['test_accuracy']:.4f}")
        print(f"Train AUC: {metrics_90_10['train_auc']:.4f}")
        print(f"Test AUC: {metrics_90_10['test_auc']:.4f}\n")
        
        # Save model
        with open(weights_dir / 'magnetic_ordering_model_90_10.pkl', 'wb') as f:
            pickle.dump(model_90_10, f)
        print(f"Saved to {weights_dir / 'magnetic_ordering_model_90_10.pkl'}\n")
        
        # 70:30 split
        print("Training with 70:30 train/test split...")
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
        with open(weights_dir / 'magnetic_ordering_model_70_30.pkl', 'wb') as f:
            pickle.dump(model_70_30, f)
        print(f"Saved to {weights_dir / 'magnetic_ordering_model_70_30.pkl'}\n")
    
    if 'moment' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING MAGNETIC MOMENT REGRESSOR")
        print(f"{'='*80}\n")
        
        # 90:10 split
        print("Training with 90:10 train/test split...")
        model_90_10, metrics_90_10 = train_moment_regressor(
            features_data, split_ratio=0.1
        )
        
        print(f"Train RMSE: {metrics_90_10['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
        print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
        print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
        print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}\n")
        
        # Save model
        with open(weights_dir / 'magnetic_moment_model_90_10.pkl', 'wb') as f:
            pickle.dump(model_90_10, f)
        print(f"Saved to {weights_dir / 'magnetic_moment_model_90_10.pkl'}\n")
        
        # 70:30 split
        print("Training with 70:30 train/test split...")
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
        with open(weights_dir / 'magnetic_moment_model_70_30.pkl', 'wb') as f:
            pickle.dump(model_70_30, f)
        print(f"Saved to {weights_dir / 'magnetic_moment_model_70_30.pkl'}\n")
    
    if 'formation' in tasks:
        print(f"\n{'='*80}")
        print("TRAINING FORMATION ENERGY REGRESSOR")
        print(f"{'='*80}\n")
        
        # 90:10 split
        print("Training with 90:10 train/test split...")
        model_90_10, metrics_90_10 = train_formation_regressor(
            features_data, split_ratio=0.1
        )
        
        print(f"Train RMSE: {metrics_90_10['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics_90_10['test_rmse']:.4f}")
        print(f"Train R²: {metrics_90_10['train_r2']:.4f}")
        print(f"Test R²: {metrics_90_10['test_r2']:.4f}")
        print(f"Test Correlation: {metrics_90_10['test_corr']:.4f}\n")
        
        # Save model
        with open(weights_dir / 'formation_energy_model_90_10.pkl', 'wb') as f:
            pickle.dump(model_90_10, f)
        print(f"Saved to {weights_dir / 'formation_energy_model_90_10.pkl'}\n")
        
        # 70:30 split
        print("Training with 70:30 train/test split...")
        model_70_30, metrics_70_30 = train_formation_regressor(
            features_data, split_ratio=0.3
        )
        
        print(f"Train RMSE: {metrics_70_30['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics_70_30['test_rmse']:.4f}")
        print(f"Train R²: {metrics_70_30['train_r2']:.4f}")
        print(f"Test R²: {metrics_70_30['test_r2']:.4f}")
        print(f"Test Correlation: {metrics_70_30['test_corr']:.4f}\n")
        
        # Save model
        with open(weights_dir / 'formation_energy_model_70_30.pkl', 'wb') as f:
            pickle.dump(model_70_30, f)
        print(f"Saved to {weights_dir / 'formation_energy_model_70_30.pkl'}\n")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"All models saved to: {weights_dir.absolute()}\n")


if __name__ == '__main__':
    main()
    
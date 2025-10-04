#!/usr/bin/env python3
"""
Materials Property Predictor CLI
Predicts magnetic ordering, magnetic moment, and formation energy for materials

This module provides a command-line interface for making predictions using
pre-trained LightGBM models. It supports two modes:
- Default: Expects feature vectors of dimension 518, computed using pipeline.py.
- Extended (-x): Expects feature vectors of dimension 1369, computed with -x option.

Feature Vector Composition (Default, 518 dimensions):
    - 256: Flattened mean of outer product matrices (16x16)
    - 256: Flattened std of outer product matrices (16x16)
    - 2: Electronegativity difference statistics (mean, std)
    - 2: Squared electronegativity difference statistics (mean, std)
    - 1: Coordination number (fixed at 12.0)
    - 1: Placeholder for space group (default)

Feature Vector Composition (Extended, 1369 dimensions):
    - 1352: Flattened mean and std of outer product matrices (26x26x2)
    - 4: Distance statistics (mean, std, min, max)
    - 4: Electronegativity difference statistics (mean, std, min, max)
    - 4: Coordination numbers (padded/truncated to 4 elements)
    - 4: Lattice parameters (a, b, c, volume)
    - 1: Space group number

Models:
    - Magnetic Ordering: Binary classification (FM vs FiM)
    - Magnetic Moment: Regression (μB/atom)
    - Formation Energy: Regression (eV/atom)

Author: Materials Prediction Pipeline
Version: 2.1
"""

import argparse
import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when models cannot be loaded properly"""
    pass


class FeatureValidationError(Exception):
    """Raised when feature dimensions or format are invalid"""
    pass


class MaterialsPredictor:
    """
    Materials property predictor using pre-trained LightGBM models.
    
    This class loads trained models and provides methods to predict:
    1. Magnetic ordering (FM or FiM)
    2. Magnetic moment per atom
    3. Formation energy per atom
    
    Attributes:
        weights_dir (Path): Directory containing model weights
        models (Dict): Dictionary of loaded models
        element_df (Optional[pd.DataFrame]): Element feature database
        expected_features (int): Expected number of input features (518 or 1369)
        extended (bool): Whether to use extended feature mode
    """
    
    DEFAULT_FEATURES = 518
    EXTENDED_FEATURES = 1369
    VALID_SPLITS = ['90_10', '70_30']
    VALID_TASKS = ['all', 'ordering', 'moment', 'formation']
    
    def __init__(self, weights_dir: str = 'weights', extended: bool = False):
        """
        Initialize the predictor by loading all models.
        
        Args:
            weights_dir: Directory containing model weights
            extended: Use extended feature mode (1369 dimensions)
            
        Raises:
            ModelLoadError: If weights directory doesn't exist or no models found
        """
        self.weights_dir = Path(weights_dir)
        self.extended = extended
        self.expected_features = self.EXTENDED_FEATURES if extended else self.DEFAULT_FEATURES
        self.models: Dict[str, Any] = {}
        self.element_df: Optional[pd.DataFrame] = None
        self.load_all_models()
        
    def load_all_models(self) -> None:
        """
        Load all trained models and element data.
        
        Raises:
            ModelLoadError: If critical files are missing
        """
        if not self.weights_dir.exists():
            raise ModelLoadError(
                f"Weights directory '{self.weights_dir}' not found. "
                "Please run trainer.py first to generate model weights."
            )
        
        # Load element feature data (optional)
        element_path = self.weights_dir / 'element_features.pkl'
        if element_path.exists():
            try:
                with open(element_path, 'rb') as f:
                    self.element_df = pickle.load(f)
                logger.info(f"Loaded element features: {len(self.element_df)} elements")
            except Exception as e:
                logger.warning(f"Could not load element features: {e}")
        else:
            logger.warning("Element features not found (optional)")
        
        # Load models
        model_files = {
            'ordering_90_10': 'magnetic_ordering_model_90_10.pkl',
            'ordering_70_30': 'magnetic_ordering_model_70_30.pkl',
            'moment_90_10': 'magnetic_moment_model_90_10.pkl',
            'moment_70_30': 'magnetic_moment_model_70_30.pkl',
            'formation_90_10': 'formation_energy_model_90_10.pkl',
            'formation_70_30': 'formation_energy_model_70_30.pkl',
        }
        
        loaded_count = 0
        for name, filename in model_files.items():
            model_path = self.weights_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    logger.info(f"Loaded model: {name}")
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        
        if loaded_count == 0:
            raise ModelLoadError(
                "No models found. Please train models first using trainer.py"
            )
        
        logger.info(f"Successfully loaded {loaded_count}/{len(model_files)} models")
    
    def _validate_features(self, features: np.ndarray) -> None:
        """
        Validate feature array dimensions and format.
        
        Args:
            features: Feature array to validate
            
        Raises:
            FeatureValidationError: If features are invalid
        """
        if not isinstance(features, np.ndarray):
            raise FeatureValidationError(
                f"Features must be numpy array, got {type(features)}"
            )
        
        if features.ndim != 2:
            raise FeatureValidationError(
                f"Features must be 2D array (samples, features), got shape {features.shape}"
            )
        
        if features.shape[1] != self.expected_features:
            raise FeatureValidationError(
                f"Expected {self.expected_features} features, got {features.shape[1]}. "
                "Ensure features are computed using the same pipeline configuration "
                f"({'extended' if self.extended else 'default'} mode)."
            )
        
        if np.isnan(features).any():
            raise FeatureValidationError("Features contain NaN values")
        
        if np.isinf(features).any():
            raise FeatureValidationError("Features contain infinite values")
    
    def predict_ordering(
        self, 
        features: np.ndarray, 
        split: str = '90_10'
    ) -> Optional[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict magnetic ordering (FM or FiM).
        
        Args:
            features: Feature array of shape (n_samples, expected_features)
            split: Model split to use ('90_10' or '70_30')
            
        Returns:
            Dictionary containing:
                - ordering: Predicted class ('FM' or 'FiM')
                - confidence: Prediction confidence [0, 1]
                - probabilities: Dict with probabilities for each class
            Returns None if prediction fails
            
        Raises:
            FeatureValidationError: If features are invalid
        """
        model_key = f'ordering_{split}'
        if model_key not in self.models:
            logger.error(f"Model {model_key} not found")
            return None
        
        try:
            self._validate_features(features)
            model = self.models[model_key]
            
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # 0: FM, 1: FiM
            label = 'FM' if prediction[0] == 0 else 'FiM'
            confidence = float(probabilities[0][prediction[0]])
            
            return {
                'ordering': label,
                'confidence': confidence,
                'probabilities': {
                    'FM': float(probabilities[0][0]), 
                    'FiM': float(probabilities[0][1])
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_moment(
        self, 
        features: np.ndarray, 
        split: str = '90_10'
    ) -> Optional[Dict[str, Union[float, str]]]:
        """
        Predict magnetic moment per atom.
        
        Args:
            features: Feature array of shape (n_samples, expected_features)
            split: Model split to use ('90_10' or '70_30')
            
        Returns:
            Dictionary containing:
                - magnetic_moment: Predicted moment value
                - unit: 'μB/atom'
            Returns None if prediction fails
            
        Raises:
            FeatureValidationError: If features are invalid
        """
        model_key = f'moment_{split}'
        if model_key not in self.models:
            logger.error(f"Model {model_key} not found")
            return None
        
        try:
            self._validate_features(features)
            model = self.models[model_key]
            prediction = model.predict(features)
            
            return {
                'magnetic_moment': float(prediction[0]),
                'unit': 'μB/atom'
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_formation_energy(
        self, 
        features: np.ndarray, 
        split: str = '90_10'
    ) -> Optional[Dict[str, Union[float, str]]]:
        """
        Predict formation energy per atom.
        
        Args:
            features: Feature array of shape (n_samples, expected_features)
            split: Model split to use ('90_10' or '70_30')
            
        Returns:
            Dictionary containing:
                - formation_energy: Predicted energy value
                - unit: 'eV/atom'
            Returns None if prediction fails
            
        Raises:
            FeatureValidationError: If features are invalid
        """
        model_key = f'formation_{split}'
        if model_key not in self.models:
            logger.error(f"Model {model_key} not found")
            return None
        
        try:
            self._validate_features(features)
            model = self.models[model_key]
            prediction = model.predict(features)
            
            return {
                'formation_energy': float(prediction[0]),
                'unit': 'eV/atom'
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def predict_from_features_file(
        self, 
        features_file: str, 
        task: str = 'all', 
        split: str = '90_10'
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Make predictions from a pre-computed features file.
        
        Args:
            features_file: Path to features file (.pkl or .csv)
            task: Prediction task ('all', 'ordering', 'moment', 'formation')
            split: Model split to use ('90_10' or '70_30')
            
        Returns:
            List of prediction dictionaries, one per material
            Returns None if loading or prediction fails
        """
        # Validate inputs
        if task not in self.VALID_TASKS:
            logger.error(f"Invalid task: {task}. Must be one of {self.VALID_TASKS}")
            return None
        
        if split not in self.VALID_SPLITS:
            logger.error(f"Invalid split: {split}. Must be one of {self.VALID_SPLITS}")
            return None
        
        # Load features
        try:
            if features_file.endswith('.pkl'):
                with open(features_file, 'rb') as f:
                    data = pickle.load(f)
            elif features_file.endswith('.csv'):
                data = pd.read_csv(features_file)
            else:
                logger.error("Unsupported file format. Use .pkl or .csv")
                return None
        except Exception as e:
            logger.error(f"Error loading features file: {e}")
            return None
        
        results = []
        
        # Handle pickle format (DataFrame with 'features' or 'outer_product' column)
        if isinstance(data, pd.DataFrame) and ('features' in data.columns or 'outer_product' in data.columns):
            feature_col = 'outer_product' if self.extended else 'features'
            if feature_col not in data.columns:
                logger.error(f"Expected '{feature_col}' column in features file for {'extended' if self.extended else 'default'} mode")
                return None
            logger.info(f"Processing {len(data)} materials from pickle format")
            
            for idx, row in data.iterrows():
                material_id = row.get('material_id', f'material_{idx}')
                
                try:
                    feature_vector = np.array(row[feature_col]).reshape(1, -1)
                    result = self._predict_single(material_id, feature_vector, task, split)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting {material_id}: {e}")
                    results.append({'material_id': material_id, 'error': str(e)})
        
        # Handle CSV format (flattened features as columns)
        elif isinstance(data, pd.DataFrame):
            logger.info(f"Processing {len(data)} materials from CSV format")
            
            feature_cols = [col for col in data.columns if col != 'material_id']
            
            if len(feature_cols) != self.expected_features:
                logger.error(
                    f"Expected {self.expected_features} feature columns, "
                    f"found {len(feature_cols)}"
                )
                return None
            
            for idx, row in data.iterrows():
                material_id = row.get('material_id', f'material_{idx}')
                
                try:
                    feature_vector = row[feature_cols].values.reshape(1, -1)
                    result = self._predict_single(material_id, feature_vector, task, split)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting {material_id}: {e}")
                    results.append({'material_id': material_id, 'error': str(e)})
        else:
            logger.error(f"Unsupported data format: {type(data)}")
            return None
        
        return results
    
    def _predict_single(
        self, 
        material_id: str, 
        features: np.ndarray, 
        task: str, 
        split: str
    ) -> Dict[str, Any]:
        """
        Make predictions for a single material.
        
        Args:
            material_id: Material identifier
            features: Feature vector
            task: Prediction task
            split: Model split
            
        Returns:
            Dictionary with material_id and predictions
        """
        result = {'material_id': material_id}
        
        if task in ['all', 'ordering']:
            result['ordering'] = self.predict_ordering(features, split)
        
        if task in ['all', 'moment']:
            result['moment'] = self.predict_moment(features, split)
        
        if task in ['all', 'formation']:
            result['formation'] = self.predict_formation_energy(features, split)
        
        return result


def save_results(results: List[Dict], output_file: str) -> None:
    """
    Save prediction results to file.
    
    Args:
        results: List of prediction dictionaries
        output_file: Output file path (.json or .csv)
    """
    try:
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        elif output_file.endswith('.csv'):
            # Flatten results for CSV
            flat_results = []
            for result in results:
                flat = {'material_id': result['material_id']}
                
                if 'error' in result:
                    flat['error'] = result['error']
                
                if 'ordering' in result and result['ordering']:
                    flat['ordering'] = result['ordering']['ordering']
                    flat['ordering_confidence'] = result['ordering']['confidence']
                    flat['prob_FM'] = result['ordering']['probabilities']['FM']
                    flat['prob_FiM'] = result['ordering']['probabilities']['FiM']
                
                if 'moment' in result and result['moment']:
                    flat['magnetic_moment'] = result['moment']['magnetic_moment']
                
                if 'formation' in result and result['formation']:
                    flat['formation_energy'] = result['formation']['formation_energy']
                
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        else:
            logger.error("Unsupported output format. Use .json or .csv")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def display_results(results: List[Dict], verbose: bool = False, max_display: int = 10) -> None:
    """
    Display prediction results to console.
    
    Args:
        results: List of prediction dictionaries
        verbose: Show detailed probabilities
        max_display: Maximum number of results to display
    """
    print(f"\n{'='*80}")
    print(f"PREDICTION RESULTS ({len(results)} materials)")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if 'error' not in r)
    if success_count < len(results):
        print(f"⚠ {len(results) - success_count} predictions failed\n")
    
    for i, result in enumerate(results[:max_display]):
        print(f"Material: {result['material_id']}")
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        
        if 'ordering' in result and result['ordering']:
            ord_result = result['ordering']
            print(f"  Magnetic Ordering: {ord_result['ordering']} "
                  f"(confidence: {ord_result['confidence']:.3f})")
            if verbose:
                print(f"    Probabilities: FM={ord_result['probabilities']['FM']:.3f}, "
                      f"FiM={ord_result['probabilities']['FiM']:.3f}")
        
        if 'moment' in result and result['moment']:
            mom_result = result['moment']
            print(f"  Magnetic Moment: {mom_result['magnetic_moment']:.4f} {mom_result['unit']}")
        
        if 'formation' in result and result['formation']:
            form_result = result['formation']
            print(f"  Formation Energy: {form_result['formation_energy']:.4f} {form_result['unit']}")
        
        print()
    
    if len(results) > max_display:
        print(f"... and {len(results) - max_display} more materials\n")


def main():
    """Main entry point for the predictor CLI."""
    parser = argparse.ArgumentParser(
        description='Predict magnetic and energetic properties of materials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all properties using 90:10 split model (default, 518 features)
  python predictor.py --features data/features.pkl --task all --split 90_10
  
  # Predict all properties using extended features (1369 features)
  python predictor.py --features data/training_data_x.pkl --task all --split 90_10 -x
  
  # Predict only magnetic ordering
  python predictor.py --features data/features.csv --task ordering
  
  # Predict formation energy using 70:30 split model
  python predictor.py --features data/features.pkl --task formation --split 70_30
  
  # Save output to file
  python predictor.py --features data/features.pkl --output predictions.json
  
  # Verbose output with full probabilities
  python predictor.py --features data/features.pkl --verbose

Notes:
  - Default mode expects 518-dimensional vectors computed by pipeline.py
  - Extended mode (-x) expects 1369-dimensional vectors computed by pipeline.py -x
  - Models must be trained with the same mode (default or extended) as features
  - Models are trained on FM and FiM magnetic materials only
  - 90:10 split typically provides better generalization
  - 70:30 split may be more conservative (higher regularization)
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to features file (.pkl or .csv)')
    parser.add_argument('--task', choices=MaterialsPredictor.VALID_TASKS,
                       default='all',
                       help='Prediction task (default: all)')
    parser.add_argument('--split', choices=MaterialsPredictor.VALID_SPLITS,
                       default='90_10',
                       help='Train/test split model to use (default: 90_10)')
    parser.add_argument('--weights-dir', default='weights',
                       help='Directory containing model weights (default: weights)')
    parser.add_argument('--extended', '-x', action='store_true',
                       help='Use extended feature mode (1369 dimensions)')
    parser.add_argument('--output', '-o',
                       help='Output file for predictions (.json or .csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with full probabilities')
    parser.add_argument('--max-display', type=int, default=10,
                       help='Maximum number of results to display (default: 10)')
    
    args = parser.parse_args()
    
    # Check if features file exists
    if not os.path.exists(args.features):
        logger.error(f"Features file '{args.features}' not found")
        sys.exit(1)
    
    # Initialize predictor
    print(f"\nInitializing Materials Predictor...")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Model split: {args.split}")
    print(f"Task: {args.task}")
    print(f"Feature mode: {'extended (1369 features)' if args.extended else 'default (518 features)'}\n")
    
    try:
        predictor = MaterialsPredictor(weights_dir=args.weights_dir, extended=args.extended)
    except ModelLoadError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Make predictions
    logger.info(f"Making predictions from {args.features}...")
    results = predictor.predict_from_features_file(
        args.features,
        task=args.task,
        split=args.split
    )
    
    if results is None:
        logger.error("Failed to make predictions")
        sys.exit(1)
    
    # Display results
    display_results(results, verbose=args.verbose, max_display=args.max_display)
    
    # Save output if requested
    if args.output:
        save_results(results, args.output)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
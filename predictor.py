#!/usr/bin/env python3
"""
Materials Property Predictor CLI
Predicts magnetic ordering, magnetic moment, and formation energy for materials
"""

import argparse
import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class MaterialsPredictor:
    def __init__(self, weights_dir='weights'):
        self.weights_dir = Path(weights_dir)
        self.models = {}
        self.element_df = None
        self.load_all_models()
        
    def load_all_models(self):
        """Load all trained models and element data"""
        if not self.weights_dir.exists():
            print(f"Error: Weights directory '{self.weights_dir}' not found.")
            print("Please run train_models.py first to generate model weights.")
            sys.exit(1)
        
        # Load element feature data
        element_path = self.weights_dir / 'element_features.pkl'
        if element_path.exists():
            with open(element_path, 'rb') as f:
                self.element_df = pickle.load(f)
            print(f"Loaded element features: {len(self.element_df)} elements")
        else:
            print("Warning: Element features not found")
        
        # Load models
        model_files = {
            'ordering_90_10': 'magnetic_ordering_model_90_10.pkl',
            'ordering_70_30': 'magnetic_ordering_model_70_30.pkl',
            'moment_90_10': 'magnetic_moment_model_90_10.pkl',
            'moment_70_30': 'magnetic_moment_model_70_30.pkl',
            'formation_90_10': 'formation_energy_model_90_10.pkl',
            'formation_70_30': 'formation_energy_model_70_30.pkl',
        }
        
        for name, filename in model_files.items():
            model_path = self.weights_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded model: {name}")
        
        if not self.models:
            print("Error: No models found. Please train models first.")
            sys.exit(1)
    
    def predict_ordering(self, features, split='90_10'):
        """Predict magnetic ordering (FM or FiM)"""
        model_key = f'ordering_{split}'
        if model_key not in self.models:
            print(f"Error: Model {model_key} not found")
            return None
        
        model = self.models[model_key]
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # 0: FM, 1: FiM
        label = 'FM' if prediction[0] == 0 else 'FiM'
        confidence = probabilities[0][prediction[0]]
        
        return {
            'ordering': label,
            'confidence': confidence,
            'probabilities': {'FM': probabilities[0][0], 'FiM': probabilities[0][1]}
        }
    
    def predict_moment(self, features, split='90_10'):
        """Predict magnetic moment per atom"""
        model_key = f'moment_{split}'
        if model_key not in self.models:
            print(f"Error: Model {model_key} not found")
            return None
        
        model = self.models[model_key]
        prediction = model.predict(features)
        
        return {
            'magnetic_moment': prediction[0],
            'unit': 'μB/atom'
        }
    
    def predict_formation_energy(self, features, split='90_10'):
        """Predict formation energy per atom"""
        model_key = f'formation_{split}'
        if model_key not in self.models:
            print(f"Error: Model {model_key} not found")
            return None
        
        model = self.models[model_key]
        prediction = model.predict(features)
        
        return {
            'formation_energy': prediction[0],
            'unit': 'eV/atom'
        }
    
    def predict_from_features_file(self, features_file, task='all', split='90_10'):
        """Make predictions from a pre-computed features file"""
        # Load features
        if features_file.endswith('.pkl'):
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
        elif features_file.endswith('.csv'):
            data = pd.read_csv(features_file)
            # Assume first column is material_id, rest are features
            if 'material_id' in data.columns:
                material_ids = data['material_id'].values
                features = data.drop('material_id', axis=1).values
            else:
                material_ids = [f"material_{i}" for i in range(len(data))]
                features = data.values
        else:
            print("Error: Unsupported file format. Use .pkl or .csv")
            return None
        
        results = []
        
        # Handle pickle format
        if isinstance(data, pd.DataFrame) and 'features' in data.columns:
            for idx, row in data.iterrows():
                material_id = row.get('material_id', f'material_{idx}')
                feature_vector = np.array(row['features']).reshape(1, -1)
                
                result = {'material_id': material_id}
                
                if task in ['all', 'ordering']:
                    result['ordering'] = self.predict_ordering(feature_vector, split)
                if task in ['all', 'moment']:
                    result['moment'] = self.predict_moment(feature_vector, split)
                if task in ['all', 'formation']:
                    result['formation'] = self.predict_formation_energy(feature_vector, split)
                
                results.append(result)
        
        # Handle CSV format
        elif isinstance(data, pd.DataFrame):
            feature_cols = [col for col in data.columns if col != 'material_id']
            for idx, row in data.iterrows():
                material_id = row.get('material_id', f'material_{idx}')
                feature_vector = row[feature_cols].values.reshape(1, -1)
                
                result = {'material_id': material_id}
                
                if task in ['all', 'ordering']:
                    result['ordering'] = self.predict_ordering(feature_vector, split)
                if task in ['all', 'moment']:
                    result['moment'] = self.predict_moment(feature_vector, split)
                if task in ['all', 'formation']:
                    result['formation'] = self.predict_formation_energy(feature_vector, split)
                
                results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict magnetic and energetic properties of materials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all properties using 90:10 split model
  python predict_materials.py --features data/features.pkl --task all --split 90_10
  
  # Predict only magnetic ordering
  python predict_materials.py --features data/features.csv --task ordering
  
  # Predict formation energy using 70:30 split model
  python predict_materials.py --features data/features.pkl --task formation --split 70_30
  
  # Save output to file
  python predict_materials.py --features data/features.pkl --output predictions.json
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to features file (.pkl or .csv)')
    parser.add_argument('--task', choices=['all', 'ordering', 'moment', 'formation'],
                       default='all',
                       help='Prediction task (default: all)')
    parser.add_argument('--split', choices=['90_10', '70_30'],
                       default='90_10',
                       help='Train/test split model to use (default: 90_10)')
    parser.add_argument('--weights-dir', default='weights',
                       help='Directory containing model weights (default: weights)')
    parser.add_argument('--output', '-o',
                       help='Output file for predictions (.json or .csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if features file exists
    if not os.path.exists(args.features):
        print(f"Error: Features file '{args.features}' not found")
        sys.exit(1)
    
    # Initialize predictor
    print(f"\nInitializing Materials Predictor...")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Model split: {args.split}")
    print(f"Task: {args.task}\n")
    
    predictor = MaterialsPredictor(weights_dir=args.weights_dir)
    
    # Make predictions
    print(f"Making predictions from {args.features}...")
    results = predictor.predict_from_features_file(
        args.features,
        task=args.task,
        split=args.split
    )
    
    if results is None:
        print("Error: Failed to make predictions")
        sys.exit(1)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"PREDICTION RESULTS ({len(results)} materials)")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:10]):  # Show first 10
        print(f"Material: {result['material_id']}")
        
        if 'ordering' in result and result['ordering']:
            ord_result = result['ordering']
            print(f"  Magnetic Ordering: {ord_result['ordering']} "
                  f"(confidence: {ord_result['confidence']:.3f})")
            if args.verbose:
                print(f"    Probabilities: FM={ord_result['probabilities']['FM']:.3f}, "
                      f"FiM={ord_result['probabilities']['FiM']:.3f}")
        
        if 'moment' in result and result['moment']:
            mom_result = result['moment']
            print(f"  Magnetic Moment: {mom_result['magnetic_moment']:.4f} {mom_result['unit']}")
        
        if 'formation' in result and result['formation']:
            form_result = result['formation']
            print(f"  Formation Energy: {form_result['formation_energy']:.4f} {form_result['unit']}")
        
        print()
    
    if len(results) > 10:
        print(f"... and {len(results) - 10} more materials\n")
    
    # Save output if requested
    if args.output:
        import json
        
        if args.output.endswith('.json'):
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        elif args.output.endswith('.csv'):
            # Flatten results for CSV
            flat_results = []
            for result in results:
                flat = {'material_id': result['material_id']}
                
                if 'ordering' in result and result['ordering']:
                    flat['ordering'] = result['ordering']['ordering']
                    flat['ordering_confidence'] = result['ordering']['confidence']
                
                if 'moment' in result and result['moment']:
                    flat['magnetic_moment'] = result['moment']['magnetic_moment']
                
                if 'formation' in result and result['formation']:
                    flat['formation_energy'] = result['formation']['formation_energy']
                
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        
        else:
            print("Warning: Unsupported output format. Use .json or .csv")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
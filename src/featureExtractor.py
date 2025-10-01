#!/usr/bin/env python3
"""
Prepare training data from the original notebook format
This script converts compound data and computed features into the format needed for training
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def prepare_training_data(compound_file, features_file, output_file):
    """
    Prepare training data by merging compound properties with features
    
    Args:
        compound_file: Path to Compound_property_data.xlsx
        features_file: Path to computed features (from notebook or compute_features.py)
        output_file: Path to save merged training data
    """
    
    print("Loading compound data...")
    compound_df = pd.read_excel(compound_file)
    print(f"Loaded {len(compound_df)} compounds")
    
    # Filter to FM and FiM only
    categories_to_keep = ['FM', 'FiM']
    compound_df = compound_df[compound_df['ordering'].isin(categories_to_keep)]
    print(f"Filtered to {len(compound_df)} FM/FiM compounds")
    
    # Encode ordering labels
    encoder = LabelEncoder()
    compound_df['ordering'] = encoder.fit_transform(compound_df['ordering'])
    print(f"Class mappings: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    
    print("\nLoading computed features...")
    # Try to load features in different formats
    if features_file.endswith('.pkl'):
        with open(features_file, 'rb') as f:
            features_data = pickle.load(f)
        
        if isinstance(features_data, list):
            df_features = pd.DataFrame(features_data)
        elif isinstance(features_data, pd.DataFrame):
            df_features = features_data
        else:
            raise ValueError(f"Unexpected features format: {type(features_data)}")
    
    elif features_file.endswith('.csv'):
        df_features = pd.read_csv(features_file)
    
    else:
        raise ValueError("Features file must be .pkl or .csv")
    
    print(f"Loaded features for {len(df_features)} materials")
    
    # Merge features with compound properties
    print("\nMerging data...")
    
    # For ordering classification
    result_ordering = df_features.merge(
        compound_df[['material_id', 'ordering']], 
        on='material_id', 
        how='left'
    )
    result_ordering = result_ordering.rename(columns={'features': 'outer_product'})
    result_ordering = result_ordering.dropna(subset=['ordering'])
    
    # For magnetic moment regression
    result_moment = df_features.merge(
        compound_df[['material_id', 'total_magnetization_normalized_atoms']], 
        on='material_id', 
        how='left'
    )
    result_moment = result_moment.rename(columns={'features': 'outer_product'})
    result_moment = result_moment.dropna(subset=['total_magnetization_normalized_atoms'])
    
    # For formation energy regression
    result_formation = df_features.merge(
        compound_df[['material_id', 'formation_energy_per_atom']], 
        on='material_id', 
        how='left'
    )
    result_formation = result_formation.rename(columns={'features': 'outer_product'})
    result_formation = result_formation.dropna(subset=['formation_energy_per_atom'])
    
    # Combine all properties
    result = df_features.copy()
    result = result.rename(columns={'features': 'outer_product'})
    
    # Add ordering
    ordering_map = dict(zip(result_ordering['material_id'], result_ordering['ordering']))
    result['ordering'] = result['material_id'].map(ordering_map)
    
    # Add moment
    moment_map = dict(zip(result_moment['material_id'], 
                         result_moment['total_magnetization_normalized_atoms']))
    result['total_magnetization_normalized_atoms'] = result['material_id'].map(moment_map)
    
    # Add formation energy
    formation_map = dict(zip(result_formation['material_id'], 
                            result_formation['formation_energy_per_atom']))
    result['formation_energy_per_atom'] = result['material_id'].map(formation_map)
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total materials: {len(result)}")
    print(f"  With ordering labels: {result['ordering'].notna().sum()}")
    print(f"  With moment values: {result['total_magnetization_normalized_atoms'].notna().sum()}")
    print(f"  With formation energy: {result['formation_energy_per_atom'].notna().sum()}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print("Done!")
    
    return result


def verify_training_data(training_file):
    """Verify the training data format"""
    print(f"\nVerifying {training_file}...")
    
    with open(training_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nData type: {type(data)}")
    
    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print(f"\nColumns: {list(data.columns)}")
        print(f"\nFirst few rows:")
        print(data.head())
        
        if 'outer_product' in data.columns:
            sample_features = data['outer_product'].iloc[0]
            if isinstance(sample_features, (list, np.ndarray)):
                print(f"\nFeature vector shape: {np.array(sample_features).shape}")
                print(f"First 10 features: {np.array(sample_features)[:10]}")
        
        if 'ordering' in data.columns:
            print(f"\nOrdering value counts:")
            print(data['ordering'].value_counts())
        
        if 'total_magnetization_normalized_atoms' in data.columns:
            print(f"\nMagnetic moment statistics:")
            print(data['total_magnetization_normalized_atoms'].describe())
        
        if 'formation_energy_per_atom' in data.columns:
            print(f"\nFormation energy statistics:")
            print(data['formation_energy_per_atom'].describe())
    
    print("\nVerification complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data from compound properties and features'
    )
    
    parser.add_argument('--compounds', required=True,
                       help='Path to Compound_property_data.xlsx')
    parser.add_argument('--features', required=True,
                       help='Path to computed features (.pkl or .csv)')
    parser.add_argument('--output', required=True,
                       help='Output path for training data (.pkl)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the output data format')
    
    args = parser.parse_args()
    
    # Check input files exist
    for path, name in [(args.compounds, 'compounds'), (args.features, 'features')]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            return 1
    
    # Prepare data
    try:
        result = prepare_training_data(args.compounds, args.features, args.output)
        
        if args.verify:
            verify_training_data(args.output)
        
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
    
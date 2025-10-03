#!/usr/bin/env python3
"""
Compute outer product features for materials
This script computes features from structure files without requiring API access
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


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


def calculate_outer_products_from_composition(composition, element_df, 
                                              distance=3.0, num_neighbors=12):
    """
    Calculate outer product features from composition string
    This is a simplified version that doesn't require crystal structure
    
    Args:
        composition: str, e.g., "Fe2O3" or dict, e.g., {"Fe": 2, "O": 3}
        element_df: DataFrame with element vectors
        distance: float, assumed average distance between atoms
        num_neighbors: int, assumed number of neighbors
    """
    
    # Parse composition
    if isinstance(composition, str):
        # Simple parser for compositions like "Fe2O3"
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, composition)
        comp_dict = {elem: int(count) if count else 1 for elem, count in matches}
    else:
        comp_dict = composition
    
    # Get element vectors
    elements = list(comp_dict.keys())
    vectors = {}
    for elem in elements:
        vec_row = element_df[element_df['Element'] == elem]
        if len(vec_row) == 0:
            raise ValueError(f"Element {elem} not found in element database")
        vectors[elem] = vec_row['Vector'].values[0]
    
    # Calculate outer products with simplified geometry
    outer_product_matrices = []
    distances = []
    en_differences = []
    en_sq = []
    
    # Create pairs of elements based on composition
    element_list = []
    for elem, count in comp_dict.items():
        element_list.extend([elem] * count)
    
    # Calculate features for all pairs
    for i, elem_i in enumerate(element_list):
        for j, elem_j in enumerate(element_list):
            if i != j:
                vec_i = vectors[elem_i]
                vec_j = vectors[elem_j]
                
                # Use assumed distance
                dist = distance
                
                outer_product = np.outer(vec_i, vec_j) / (dist * dist)
                outer_product_matrices.append(outer_product)
                distances.append(dist)
                
                en_i = vec_i[4]
                en_j = vec_j[4]
                en_diff1 = abs(en_i - en_j)
                en_diff2 = abs(en_i * en_i - en_j * en_j)
                
                en_differences.append(en_diff1)
                en_sq.append(en_diff2)
    
    if len(outer_product_matrices) == 0:
        # Single element - use self interaction
        elem = element_list[0]
        vec = vectors[elem]
        outer_product = np.outer(vec, vec) / (distance * distance)
        outer_product_matrices = [outer_product]
        distances = [distance]
        en_differences = [0.0]
        en_sq = [0.0]
    
    outer_product_matrices = np.array(outer_product_matrices)
    
    # Calculate statistics
    mean_matrix = np.mean(outer_product_matrices, axis=0)
    std_matrix = np.std(outer_product_matrices, axis=0)
    
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    
    en_diff_mean = np.mean(en_differences)
    en_diff_std = np.std(en_differences)
    en_sq_mean = np.mean(en_sq)
    en_sq_std = np.std(en_sq)
    
    # Flatten and concatenate
    flattened_mean = mean_matrix.flatten()
    flattened_std = std_matrix.flatten()
    
    feature_vector = np.concatenate([
        flattened_mean, flattened_std,
        [dist_mean, dist_std],
        [en_diff_mean, en_diff_std],
        [en_sq_mean, en_sq_std]
    ])
    
    return feature_vector


def main():
    parser = argparse.ArgumentParser(
        description='Compute features from material compositions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute features from a CSV file with compositions
  python compute_features.py --input compositions.csv --elements Elemental_property_data.csv --output features.pkl
  
  # Compute for a single composition
  python compute_features.py --composition "Fe2O3" --elements Elemental_property_data.csv
  
CSV format for --input:
  material_id,composition
  mat_001,Fe2O3
  mat_002,NiFe2O4
        """
    )
    
    parser.add_argument('--input', '-i',
                       help='Input CSV file with material_id and composition columns')
    parser.add_argument('--composition', '-c',
                       help='Single composition to compute (e.g., "Fe2O3")')
    parser.add_argument('--elements', required=True,
                       help='Path to element properties CSV file')
    parser.add_argument('--output', '-o',
                       help='Output pickle file for features')
    parser.add_argument('--distance', type=float, default=3.0,
                       help='Assumed average atomic distance (default: 3.0 Å)')
    parser.add_argument('--neighbors', type=int, default=12,
                       help='Assumed number of neighbors (default: 12)')
    
    args = parser.parse_args()
    
    if not args.input and not args.composition:
        parser.error("Either --input or --composition must be specified")
    
    # Load element features
    print(f"Loading element properties from {args.elements}...")
    element_df = prepare_element_features(args.elements)
    print(f"Loaded {len(element_df)} elements\n")
    
    results = []
    
    if args.composition:
        # Single composition
        print(f"Computing features for {args.composition}...")
        try:
            features = calculate_outer_products_from_composition(
                args.composition, element_df,
                distance=args.distance,
                num_neighbors=args.neighbors
            )
            print(f"Feature vector shape: {features.shape}")
            print(f"First 10 features: {features[:10]}")
            
            if args.output:
                result = pd.DataFrame([{
                    'material_id': args.composition,
                    'outer_product': features
                }])
                with open(args.output, 'wb') as f:
                    pickle.dump(result, f)
                print(f"\nSaved to {args.output}")
        
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    else:
        # Batch processing from CSV
        print(f"Loading compositions from {args.input}...")
        df = pd.read_csv(args.input)
        
        if 'material_id' not in df.columns or 'composition' not in df.columns:
            print("Error: Input CSV must have 'material_id' and 'composition' columns")
            return 1
        
        print(f"Processing {len(df)} materials...\n")
        
        for idx, row in df.iterrows():
            material_id = row['material_id']
            composition = row['composition']
            
            try:
                features = calculate_outer_products_from_composition(
                    composition, element_df,
                    distance=args.distance,
                    num_neighbors=args.neighbors
                )
                
                results.append({
                    'material_id': material_id,
                    'outer_product': features
                })
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df)} materials...")
            
            except Exception as e:
                print(f"Error processing {material_id} ({composition}): {e}")
                continue
        
        print(f"\nSuccessfully computed features for {len(results)} materials")
        
        if args.output:
            result_df = pd.DataFrame(results)
            with open(args.output, 'wb') as f:
                pickle.dump(result_df, f)
            print(f"Saved to {args.output}")
        
        else:
            print("\nFirst 5 results:")
            for i, result in enumerate(results[:5]):
                print(f"{result['material_id']}: {result['outer_product'][:5]}...")
    
    return 0


if __name__ == '__main__':
    exit(main())
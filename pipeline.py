#!/usr/bin/env python3
"""
Unified Materials Data Pipeline
Handles data download, feature extraction, and training data preparation
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN


# ============================================================================
# UTILITY FUNCTION FOR GEOMETRY EXTRACTION
# ============================================================================

def extract_geometry_from_cif(cif_path, max_distance=5.0):
    """
    Extract interatomic distances and coordination numbers from a CIF file.

    Args:
        cif_path (str): Path to the CIF file.
        max_distance (float): Maximum distance (in Å) for neighbor detection.

    Returns:
        tuple: (distances, avg_coordination)
            - distances (list): List of interatomic distances for all neighbor pairs.
            - avg_coordination (float): Average coordination number across sites.
    """
    try:
        # Load structure from CIF
        structure = Structure.from_file(cif_path)
        
        # Use CrystalNN for neighbor analysis
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        
        distances = []
        coordination_numbers = []
        
        # Iterate over all sites in the structure
        for i, site in enumerate(structure):
            # Get neighbors for the current site
            neighbors = cnn.get_nn_info(structure, i)
            
            # Extract distances to neighbors within max_distance
            for neighbor in neighbors:
                dist = neighbor['site'].distance(site)
                if dist <= max_distance:
                    distances.append(dist)
            
            # Store coordination number (number of neighbors)
            coordination_numbers.append(len(neighbors))
        
        # Calculate average coordination number
        avg_coordination = np.mean(coordination_numbers) if coordination_numbers else 12.0
        
        # If no distances found, use a default
        if not distances:
            distances = [3.0]  # Fallback to default distance
        
        return distances, avg_coordination
    
    except Exception as e:
        print(f"Error extracting geometry from {cif_path}: {e}")
        return [3.0], 12.0  # Fallback to defaults


# ============================================================================
# FEATURE EXTRACTION MODULE (MODIFIED)
# ============================================================================

def calculate_outer_products_from_composition(composition, element_df, cif_path=None, max_distance=5.0):
    """
    Calculate outer product features from composition string, using dynamic geometry from CIF.

    Args:
        composition: str (e.g., "Fe2O3") or dict (e.g., {"Fe": 2, "O": 3})
        element_df: DataFrame with element vectors
        cif_path: str, path to CIF file for geometry extraction (optional)
        max_distance: float, maximum distance for neighbor detection (in Å)

    Returns:
        np.ndarray: Feature vector including outer products and geometry statistics
    """
    # Parse composition
    if isinstance(composition, str):
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
    
    # Extract dynamic geometry from CIF if provided
    if cif_path and os.path.exists(cif_path):
        distances, avg_coordination = extract_geometry_from_cif(cif_path, max_distance)
    else:
        # Fallback to default values if no CIF or CIF processing fails
        distances = [3.0]
        avg_coordination = 12.0
        if cif_path:
            print(f"Warning: CIF file {cif_path} not found, using default distance=3.0 Å and neighbors={avg_coordination}")

    # Calculate outer products with dynamic geometry
    outer_product_matrices = []
    used_distances = []
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
                
                # Use average distance from CIF or cycle through distances
                dist = distances[(i + j) % len(distances)]  # Distribute distances across pairs
                outer_product = np.outer(vec_i, vec_j) / (dist * dist)
                outer_product_matrices.append(outer_product)
                used_distances.append(dist)
                
                en_i = vec_i[4]  # Electronegativity index
                en_j = vec_j[4]
                en_diff1 = abs(en_i - en_j)
                en_diff2 = abs(en_i * en_i - en_j * en_j)
                
                en_differences.append(en_diff1)
                en_sq.append(en_diff2)
    
    if len(outer_product_matrices) == 0:
        # Single element - use self interaction
        elem = element_list[0]
        vec = vectors[elem]
        dist = distances[0]  # Use first available distance
        outer_product = np.outer(vec, vec) / (dist * dist)
        outer_product_matrices = [outer_product]
        used_distances = [dist]
        en_differences = [0.0]
        en_sq = [0.0]
    
    outer_product_matrices = np.array(outer_product_matrices)
    
    # Calculate statistics
    mean_matrix = np.mean(outer_product_matrices, axis=0)
    std_matrix = np.std(outer_product_matrices, axis=0)
    
    dist_mean = np.mean(used_distances)
    dist_std = np.std(used_distances)
    
    en_diff_mean = np.mean(en_differences)
    en_diff_std = np.std(en_differences)
    en_sq_mean = np.mean(en_sq)
    en_sq_std = np.std(en_sq)
    
    # Add coordination number as a feature
    feature_vector = np.concatenate([
        mean_matrix.flatten(),
        std_matrix.flatten(),
        [dist_mean, dist_std],
        [en_diff_mean, en_diff_std],
        [en_sq_mean, en_sq_std],
        [avg_coordination]  # New feature: average coordination number
    ])
    
    return feature_vector


def compute_features(input_csv, element_df, cif_dir='cifs', max_distance=5.0):
    """Compute features from compositions in CSV file, using CIF files for geometry."""
    df = pd.read_csv(input_csv)
    
    if 'material_id' not in df.columns:
        raise ValueError("Input CSV must have 'material_id' column")
    
    if 'composition' not in df.columns:
        raise ValueError("Input CSV must have 'composition' column")
    
    # Filter out rows with missing compositions
    initial_count = len(df)
    df = df.dropna(subset=['composition'])
    if len(df) < initial_count:
        print(f"Warning: Dropped {initial_count - len(df)} materials with missing compositions")
    
    print(f"Computing features for {len(df)} materials...")
    
    results = []
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing features"):
        material_id = row['material_id']
        composition = row['composition']
        
        # Construct CIF path
        cif_path = os.path.join(cif_dir, f"{material_id}.cif")
        
        try:
            features = calculate_outer_products_from_composition(
                composition, element_df,
                cif_path=cif_path,
                max_distance=max_distance
            )
            
            results.append({
                'material_id': material_id,
                'features': features
            })
        
        except Exception as e:
            failed.append((material_id, str(e)))
    
    if failed:
        print(f"\nFailed to compute features for {len(failed)} materials")
        for mid, err in failed[:5]:
            print(f"  {mid}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print(f"Successfully computed features for {len(results)} materials")
    return pd.DataFrame(results)


# ============================================================================
# MAIN PIPELINE (MODIFIED)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified materials data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: compute features, prepare training data (assumes CIFs pre-downloaded)
  python pipeline.py -c compounds.xlsx -e elements.csv -o training.pkl --skip-download
  
  # Only compute features
  python pipeline.py -i compositions.csv -e elements.csv -f features.pkl --features-only
        """
    )
    
    # Input files
    parser.add_argument('-c', '--compounds',
                       help='Compound properties CSV file')
    parser.add_argument('-e', '--elements', required=True,
                       help='Element properties CSV file')
    parser.add_argument('-i', '--input',
                       help='Input CSV with material_id and composition columns')
    
    # Output files
    parser.add_argument('-o', '--output',
                       help='Output pickle file for training data')
    parser.add_argument('-f', '--features-out',
                       help='Output pickle file for features only')
    
    # API and directories
    parser.add_argument('-d', '--cif-dir', default='cifs',
                       help='Directory for pre-downloaded CIF files (default: cifs)')
    
    # Pipeline control
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip CIF download step (assumes CIFs are pre-downloaded)')
    parser.add_argument('--features-only', action='store_true',
                       help='Only compute features, no training data prep')
    parser.add_argument('--max-distance', type=float, default=5.0,
                       help='Maximum distance for neighbor detection in Å (default: 5.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.features_only:
        if not args.input:
            parser.error("--features-only requires -i/--input")
        if not args.features_out:
            parser.error("--features-only requires -f/--features-out")
    else:
        if not args.compounds:
            parser.error("Full pipeline requires -c/--compounds")
        if not args.output:
            parser.error("Full pipeline requires -o/--output")
    
    # Check element file exists
    if not Path(args.elements).exists():
        print(f"Error: Element file not found: {args.elements}")
        return 1
    
    # Check CIF directory exists
    if not Path(args.cif_dir).exists():
        print(f"Error: CIF directory '{args.cif_dir}' not found")
        return 1
    
    print("="*80)
    print("MATERIALS DATA PIPELINE")
    print("="*80)
    
    # Load element features
    print("\nPreparing element features...")
    element_df = prepare_element_features(args.elements)
    print(f"Loaded {len(element_df)} elements")
    
    # Step 1: Skip CIF download (assumed pre-downloaded)
    if not args.skip_download:
        print("\nWarning: --skip-download not specified, but CIF download is skipped as per assumption.")
    
    # Step 2: Compute features
    print("\n" + "="*80)
    print("STEP 2: COMPUTING FEATURES")
    print("="*80 + "\n")
    
    if args.input:
        # Use provided composition file
        features_df = compute_features(
            args.input, element_df,
            cif_dir=args.cif_dir,
            max_distance=args.max_distance
        )
    elif args.compounds:
        # Extract compositions from compounds file
        compound_df = pd.read_excel(args.compounds)
        
        # Check available columns
        print(f"Available columns in compounds file: {list(compound_df.columns)}")
        
        # Try to find a composition/formula column
        composition_col = None
        for col_name in ['composition', 'formula', 'pretty_formula', 'reduced_cell_formula']:
            if col_name in compound_df.columns:
                composition_col = col_name
                break
        
        if composition_col is None:
            # Try to extract from CIF files
            print("\nNo composition column found. Extracting from CIF files...")
            
            try:
                compositions_dict = extract_compositions_from_cifs(
                    compound_df['material_id'].values,
                    args.cif_dir
                )
                
                if not compositions_dict:
                    print("\nError: No compositions could be extracted from CIF files")
                    return 1
                
                # Add compositions to dataframe
                compound_df['composition'] = compound_df['material_id'].map(compositions_dict)
                composition_col = 'composition'
                
            except ImportError:
                print("\nError: pymatgen not installed. Install with: pip install pymatgen")
                print("Or add a 'composition' column to your compounds file manually")
                return 1
        
        print(f"Using '{composition_col}' column for compositions")
        
        # Save temporary CSV
        temp_csv = Path('temp_compositions.csv')
        temp_df = compound_df[['material_id', composition_col]].copy()
        temp_df.rename(columns={composition_col: 'composition'}, inplace=True)
        temp_df.to_csv(temp_csv, index=False)
        
        features_df = compute_features(
            temp_csv, element_df,
            cif_dir=args.cif_dir,
            max_distance=args.max_distance
        )
        
        temp_csv.unlink()  # Clean up
    
    # Save features if requested
    if args.features_out:
        print(f"\nSaving features to {args.features_out}...")
        with open(args.features_out, 'wb') as f:
            pickle.dump(features_df, f)
        print("Features saved!")
    
    # Step 3: Prepare training data (if not features-only mode)
    if not args.features_only:
        print("\n" + "="*80)
        print("STEP 3: PREPARING TRAINING DATA")
        print("="*80 + "\n")
        
        training_data = prepare_training_data(args.compounds, features_df)
        
        print(f"\nSaving training data to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(training_data, f)
        print("Training data saved!")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
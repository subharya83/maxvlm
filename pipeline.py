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


# ============================================================================
# DATA DOWNLOAD MODULE
# ============================================================================

def download_cif_files(compound_df, cif_dir, api_key):
    """Download CIF files from Materials Project"""
    try:
        from pymatgen.ext.matproj import MPRester
    except ImportError:
        print("Error: pymatgen not installed. Install with: pip install pymatgen")
        return False
    
    os.makedirs(cif_dir, exist_ok=True)
    material_ids = compound_df['material_id'].unique()
    
    # Check which files already exist
    existing_cifs = set(f.stem for f in Path(cif_dir).glob("*.cif"))
    materials_to_download = [mid for mid in material_ids if mid not in existing_cifs]
    
    if not materials_to_download:
        print(f"All {len(material_ids)} CIF files already exist in {cif_dir}")
        return True
    
    print(f"Found {len(existing_cifs)} existing CIF files")
    print(f"Downloading {len(materials_to_download)} remaining CIF files...")
    
    failed = []
    with MPRester(api_key) as m:
        for material_id in tqdm(materials_to_download, desc="Downloading CIFs"):
            try:
                structure = m.get_structure_by_material_id(material_id)
                cif_path = os.path.join(cif_dir, f"{material_id}.cif")
                structure.to(filename=cif_path)
            except Exception as e:
                print(f"\nError downloading {material_id}: {e}")
                failed.append(material_id)
    
    if failed:
        print(f"\nFailed to download {len(failed)} materials: {failed[:5]}...")
    
    print(f"Download complete. Files saved to {cif_dir}")
    return True


# ============================================================================
# FEATURE EXTRACTION MODULE
# ============================================================================

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
    
    Args:
        composition: str (e.g., "Fe2O3") or dict (e.g., {"Fe": 2, "O": 3})
        element_df: DataFrame with element vectors
        distance: float, assumed average distance between atoms
        num_neighbors: int, assumed number of neighbors
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


def compute_features(input_csv, element_df, distance=3.0, neighbors=12):
    """Compute features from compositions in CSV file"""
    df = pd.read_csv(input_csv)
    
    if 'material_id' not in df.columns or 'composition' not in df.columns:
        raise ValueError("Input CSV must have 'material_id' and 'composition' columns")
    
    print(f"Computing features for {len(df)} materials...")
    
    results = []
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing features"):
        material_id = row['material_id']
        composition = row['composition']
        
        try:
            features = calculate_outer_products_from_composition(
                composition, element_df,
                distance=distance,
                num_neighbors=neighbors
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
    
    print(f"Successfully computed features for {len(results)} materials")
    return pd.DataFrame(results)


# ============================================================================
# TRAINING DATA PREPARATION MODULE
# ============================================================================

def prepare_training_data(compound_file, features_df):
    """
    Prepare training data by merging compound properties with features
    
    Args:
        compound_file: Path to compound properties Excel file
        features_df: DataFrame with computed features
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
    
    # Merge features with compound properties
    print("\nMerging data...")
    result = features_df.copy()
    result = result.rename(columns={'features': 'outer_product'})
    
    # Add ordering
    ordering_map = dict(zip(
        compound_df['material_id'],
        compound_df['ordering']
    ))
    result['ordering'] = result['material_id'].map(ordering_map)
    
    # Add moment
    if 'total_magnetization_normalized_atoms' in compound_df.columns:
        moment_map = dict(zip(
            compound_df['material_id'],
            compound_df['total_magnetization_normalized_atoms']
        ))
        result['total_magnetization_normalized_atoms'] = result['material_id'].map(moment_map)
    
    # Add formation energy
    if 'formation_energy_per_atom' in compound_df.columns:
        formation_map = dict(zip(
            compound_df['material_id'],
            compound_df['formation_energy_per_atom']
        ))
        result['formation_energy_per_atom'] = result['material_id'].map(formation_map)
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total materials: {len(result)}")
    print(f"  With ordering labels: {result['ordering'].notna().sum()}")
    if 'total_magnetization_normalized_atoms' in result.columns:
        print(f"  With moment values: {result['total_magnetization_normalized_atoms'].notna().sum()}")
    if 'formation_energy_per_atom' in result.columns:
        print(f"  With formation energy: {result['formation_energy_per_atom'].notna().sum()}")
    
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified materials data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download CIFs, compute features, prepare training data
  python pipeline.py -c compounds.xlsx -e elements.csv -o training.pkl -k YOUR_API_KEY
  
  # Skip download (CIFs already exist)
  python pipeline.py -c compounds.xlsx -e elements.csv -o training.pkl --skip-download
  
  # Only compute features (no download, no training data prep)
  python pipeline.py -i compositions.csv -e elements.csv -f features.pkl --features-only
        """
    )
    
    # Input files
    parser.add_argument('-c', '--compounds',
                       help='Compound properties Excel file')
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
    parser.add_argument('-k', '--api-key',
                       help='Materials Project API key')
    parser.add_argument('-d', '--cif-dir', default='cifs',
                       help='Directory for CIF files (default: cifs)')
    
    # Pipeline control
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip CIF download step')
    parser.add_argument('--features-only', action='store_true',
                       help='Only compute features, no training data prep')
    parser.add_argument('--distance', type=float, default=3.0,
                       help='Assumed atomic distance in Å (default: 3.0)')
    parser.add_argument('--neighbors', type=int, default=12,
                       help='Assumed number of neighbors (default: 12)')
    
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
    
    print("="*80)
    print("MATERIALS DATA PIPELINE")
    print("="*80)
    
    # Load element features
    print("\nPreparing element features...")
    element_df = prepare_element_features(args.elements)
    print(f"Loaded {len(element_df)} elements")
    
    # Step 1: Download CIFs (if needed)
    if not args.skip_download and not args.features_only:
        if not args.api_key:
            print("\nWarning: No API key provided. Skipping CIF download.")
            print("Use -k/--api-key to enable download, or --skip-download to suppress this warning.")
        else:
            print("\n" + "="*80)
            print("STEP 1: DOWNLOADING CIF FILES")
            print("="*80)
            
            compound_df = pd.read_excel(args.compounds)
            download_cif_files(compound_df, args.cif_dir, args.api_key)
    
    # Step 2: Compute features
    print("\n" + "="*80)
    print("STEP 2: COMPUTING FEATURES")
    print("="*80 + "\n")
    
    if args.input:
        # Use provided composition file
        features_df = compute_features(
            args.input, element_df,
            distance=args.distance,
            neighbors=args.neighbors
        )
    elif args.compounds:
        # Extract compositions from compounds file
        compound_df = pd.read_excel(args.compounds)
        
        # Check if composition column exists
        if 'composition' not in compound_df.columns:
            # Try to use formula or other columns
            if 'formula' in compound_df.columns:
                compound_df['composition'] = compound_df['formula']
            elif 'pretty_formula' in compound_df.columns:
                compound_df['composition'] = compound_df['pretty_formula']
            else:
                print("Error: No composition/formula column found in compounds file")
                return 1
        
        # Save temporary CSV
        temp_csv = Path('temp_compositions.csv')
        compound_df[['material_id', 'composition']].to_csv(temp_csv, index=False)
        
        features_df = compute_features(
            temp_csv, element_df,
            distance=args.distance,
            neighbors=args.neighbors
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
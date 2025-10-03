#!/usr/bin/env python3
"""
Unified Materials Data Pipeline

Supports:
- Default: 518-dimensional feature vector (8 element features)
- With -x: Extended feature vector (>1367 dimensions, including geometry features)
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


def derive_unpaired_electrons(electron_config):
    if pd.isna(electron_config) or electron_config == '':
        return 0
    orbitals = electron_config.split()
    if not orbitals:
        return 0
    last_orbital = orbitals[-1]
    match = re.search(r'([spdf])(\d+)', last_orbital)
    if not match:
        return 0
    orbital_type = match.group(1)
    num_electrons = int(match.group(2))
    max_electrons = {'s': 2, 'p': 6, 'd': 10, 'f': 14}
    max_e = max_electrons.get(orbital_type, 0)
    return num_electrons if num_electrons <= max_e // 2 else max_e - num_electrons


def derive_group_period(atomic_number):
    period_boundaries = [0, 2, 10, 18, 36, 54, 86, 118]
    period = next(i for i, b in enumerate(period_boundaries[1:], 1) if atomic_number <= b)
    group_map = {1: 1, 2: 18, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}
    if atomic_number <= 2:
        group = atomic_number
    elif atomic_number <= 18:
        group = atomic_number - 2 if atomic_number <= 10 else atomic_number - 10 + 13
    else:
        group = ((atomic_number - 2) % 18) + 1
    return group, period


def prepare_element_features(element_csv, extended=False):
    element_df = pd.read_csv(element_csv)
    print("Deriving element properties...")
    element_df[['Group', 'Period']] = element_df['AtomicNumber'].apply(
        lambda x: pd.Series(derive_group_period(x))
    )
    element_df['UE'] = element_df['ElectronConfiguration'].apply(derive_unpaired_electrons)
    if extended:
        def parse_oxidation_states(states):
            if pd.isna(states) or states == '':
                return 0
            try:
                states = [int(s) for s in states.replace('+', '').split(',')]
                return max(states, key=abs) if states else 0
            except:
                return 0
        element_df['OxidationState'] = element_df['OxidationStates'].apply(parse_oxidation_states)
        numeric_cols = ['AtomicMass', 'Density', 'Electronegativity', 'IonizationEnergy',
                        'AtomicRadius', 'ElectronAffinity', 'MeltingPoint', 'BoilingPoint']
        feature_cols = ['AtomicNumber', 'Group', 'Period', 'AtomicMass', 'Density',
                        'Electronegativity', 'UE', 'IonizationEnergy', 'AtomicRadius',
                        'ElectronAffinity', 'MeltingPoint', 'BoilingPoint', 'OxidationState']
    else:
        numeric_cols = ['Density', 'Electronegativity', 'IonizationEnergy', 'AtomicRadius']
        feature_cols = ['AtomicNumber', 'Group', 'Period', 'Density',
                        'Electronegativity', 'UE', 'IonizationEnergy', 'AtomicRadius']
    for col in numeric_cols:
        if col in element_df.columns and element_df[col].isna().any():
            missing_count = element_df[col].isna().sum()
            median_val = element_df[col].median()
            element_df.loc[:, col] = element_df[col].fillna(median_val)
            print(f"  Filled {missing_count} missing {col} values with median")
    features = element_df[feature_cols]
    scaled_features = StandardScaler().fit_transform(features)
    squared_features = np.square(scaled_features)
    extended_features = np.hstack((scaled_features, squared_features))
    element_df['vector'] = list(extended_features)
    element_df = element_df.filter(items=['Symbol', 'vector']).rename(
        columns={'Symbol': 'Element', 'vector': 'Vector'}
    )
    return element_df


def extract_geometry_from_cif(cif_path, max_distance=5.0):
    try:
        structure = Structure.from_file(cif_path)
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        pair_distances = {}
        coordination_numbers = {}
        species = set(structure.species)
        for sp1 in species:
            coordination_numbers[sp1.symbol] = []
            for sp2 in species:
                pair_distances[(sp1.symbol, sp2.symbol)] = []
        for i, site in enumerate(structure):
            neighbors = cnn.get_nn_info(structure, i)
            site_symbol = site.species_string
            coordination_numbers[site_symbol].append(len(neighbors))
            for neighbor in neighbors:
                dist = neighbor['site'].distance(site)
                if dist <= max_distance:
                    neighbor_symbol = neighbor['site'].species_string
                    pair_key = tuple(sorted([site_symbol, neighbor_symbol]))
                    pair_distances[pair_key].append(dist)
        for pair in pair_distances:
            if not pair_distances[pair]:
                pair_distances[pair] = [3.0]
        coordination_numbers = {elem: np.mean(cns) if cns else 12.0 for elem, cns in coordination_numbers.items()}
        lattice_params = [structure.lattice.a, structure.lattice.b, structure.lattice.c, structure.volume]
        space_group = structure.get_space_group_info()[1]
        return {
            'pair_distances': pair_distances,
            'coordination_numbers': coordination_numbers,
            'lattice_params': lattice_params,
            'space_group': space_group
        }
    except Exception as e:
        print(f"Error extracting geometry from {cif_path}: {e}")
        species = Structure.from_file(cif_path).composition.elements
        pair_distances = {(sp1.symbol, sp2.symbol): [3.0] for sp1 in species for sp2 in species}
        coordination_numbers = {sp.symbol: 12.0 for sp in species}
        return {
            'pair_distances': pair_distances,
            'coordination_numbers': coordination_numbers,
            'lattice_params': [0.0, 0.0, 0.0, 0.0],
            'space_group': 1
        }


def calculate_outer_products_from_composition(composition, element_df, cif_path=None, max_distance=5.0, extended=False):
    if isinstance(composition, str):
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', composition)
        comp_dict = {elem: int(count) if count else 1 for elem, count in matches}
    else:
        comp_dict = composition
    elements = list(comp_dict.keys())
    vectors = {}
    for elem in elements:
        vec_row = element_df[element_df['Element'] == elem]
        if len(vec_row) == 0:
            raise ValueError(f"Element {elem} not found")
        vectors[elem] = vec_row['Vector'].values[0]
    en_idx = 5 if extended else 4  # Electronegativity index
    if extended and cif_path and os.path.exists(cif_path):
        geometry = extract_geometry_from_cif(cif_path, max_distance)
        pair_distances = geometry['pair_distances']
        coordination_numbers = geometry['coordination_numbers']
        lattice_params = geometry['lattice_params']
        space_group = geometry['space_group']
    else:
        pair_distances = {(e1, e2): [3.0] for e1 in elements for e2 in elements}
        coordination_numbers = {e: 12.0 for e in elements}
        lattice_params = [0.0, 0.0, 0.0, 0.0]
        space_group = 1
        if extended and cif_path:
            print(f"Warning: CIF file {cif_path} not found, using defaults")
    outer_product_matrices = []
    used_distances = []
    en_differences = []
    en_sq = []
    pair_weights = []
    element_list = [elem for elem, count in comp_dict.items() for _ in range(count)]
    for i, elem_i in enumerate(element_list):
        for j, elem_j in enumerate(element_list):
            if i != j:
                vec_i = vectors[elem_i]
                vec_j = vectors[elem_j]
                pair_key = tuple(sorted([elem_i, elem_j]))
                distances = pair_distances.get(pair_key, [3.0])
                dist = np.mean(distances)
                pair_count = len(distances)
                weight = pair_count / sum(len(d) for d in pair_distances.values()) if extended else 1.0
                outer_product = np.outer(vec_i, vec_j) / (dist * dist)
                outer_product_matrices.append(outer_product * weight)
                used_distances.append(dist)
                pair_weights.append(weight)
                en_i = vec_i[en_idx]
                en_j = vec_j[en_idx]
                en_differences.append(abs(en_i - en_j) * weight)
                en_sq.append(abs(en_i * en_i - en_j * en_j) * weight)
    if not outer_product_matrices:
        elem = element_list[0]
        vec = vectors[elem]
        pair_key = (elem, elem)
        dist = np.mean(pair_distances.get(pair_key, [3.0]))
        outer_product = np.outer(vec, vec) / (dist * dist)
        outer_product_matrices = [outer_product]
        used_distances = [dist]
        en_differences = [0.0]
        en_sq = [0.0]
        pair_weights = [1.0]
    outer_product_matrices = np.array(outer_product_matrices)
    mean_matrix = np.mean(outer_product_matrices, axis=0)
    std_matrix = np.std(outer_product_matrices, axis=0)
    feature_vector = [mean_matrix.flatten(), std_matrix.flatten()]
    if extended:
        dist_stats = [
            np.mean(used_distances),
            np.std(used_distances),
            np.min(used_distances) if used_distances else 3.0,
            np.max(used_distances) if used_distances else 3.0
        ]
        feature_vector.extend([dist_stats, [np.mean(en_differences), np.std(en_differences)],
                              [np.mean(en_sq), np.std(en_sq)], [coordination_numbers.get(elem, 12.0) for elem in elements],
                              lattice_params, [space_group]])
    else:
        feature_vector.extend([[np.mean(en_differences), np.std(en_differences)],
                              [np.mean(en_sq), np.std(en_sq)], [12.0]])
    feature_vector = np.concatenate(feature_vector)
    return feature_vector


def extract_composition_from_cif(cif_path):
    try:
        structure = Structure.from_file(cif_path)
        return structure.composition.reduced_formula
    except Exception as e:
        print(f"Error extracting composition from {cif_path}: {e}")
        return None


def extract_compositions_from_cifs(material_ids, cif_dir):
    compositions = {}
    missing = []
    print(f"\nExtracting compositions from CIFs in {cif_dir}...")
    for material_id in tqdm(material_ids, desc="Reading CIFs"):
        cif_path = Path(cif_dir) / f"{material_id}.cif"
        if cif_path.exists():
            comp = extract_composition_from_cif(cif_path)
            if comp:
                compositions[material_id] = comp
            else:
                missing.append(material_id)
        else:
            missing.append(material_id)
    if missing:
        print(f"Warning: Could not extract composition for {len(missing)} materials")
    print(f"Extracted {len(compositions)} compositions")
    return compositions


def compute_features(input_csv, element_df, cif_dir='cifs', max_distance=5.0, extended=False):
    df = pd.read_csv(input_csv)
    if 'material_id' not in df.columns or 'composition' not in df.columns:
        raise ValueError("Input CSV must have 'material_id' and 'composition' columns")
    initial_count = len(df)
    df = df.dropna(subset=['composition'])
    if len(df) < initial_count:
        print(f"Dropped {initial_count - len(df)} materials with missing compositions")
    print(f"Computing features for {len(df)} materials...")
    results = []
    failed = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing features"):
        material_id = row['material_id']
        composition = row['composition']
        cif_path = os.path.join(cif_dir, f"{material_id}.cif") if extended else None
        try:
            features = calculate_outer_products_from_composition(
                composition, element_df, cif_path=cif_path, max_distance=max_distance, extended=extended
            )
            results.append({'material_id': material_id, 'features': features})
        except Exception as e:
            failed.append((material_id, str(e)))
    if failed:
        print(f"\nFailed to compute features for {len(failed)} materials")
    print(f"Computed features for {len(results)} materials")
    return pd.DataFrame(results)


def prepare_training_data(compound_file, features_df):
    print("Loading compound data...")
    compound_df = pd.read_csv(compound_file)
    print(f"Loaded {len(compound_df)} compounds")
    categories_to_keep = ['FM', 'FiM']
    compound_df = compound_df[compound_df['ordering'].isin(categories_to_keep)]
    print(f"Filtered to {len(compound_df)} FM/FiM compounds")
    encoder = LabelEncoder()
    compound_df['ordering'] = encoder.fit_transform(compound_df['ordering'])
    print(f"Class mappings: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    print("\nMerging data...")
    result = features_df.copy().rename(columns={'features': 'outer_product'})
    ordering_map = dict(zip(compound_df['material_id'], compound_df['ordering']))
    result['ordering'] = result['material_id'].map(ordering_map)
    if 'total_magnetization_normalized_atoms' in compound_df.columns:
        moment_map = dict(zip(compound_df['material_id'], compound_df['total_magnetization_normalized_atoms']))
        result['total_magnetization_normalized_atoms'] = result['material_id'].map(moment_map)
    if 'formation_energy_per_atom' in compound_df.columns:
        formation_map = dict(zip(compound_df['material_id'], compound_df['formation_energy_per_atom']))
        result['formation_energy_per_atom'] = result['material_id'].map(formation_map)
    print(f"\nFinal dataset statistics:")
    print(f"  Total materials: {len(result)}")
    print(f"  With ordering labels: {result['ordering'].notna().sum()}")
    if 'total_magnetization_normalized_atoms' in result.columns:
        print(f"  With moment values: {result['total_magnetization_normalized_atoms'].notna().sum()}")
    if 'formation_energy_per_atom' in result.columns:
        print(f"  With formation energy: {result['formation_energy_per_atom'].notna().sum()}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Materials Data Pipeline')
    parser.add_argument('-c', help='Compound properties CSV')
    parser.add_argument('-e', required=True, help='Element properties CSV')
    parser.add_argument('-i', help='Input CSV with material_id and composition')
    parser.add_argument('-o', help='Output pickle for training data')
    parser.add_argument('-f', help='Output pickle for features')
    parser.add_argument('-d', default='cifs', help='CIF directory (default: cifs)')
    parser.add_argument('-s', action='store_true', help='Skip CIF download')
    parser.add_argument('-F', action='store_true', help='Compute features only')
    parser.add_argument('-m', type=float, default=5.0, help='Max distance for neighbors (default: 5.0)')
    parser.add_argument('-x', action='store_true', help='Use extended feature extraction')
    args = parser.parse_args()

    if args.F:
        if not args.i or not args.f:
            parser.error("-F requires -i and -f")
    else:
        if not args.c or not args.o:
            parser.error("Full pipeline requires -c and -o")

    if not Path(args.e).exists():
        print(f"Error: Element file not found: {args.e}")
        return 1
    if not Path(args.d).exists():
        print(f"Error: CIF directory '{args.d}' not found")
        return 1

    print("="*80)
    print(f"MATERIALS DATA PIPELINE ({'EXTENDED' if args.x else 'ORIGINAL'})")
    print("="*80)

    print("\nPreparing element features...")
    element_df = prepare_element_features(args.e, extended=args.x)
    print(f"Loaded {len(element_df)} elements")

    if not args.s:
        print("\nWarning: -s not specified, but CIF download skipped as per assumption.")

    print("\n" + "="*80)
    print("STEP 2: COMPUTING FEATURES")
    print("="*80 + "\n")

    if args.i:
        features_df = compute_features(args.i, element_df, cif_dir=args.d, max_distance=args.m, extended=args.x)
    elif args.c:
        compound_df = pd.read_csv(args.c)
        composition_col = None
        for col_name in ['composition', 'Chemsys', 'formula', 'pretty_formula']:
            if col_name in compound_df.columns:
                composition_col = col_name
                break
        if composition_col is None:
            print("\nNo composition column found. Extracting from CIFs...")
            try:
                compositions_dict = extract_compositions_from_cifs(compound_df['material_id'].values, args.d)
                if not compositions_dict:
                    print("Error: No compositions extracted")
                    return 1
                compound_df['composition'] = compound_df['material_id'].map(compositions_dict)
                composition_col = 'composition'
            except ImportError:
                print("Error: pymatgen not installed")
                return 1
        print(f"Using '{composition_col}' column for compositions")
        temp_csv = Path('temp_compositions.csv')
        temp_df = compound_df[['material_id', composition_col]].copy().rename(
            columns={composition_col: 'composition'}
        )
        temp_df.to_csv(temp_csv, index=False)
        features_df = compute_features(temp_csv, element_df, cif_dir=args.d, max_distance=args.m, extended=args.x)
        temp_csv.unlink()

    if args.f:
        print(f"\nSaving features to {args.f}...")
        with open(args.f, 'wb') as f:
            pickle.dump(features_df, f)
        print("Features saved!")

    if not args.F:
        print("\n" + "="*80)
        print("STEP 3: PREPARING TRAINING DATA")
        print("="*80 + "\n")
        training_data = prepare_training_data(args.c, features_df)
        print(f"\nSaving training data to {args.o}...")
        with open(args.o, 'wb') as f:
            pickle.dump(training_data, f)
        print("Training data saved!")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80 + "\n")
    return 0


if __name__ == '__main__':
    exit(main())
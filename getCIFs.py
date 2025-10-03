import argparse
import pandas as pd
import os
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download CIF files from Materials Project.")
parser.add_argument("--compounds_csv", default="compounds_filtered.csv", help="Filtered compounds CSV.")
parser.add_argument("--cif_dir", default="cifs", help="Directory to save CIF files.")
parser.add_argument("--api_key", required=True, help="Materials Project API key.")
args = parser.parse_args()

# Create the directory if it doesn't exist
os.makedirs(args.cif_dir, exist_ok=True)

compound_df = pd.read_csv(args.compounds_csv)
material_ids = compound_df['material_id'].unique()

with MPRester(args.api_key) as m:
    for material_id in tqdm(material_ids, desc="Downloading CIFs"):
        try:
            structure = m.get_structure_by_material_id(material_id)
            cif_path = os.path.join(args.cif_dir, f"{material_id}.cif")
            structure.to(filename=cif_path)
            print(f"Downloaded {material_id}.cif")
        except Exception as e:
            print(f"Error downloading {material_id}: {e}")

print(f"All downloads attempted. Files saved to {args.cif_dir}")
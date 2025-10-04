# Materials Property Prediction CLI

A command-line tool for predicting magnetic and energetic properties of 
materials using LightGBM models.

## Features

- **Magnetic Ordering**: Classify FM (Ferromagnetic) vs FiM (Ferrimagnetic)
- **Magnetic Moment**: Predict magnetization per atom (μB/atom)
- **Formation Energy**: Predict formation energy per atom (eV/atom)
- Supports default (518 features) and extended (1369 features) modes
- Train/test splits: 90:10 and 70:30

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm scipy pymatgen matplotlib
```

## Directory Structure

```
.
├── pipeline.py              # Feature engineering
├── trainer.py               # Model training
├── predictor.py             # Prediction
├── README.md                # This file
├── data/                    # Data directory
│   ├── Compound_property_data.csv
│   ├── PubChemElements_all.csv
│   └── cifs/                # CIF files (optional)
├── output/                  # Feature output
│   ├── features.pkl         # Default features (518)
│   └── training_data_x.pkl  # Extended features (1369)
├── weights/                 # Default model weights
└── weights_x/               # Extended model weights
```

## Usage

### 1. Compute Features

Generate features from compositions:

```bash
# From CSV
python pipeline.py -c data/Compound_property_data.csv -e data/PubChemElements_all.csv -o output/training_data.pkl -f output/features.pkl -d data/cifs

# Extended mode (-x)
python pipeline.py -c data/Compound_property_data.csv -e data/PubChemElements_all.csv -o output/training_data_x.pkl -f output/features_x.pkl -d data/cifs -x

# Single composition
python pipeline.py -c "Fe2O3" -e data/PubChemElements_all.csv -o output/Fe2O3_features.pkl
```

**Input CSV Format**:
```csv
material_id,composition
mat_001,Fe2O3
mat_002,NiFe2O4
```

### 2. Train Models

Train models on pre-computed features:

```bash
# Default (518 features)
python trainer.py -f output/training_data.pkl -e data/PubChemElements_all.csv -w weights -t all

# Extended (1369 features)
python trainer.py -f output/training_data_x.pkl -e data/PubChemElements_all.csv -w weights_x -t all -x
```

### 3. Make Predictions

Predict using trained models:

```bash
# Default (518 features)
python predictor.py -f output/features.pkl -t all -s 90_10 -w weights

# Extended (1369 features)
python predictor.py -f output/training_data_x.pkl -t all -s 90_10 -w weights_x -x

# Save output
python predictor.py -f output/training_data_x.pkl -t all -s 90_10 -w weights_x -x -o predictions.json

# Verbose output
python predictor.py -f output/features.pkl -t ordering -w weights -v
```

## Command-Line Options

### pipeline.py
```
usage: pipeline.py [-h] [-c INPUT | -c COMPOSITION] -e ELEMENTS [-o OUTPUT] [-f FEATURES] [-d CIF_DIR] [-x]

Options:
  -c FILE|COMPOSITION  Input CSV or single composition (e.g., "Fe2O3")
  -e FILE              Element properties CSV
  -o FILE              Output pickle file (training data)
  -f FILE              Output pickle file (features)
  -d DIR               Directory with CIF files
  -x                   Extended mode (1369 features)
```

### trainer.py
```
usage: trainer.py [-h] -f FEATURES -e ELEMENTS [-w WEIGHTS_DIR] [-t {ordering,moment,formation,all}] [-n N_FEATURES] [-x]

Options:
  -f FILE              Training features pickle file
  -e FILE              Element properties CSV
  -w DIR               Directory to save models (default: weights)
  -t TASK              Tasks to train: ordering, moment, formation, all (default: all)
  -n INT               Number of features to select (default: 100)
  -x                   Extended mode (1369 features)
```

### predictor.py
```
usage: predictor.py [-h] -f FEATURES [-t {all,ordering,moment,formation}] [-s {90_10,70_30}] [-w WEIGHTS_DIR] [-x] [-o OUTPUT] [-v] [-m MAX_DISPLAY]

Options:
  -f FILE              Features file (.pkl or .csv)
  -t TASK              Prediction task: all, ordering, moment, formation (default: all)
  -s SPLIT             Model split: 90_10, 70_30 (default: 90_10)
  -w DIR               Directory with model weights (default: weights)
  -x                   Extended mode (1369 features)
  -o FILE              Output file (.json or .csv)
  -v                   Verbose output
  -m INT               Max results to display (default: 10)
```

## Model Performance (90:10 Split)

- **Magnetic Ordering**:
  - Test Accuracy: ~84%
  - Test AUC-ROC: ~0.88
- **Magnetic Moment**:
  - Test RMSE: ~0.28 μB/atom
  - Test R²: ~0.87
- **Formation Energy**:
  - Test RMSE: ~0.20 eV/atom
  - Test R²: ~0.96

---

### **Changes Made**

1. **Shortened Content**:
   - Removed redundant details (e.g., detailed data formats, installation steps like `requirements.txt`).
   - Simplified directory structure to focus on key files.
   - Condensed usage examples and command-line options.

2. **Short-Form Switches**:
   - Updated all commands to use `-f`, `-e`, `-c`, `-o`, `-d`, `-t`, `-s`, `-w`, `-x`, `-v`, `-m` to match `pipeline.py`, `trainer.py`, and `predictor.py`.
   - Updated CLI options sections to reflect short-form switches.

3. **Extended Mode**:
   - Added `-x` support for 1369-dimensional features.
   - Included examples for both default (`features.pkl`, `weights/`) and extended (`training_data_x.pkl`, `weights_x/`) modes.
   - Clarified feature file expectations (`features` vs `outer_product` columns).

4. **File Naming**:
   - Renamed scripts to match provided files: `compute_features.py` → `pipeline.py`, `train_models.py` → `trainer.py`, `predict_materials.py` → `predictor.py`.
   - Updated data files: `Elemental_property_data.csv` → `PubChemElements_all.csv`, `Compound_property_data.xlsx` → `Compound_property_data.csv`.

5. **Performance**:
   - Kept metrics concise, focusing on key values (accuracy, RMSE, R²).
   - Removed correlation metrics to simplify.

---

### **How to Use**

1. **Test Commands**:
   - Generate features:
     ```bash
     python pipeline.py -c data/Compound_property_data.csv -e data/PubChemElements_all.csv -o output/training_data_x.pkl -f output/features_x.pkl -d data/cifs -x
     ```
   - Train models:
     ```bash
     python trainer.py -f output/training_data_x.pkl -e data/PubChemElements_all.csv -w weights_x -t all -x
     ```
   - Predict:
     ```bash
     python predictor.py -f output/training_data_x.pkl -t all -s 90_10 -w weights_x -x -o predictions.json
     ```
2. **Verify Features**:
   ```python
   import pickle
   import pandas as pd
   with open('output/training_data_x.pkl', 'rb') as f:
       data = pickle.load(f)
   print("Columns:", data.columns.tolist())
   print("Feature column:", 'outer_product' if 'outer_product' in data.columns else 'features')
   print("Vector length:", len(data['outer_product' if 'outer_product' in data.columns else 'features'].iloc[0]))
   ```

---

### **Troubleshooting**

- **Feature Mismatch**:
  - Error: `Expected 1369 features, got X`
  - Fix: Use `-x` with `training_data_x.pkl` from `pipeline.py -x`.
- **Model Not Found**:
  - Error: `Weights directory 'weights_x' not found`
  - Fix: Train with `trainer.py -x` into `weights_x/`.
- **Dependencies**:
  ```bash
  pip install pandas numpy scikit-learn lightgbm scipy pymatgen matplotlib
  ```

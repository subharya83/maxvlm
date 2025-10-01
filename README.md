# Materials Property Prediction CLI

A command-line tool for predicting magnetic and energetic properties of materials using machine learning.

## Features

- **Magnetic Ordering Classification**: Predict FM (Ferromagnetic) vs FiM (Ferrimagnetic)
- **Magnetic Moment Regression**: Predict total magnetization per atom (μB/atom)
- **Formation Energy Regression**: Predict formation energy per atom (eV/atom)
- No API keys required - works with pre-computed features
- Multiple train/test split models (90:10 and 70:30)

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- lightgbm
- scipy

### Directory Structure

```
.
├── predict_materials.py      # Main prediction script
├── train_models.py           # Model training script
├── compute_features.py       # Feature engineering script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Data directory
│   ├── Elemental_property_data.csv
│   ├── Compound_property_data.xlsx (for training)
│   └── features.pkl (computed features)
└── weights/                  # Model weights directory
    ├── element_features.pkl
    ├── magnetic_ordering_model_90_10.pkl
    ├── magnetic_ordering_model_70_30.pkl
    ├── magnetic_moment_model_90_10.pkl
    ├── magnetic_moment_model_70_30.pkl
    ├── formation_energy_model_90_10.pkl
    └── formation_energy_model_70_30.pkl
```

## Usage

### 1. Prepare Data Files

You need two data files:
- `Elemental_property_data.csv` - Element properties (Symbol, Atomic number, Group, Period, Density, Electronegativity, Ionisation Energy, Atomic radius, UE)
- Pre-computed features file (`.pkl` format) OR composition data to compute features

### 2. Compute Features (Optional)

If you have material compositions but not pre-computed features:

```bash
# From a CSV file with compositions
python compute_features.py \
    --input data/compositions.csv \
    --elements data/Elemental_property_data.csv \
    --output data/features.pkl

# For a single composition
python compute_features.py \
    --composition "Fe2O3" \
    --elements data/Elemental_property_data.csv \
    --output data/Fe2O3_features.pkl
```

**CSV format for compositions:**
```csv
material_id,composition
mat_001,Fe2O3
mat_002,NiFe2O4
mat_003,CoFe2O4
```

### 3. Train Models

If you have training data with known properties:

```bash
# Train all models
python train_models.py \
    --features data/training_features.pkl \
    --elements data/Elemental_property_data.csv \
    --weights-dir weights

# Train specific tasks
python train_models.py \
    --features data/training_features.pkl \
    --elements data/Elemental_property_data.csv \
    --tasks ordering moment \
    --weights-dir weights
```

**Training data format** (pickle file with DataFrame):
```python
# DataFrame with columns:
# - material_id: str
# - outer_product: list/array (518-dimensional feature vector)
# - ordering: int (0=FM, 1=FiM) [for classification]
# - total_magnetization_normalized_atoms: float [for moment prediction]
# - formation_energy_per_atom: float [for formation energy prediction]
```

### 4. Make Predictions

Once models are trained:

```bash
# Predict all properties
python predict_materials.py \
    --features data/features.pkl \
    --task all \
    --split 90_10

# Predict specific property
python predict_materials.py \
    --features data/features.pkl \
    --task ordering \
    --split 90_10

# Save predictions to file
python predict_materials.py \
    --features data/features.pkl \
    --task all \
    --output predictions.json

# Use 70:30 split model
python predict_materials.py \
    --features data/features.pkl \
    --split 70_30 \
    --output predictions.csv
```

## Command-Line Options

### predict_materials.py

```
usage: predict_materials.py [-h] --features FEATURES
                           [--task {all,ordering,moment,formation}]
                           [--split {90_10,70_30}]
                           [--weights-dir WEIGHTS_DIR]
                           [--output OUTPUT] [--verbose]

Options:
  --features FEATURES   Path to features file (.pkl or .csv)
  --task TASK          Prediction task: all, ordering, moment, formation
  --split SPLIT        Model split: 90_10 or 70_30 (default: 90_10)
  --weights-dir DIR    Directory containing model weights
  --output, -o FILE    Output file (.json or .csv)
  --verbose, -v        Verbose output
```

### train_models.py

```
usage: train_models.py [-h] --features FEATURES --elements ELEMENTS
                      [--weights-dir WEIGHTS_DIR]
                      [--tasks {ordering,moment,formation,all} [...]]

Options:
  --features FEATURES   Path to training features pickle file
  --elements ELEMENTS   Path to element properties CSV
  --weights-dir DIR     Directory to save models (default: weights)
  --tasks TASKS         Tasks to train (default: all)
```

### compute_features.py

```
usage: compute_features.py [-h] [--input INPUT] [--composition COMPOSITION]
                           --elements ELEMENTS [--output OUTPUT]
                           [--distance DISTANCE] [--neighbors NEIGHBORS]

Options:
  --input, -i FILE      Input CSV with material_id and composition
  --composition, -c     Single composition (e.g., "Fe2O3")
  --elements ELEMENTS   Path to element properties CSV
  --output, -o FILE     Output pickle file
  --distance DISTANCE   Assumed atomic distance in Å (default: 3.0)
  --neighbors NEIGHBORS Assumed coordination number (default: 12)
```

## Model Performance

### Magnetic Ordering (90:10 split)
- Training Accuracy: ~85%
- Test Accuracy: ~84%
- Test AUC-ROC: ~0.88

### Magnetic Moment (90:10 split)
- Test RMSE: ~0.28 μB/atom
- Test R²: ~0.87
- Test Correlation: ~0.93

### Formation Energy (90:10 split)
- Test RMSE: ~0.20 eV/atom
- Test R²: ~0.96
- Test Correlation: ~0.98


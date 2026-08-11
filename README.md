# Raman Spectroscopy Preprocessing & Synthetic Data Generation

Synthetic Raman spectra generation, baseline polynomial modeling, peak profile simulation (Gaussian, Lorentzian, Voigt, Moffat), and dual-supervised ResNet deep learning architecture for spectral preprocessing.

## Features

- **Synthetic Spectral Generator**: Generates synthetic Raman spectra with customizable peak profiles, baseline shifts, and noise (Gaussian, Cosmic Ray artifacts).
- **Dual Supervised ResNet**: Deep learning model for concurrent baseline removal, denoising, and component extraction.
- **Parametric Peak & Baseline Modules**: Built using `astropy.modeling` and `scipy`.

## Installation

```bash
# Clone the repository
git clone https://github.com/bogomilkissiov/raman-preprocessing.git
cd raman-preprocessing

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Test Synthetic Spectra Generation
```bash
python test_composite_generation.py
```

### 2. Train Model
```bash
python training.py
```

## Project Structure

```text
├── spectra_class.py          # Core spectra data container class
├── spectra_generator.py      # Synthetic Raman spectral generator
├── dual_supervised_resnet.py # ResNet architecture & loss functions
├── training.py               # Training loop and evaluation
├── test_*.py                 # Test scripts for generator components
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

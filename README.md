# Age Estimation Model

This model combines the SpeechBrain ECAPA-TDNN speaker embedding model with SVR/ANN regressors to predict speaker age from audio input. Several model variants are available, trained on different datasets and feature sets.
We advice to use the combined ANN model with extended features for best overall performance.

## Model Details
- **Architectures**: Multiple variants available:
  1. VoxCeleb2 SVR with extended features:
     - ECAPA-TDNN embeddings (192-dim) + Librosa features (31-dim)
     - MAE: 7.88 years on VoxCeleb2 test set
  2. VoxCeleb2 SVR with base features:
     - ECAPA-TDNN embeddings only (192-dim)
     - MAE: 7.89 years on VoxCeleb2 test set
  3. TIMIT ANN with base features:
     - ECAPA-TDNN embeddings only (192-dim)
     - MAE: 4.95 years on TIMIT test set
  4. Combined (VoxCeleb2 + TIMIT) ANN with extended features:
     - ECAPA-TDNN embeddings (192-dim) + Librosa features (31-dim)
     - MAE: 6.93 years on combined test set

- **Feature Sets**:
  1. Base Features (192 dimensions):
     - SpeechBrain ECAPA-TDNN embeddings
  2. Extended Features (223 dimensions):
     - SpeechBrain ECAPA-TDNN embeddings (192-dim)
     - Librosa acoustic features (31-dim):
       - 13 MFCCs
       - 13 Delta MFCCs
       - Zero crossing rate
       - Spectral centroid
       - Spectral bandwidth
       - Spectral contrast
       - Spectral flatness

- **Training Data**:
  - VoxCeleb2 dataset: Celebrity voices from YouTube interviews
  - TIMIT dataset: High-quality studio recordings
  - Combined dataset: Mixture of VoxCeleb2 and TIMIT
  - The age data was taken from VOXCELEB ENRICHMENT FOR AGE AND GENDER RECOGNITION [dataset](https://github.com/hechmik/voxceleb_enrichment_age_gender)

- **Audio Processing**:
  - Input format: Any audio file format supported by soundfile
  - Automatically converted to: 16kHz, mono, single channel
  - Voice activity detection applied to extract primary voiced segments

## Installation

You can install the package directly from GitHub:

```bash
# Combined ANN model with extended features, also suitable for all other models
pip install git+https://github.com/griko/voice-age-regression.git[full]  # MAE 6.93 years on the combined VoxCeleb2 + TIMIT test set
# OR install a specific model variant
pip install git+https://github.com/griko/voice-age-regression.git[svr-ecapa-voxceleb2]  # VoxCeleb2 SVR with base features, MAE 7.89 years
pip install git+https://github.com/griko/voice-age-regression.git[svr-ecapa-librosa-voxceleb2]  # VoxCeleb2 SVR with extended features, MAE 7.88 years
pip install git+https://github.com/griko/voice-age-regression.git[ann-ecapa-timit]  # TIMIT ANN with base features, MAE 4.95 years
```

## Usage

```python
from voice_age_regression import AgeRegressionPipeline

# Load one of the available models:
# 1. VoxCeleb2 SVR with extended features
regressor = AgeRegressionPipeline.from_pretrained(
    "griko/age_reg_svr_ecapa_librosa_voxceleb2"
)

# 2. VoxCeleb2 SVR with base features
regressor = AgeRegressionPipeline.from_pretrained(
    "griko/age_reg_svr_ecapa_voxceleb2"
)

# 3. TIMIT ANN with base features
regressor = AgeRegressionPipeline.from_pretrained(
    "griko/age_reg_ann_ecapa_timit"
)

# 4. Combined ANN with extended features
regressor = AgeRegressionPipeline.from_pretrained(
    "griko/age_reg_ann_ecapa_librosa_combined"
)

# Single file prediction
result = regressor("path/to/audio.wav")
print(f"Predicted age: {result[0]:.1f} years")

# Batch prediction
results = regressor(["audio1.wav", "audio2.wav"])
print(f"Predicted ages: {[f'{age:.1f}' for age in results]} years")
```

## Limitations
- Performance may vary on:
  - Different audio qualities and recording conditions
  - Multiple simultaneous speakers

## Citation
If you use this model in your research, please cite:
```bibtex
TBD
```

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- VoxCeleb2 dataset for providing training data
- SpeechBrain team for their excellent speech processing toolkit

from setuptools import setup, find_packages

setup(
    name="voice-age-regressor",
    version="0.1.0",
    description="Age regression pipeline using SpeechBrain ECAPA embeddings and optional additional features and SVR/ANN regressors",
    author="Gregory Koushnir",
    author_email="koushgre@post.bgu.ac.il",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn",
        "pandas",
        "soundfile",
        "speechbrain",
        "torch",
        "torchaudio",
    ],
    extras_require={
        "svr-ecapa-voxceleb2": [],  # No additional requirements for SVR models
        "ann-ecapa-timit": ["tensorflow"],  # Required for ANN models
        "svr-ecapa-librosa-voxceleb2": ["librosa"],  # Required for extended features
        "full": [  # All possible requirements
            "tensorflow",
            "librosa",
        ],
        "ann-ecapa-librosa-combined": [  # All possible requirements
            "tensorflow",
            "librosa",
        ],
    },
    python_requires=">=3.8"
)
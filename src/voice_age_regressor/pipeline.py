import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Union, List, Dict, Optional
from speechbrain.inference.speaker import EncoderClassifier

class AgeRegressionPipeline:
    def __init__(self, model, scaler, model_type: str = "svr", feature_set: str = "ecapa", device="cpu"):
        """
        Initialize the pipeline with model and scaler.
        
        Args:
            model: Trained model (SVR or ANN)
            scaler: Fitted StandardScaler
            model_type: Type of model ("svr" or "ann")
            feature_set: Feature set to use ("ecapa" or "ecapa_librosa")
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.model = model
        self.scaler = scaler
        self.model_type = model_type
        self.feature_set = feature_set
        
        # Initialize ECAPA encoder
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)}
        )
        
        # Initialize feature names
        self.ecapa_features = [f"{i}_speechbrain_embedding" for i in range(192)]
        self.librosa_features = [
            'zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 
            'spectral_contrast', 'spectral_flatness'  #, 'tonnetz'
        ]
        self.mfcc_features = [f'mfcc_{i}' for i in range(13)] + [f'd_mfcc_{i}' for i in range(13)]
        
        self.feature_names = (
            self.ecapa_features + self.librosa_features + self.mfcc_features 
            if feature_set == "ecapa_librosa" else self.ecapa_features
        )
        
    def _process_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Process audio to 16kHz mono."""
        if len(waveform.shape) > 1:
            waveform = torch.mean(waveform, dim=-1)
        
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        
        if waveform.abs().max() > 1:
            waveform = waveform / waveform.abs().max()
        
        return waveform

    def _extract_librosa_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract Librosa features from audio."""
        features = {}
        
        # Extract MFCCs and delta MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
        
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfcc[i])
            features[f'd_mfcc_{i}'] = np.mean(delta_mfcc[i])
        
        # Extract other features
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
        # features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
        
        return features

    def preprocess(self, audio_input: Union[str, List[str], np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Preprocess audio input."""
        def load_audio(audio_file):
            wave, sr = sf.read(audio_file)
            wave = torch.from_numpy(wave).float()
            return wave, sr

        if isinstance(audio_input, list):
            waveforms = []
            wav_lens = []
            for audio_file in audio_input:
                wave, sr = load_audio(audio_file)
                wave = self._process_audio(wave, sr)
                waveforms.append(wave)
                wav_lens.append(wave.shape[0])

            max_len = max(wav_lens)
            padded_waveforms = [torch.nn.functional.pad(wave, (0, max_len - wave.shape[0])) for wave in waveforms]
            inputs = torch.stack(padded_waveforms).to(self.device)
            wav_lens = torch.tensor(wav_lens, dtype=torch.float32) / max_len
            return {"inputs": inputs, "wav_lens": wav_lens.to(self.device)}

        if isinstance(audio_input, str):
            waveform, sr = load_audio(audio_input)
            waveform = self._process_audio(waveform, sr)
            inputs = waveform.unsqueeze(0).to(self.device)
            wav_lens = torch.tensor([1.0], dtype=torch.float32).to(self.device)
            return {"inputs": inputs, "wav_lens": wav_lens}

        elif isinstance(audio_input, (np.ndarray, torch.Tensor)):
            waveform = torch.as_tensor(audio_input).float()
            waveform = self._process_audio(waveform, 16000)
            inputs = waveform.unsqueeze(0).to(self.device)
            wav_lens = torch.tensor([1.0], dtype=torch.float32).to(self.device)
            return {"inputs": inputs, "wav_lens": wav_lens}
            
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """Extract embeddings and features."""
        with torch.no_grad():
            # Get ECAPA embeddings
            embeddings = self.encoder.encode_batch(inputs["inputs"], inputs["wav_lens"])
            embeddings = embeddings.squeeze(1).cpu().numpy()
            
            # Create DataFrame with ECAPA features
            features_df = pd.DataFrame(embeddings, columns=self.ecapa_features)
            
            # Extract Librosa features if needed
            if self.feature_set == "ecapa_librosa":
                audio_np = inputs["inputs"].cpu().numpy()
                for i in range(len(audio_np)):
                    librosa_feats = self._extract_librosa_features(audio_np[i], sr=16000)
                    for feat_name, feat_value in librosa_feats.items():
                        features_df.at[i, feat_name] = feat_value
            
            return features_df

    def postprocess(self, features_df: pd.DataFrame) -> List[float]:
        """Get predictions from features."""
        # Scale features
        scaled_features = self.scaler.transform(features_df[self.feature_names])
        
        # Make predictions
        if self.model_type == "ann":
            predictions = self.model.predict(scaled_features)
            return predictions.flatten().tolist()
        else:  # svr
            return self.model.predict(scaled_features).tolist()

    def __call__(self, audio_input: Union[str, List[str], np.ndarray, torch.Tensor]) -> List[float]:
        """Run the pipeline on the input."""
        inputs = self.preprocess(audio_input)
        features = self.forward(inputs)
        return self.postprocess(features)

    def save_pretrained(self, save_directory: str):
        """Save model components."""
        import os
        import joblib
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model and scaler
        if self.model_type == "ann":
            self.model.save(os.path.join(save_directory, "model.h5"))
        else:
            joblib.dump(self.model, os.path.join(save_directory, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(save_directory, "scaler.joblib"))
        
        # Save the configuration
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "feature_set": self.feature_set
            }, f)

    @classmethod
    def from_pretrained(cls, model_path: str, device="cpu"):
        """Load model components."""
        import os
        import joblib
        import json
        from huggingface_hub import hf_hub_download

        if os.path.isdir(model_path):
            base_path = model_path
            load_file = lambda f: os.path.join(base_path, f)
        else:
            load_file = lambda f: hf_hub_download(repo_id=model_path, filename=f)

        # Load configuration
        with open(load_file("config.json"), "r") as f:
            config = json.load(f)

        # Load model based on type
        model_type = config["model_type"]
        if model_type == "ann":
            import tensorflow as tf
            model = tf.keras.models.load_model(load_file("model.h5"))
            model.compile(loss='mse')
        else:
            model = joblib.load(load_file("model.joblib"))
            
        # Load scaler
        scaler = joblib.load(load_file("scaler.joblib"))
        
        # Create pipeline instance
        pipeline = cls(
            model=model,
            scaler=scaler,
            model_type=model_type,
            feature_set=config["feature_set"],
            device=device
        )
        
        return pipeline
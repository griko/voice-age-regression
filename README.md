# voice-age-regression
Two different models combine the SpeechBrain ECAPA-TDNN speaker embeddings with either an ANN or SVR regressor to predict speaker age from audio input. One model utilizes the 192-dimensional SpeechBrain ECAPA embeddings alone, while the other enhances these embeddings with an additional 31 features extracted using Librosa.

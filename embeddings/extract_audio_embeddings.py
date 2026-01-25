import os
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm

DATA_DIR = "data/processed"
SAVE_DIR = "embeddings/saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

LABEL_MAP = {
    "drums": 0,
    "keys": 1
}

audio_embeddings = []
labels = []
filenames = []

def extract_embedding(wav_path):
    # Load audio (must be mono, 16kHz)
    waveform, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Convert to tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Run YAMNet
    scores, embeddings, spectrogram = yamnet_model(waveform)

    # embeddings shape: (time_frames, 1024)
    embedding = tf.reduce_mean(embeddings, axis=0)

    return embedding.numpy()

def main():
    for class_name in ["drums", "keys"]:
        class_dir = os.path.join(DATA_DIR, class_name)

        for file in tqdm(sorted(os.listdir(class_dir)), desc=f"Embedding {class_name}"):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(class_dir, file)
            emb = extract_embedding(path)

            audio_embeddings.append(emb)
            labels.append(LABEL_MAP[class_name])
            filenames.append(file)

    np.save(os.path.join(SAVE_DIR, "audio_embeddings.npy"), np.vstack(audio_embeddings))
    np.save(os.path.join(SAVE_DIR, "labels.npy"), np.array(labels))
    np.save(os.path.join(SAVE_DIR, "filenames.npy"), np.array(filenames))

    print("✅ Audio embeddings extracted using YAMNet.")

if __name__ == "__main__":
    main()

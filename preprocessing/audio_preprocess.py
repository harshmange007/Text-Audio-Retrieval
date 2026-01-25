import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

SAMPLE_RATE = 16000
DURATION = 5.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

os.makedirs(PROCESSED_DIR, exist_ok=True)

metadata_rows = []

def process_class(class_name):
    input_dir = os.path.join(RAW_DIR, class_name)
    output_dir = os.path.join(PROCESSED_DIR, class_name)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    files.sort()

    for idx, file in enumerate(tqdm(files, desc=f"Processing {class_name}")):
        input_path = os.path.join(input_dir, file)

        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

        # Trim or pad
        if len(audio) > NUM_SAMPLES:
            audio = audio[:NUM_SAMPLES]
        else:
           audio = librosa.util.fix_length(audio, size=NUM_SAMPLES)


        # Normalize
        audio = audio / (abs(audio).max() + 1e-8)

        new_filename = f"{class_name}_{idx+1:02d}.wav"
        output_path = os.path.join(output_dir, new_filename)

        sf.write(output_path, audio, SAMPLE_RATE)

        metadata_rows.append({
            "file_name": new_filename,
            "class": class_name,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": DURATION
        })

def main():
    for class_name in ["drums", "keys"]:
        process_class(class_name)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(
        os.path.join(PROCESSED_DIR, "metadata_clean.csv"),
        index=False
    )

    print("✅ Audio preprocessing complete.")
    print("✅ Clean metadata generated.")

if __name__ == "__main__":
    main()

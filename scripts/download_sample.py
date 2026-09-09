from pathlib import Path

import soundfile as sf
from datasets import load_dataset


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sample.wav"


print("Downloading LibriSpeech sample...")

dataset = load_dataset(
    "AudioLLMs/librispeech_test_clean",
    split="test"
)

sample = dataset[2]

audio = sample["audio"]["array"]
sample_rate = sample["audio"]["sampling_rate"]

sf.write(
    OUTPUT_FILE,
    audio,
    sample_rate
)

print(f"\nSaved to: {OUTPUT_FILE}")
print(f"Sample Rate : {sample_rate}")
print(f"Duration    : {len(audio)/sample_rate:.2f} sec")
print(f"Speaker ID  : {sample['speaker_id']}")
print(f"Text        : {sample['text']}")
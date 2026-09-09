"""
SemBind-Audio

Audio Dataset Manager

Loads LibriSpeech audio files and their corresponding transcripts.
"""

from pathlib import Path
import soundfile as sf


class AudioDataset:
    """
    Dataset manager for the LibriSpeech test-clean dataset.
    """

    def __init__(self, root="data/raw/LibriSpeech/test-clean"):

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.root}"
            )

        self.audio_files = sorted(self.root.rglob("*.flac"))

    def __len__(self):
        return len(self.audio_files)

    def get_audio_path(self, index: int) -> Path:
        return self.audio_files[index]

    def load(self, index: int):

        audio_path = self.audio_files[index]

        audio, sample_rate = sf.read(audio_path)

        # ----------------------------------------------------
        # Extract speaker/chapter information
        # ----------------------------------------------------

        speaker_id = audio_path.parts[-3]
        chapter_id = audio_path.parts[-2]
        utterance_id = audio_path.stem

        # ----------------------------------------------------
        # Locate transcript file
        # Example:
        # 1089-134686.trans.txt
        # ----------------------------------------------------

        transcript_file = (
            audio_path.parent
            / f"{speaker_id}-{chapter_id}.trans.txt"
        )

        if not transcript_file.exists():
            raise FileNotFoundError(
                f"Transcript file not found:\n{transcript_file}"
            )

        transcript = ""

        with open(transcript_file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(" ", 1)

                if len(parts) != 2:
                    continue

                uid, text = parts

                if uid == utterance_id:
                    transcript = text
                    break

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "utterance_id": utterance_id,
            "transcript": transcript,
            "path": str(audio_path),
        }

    def info(self):

        print("=" * 60)
        print("SemBind-Audio Dataset")
        print("=" * 60)
        print(f"Dataset Root : {self.root}")
        print(f"Samples      : {len(self)}")
        print("=" * 60)
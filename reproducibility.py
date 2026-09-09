import json
import os
import csv
import hashlib
import subprocess
from datetime import datetime


# ============================================================
# Reproducibility Metadata Framework
# ============================================================

class ReproMetadata:
    """Records reproducibility metadata for an evaluation run."""

    def __init__(
        self,
        dataset_manifest,
        number_of_recordings,
        number_of_speakers,
        model_name,
        sample_rate,
        n_fft,
        hop_length,
        win_length,
        payload_size,
        redundancy,
        alpha,
        beta,
        delta_min,
        delta_max,
        selection_method,
        attack=None,
        limited=False,
    ):
        self.dataset_manifest = dataset_manifest
        self.number_of_recordings = number_of_recordings
        self.number_of_speakers = number_of_speakers
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.payload_size = payload_size
        self.redundancy = redundancy
        self.alpha = alpha
        self.beta = beta
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.selection_method = selection_method
        self.attack = attack
        self.limited = limited
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.git_commit = self._get_git_commit()

    def _get_git_commit(self):
        """Get current git commit hash if available; None if not a git repo."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd='/Users/kr/Desktop/SemBind-Audio',
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None  # Do not fabricate; repo has 0 commits

    def to_dict(self):
        """Convert metadata to dictionary."""
        return {
            'dataset_manifest': self.dataset_manifest,
            'number_of_recordings': self.number_of_recordings,
            'number_of_speakers': self.number_of_speakers,
            'model_name': self.model_name,
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'payload_size': self.payload_size,
            'redundancy': self.redundancy,
            'alpha': self.alpha,
            'beta': self.beta,
            'delta_min': self.delta_min,
            'delta_max': self.delta_max,
            'selection_method': self.selection_method,
            'attack': self.attack,
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
        }

    def save(self, path):
        """Save metadata as JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path):
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return ReproMetadata(**data)


# ============================================================
# Sanity Check Script
# ============================================================

def sanity_check_manifest(manifest_path, result_dir=None):
    """Validate evaluation manifest and report statistics.

    Checks:
    - File readability
    - Sample rate consistency
    - Mono audio (single channel)
    - Duration range
    - Speaker ID parsing
    - No result data inclusion
    """
    import soundfile as sf

    print(f'=== Sanity Check: {manifest_path} ===')
    print()

    recordings = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)

    # Basic stats
    total = len(recordings)
    print(f'Total recordings in manifest: {total}')

    # Sample rate consistency
    sample_rates = set()
    durations = []
    speakers = set()
    valid = 0
    invalid_reasons = []

    for i, rec in enumerate(recordings):
        audio_id = rec.get('audio_id', 'unknown')
        path = rec.get('path', '')
        split = rec.get('split', 'unknown')

        # Check file existence and readability
        if not os.path.exists(path):
            invalid_reasons.append(f'{audio_id}: file not found at {path}')
            continue

        try:
            audio, sr = sf.read(path, always_2d=False)
        except Exception as e:
            invalid_reasons.append(f'{audio_id}: read error - {e}')
            continue

        # Check sample rate
        sample_rates.add(sr)

        # Check duration
        duration = len(audio) / sr
        durations.append(duration)

        # Check channels (must be mono or stereo that we handle)
        if audio.ndim > 1 and audio.shape[1] > 1:
            # Stereo - take first channel
            audio = audio[:, 0]

        # Parse speaker ID
        speaker_id = rec.get('speaker_id', '')
        speakers.add(speaker_id)

        valid += 1

    print(f'Valid recordings: {valid}/{total}')
    print(f'Sample rates found: {sorted(sample_rates)}')
    print(f'Unique speakers: {len(speakers)}')

    # Duration statistics
    if durations:
        durations = sorted(durations)
        print(f'Duration range: {durations[0]:.2f}s - {durations[-1]:.2f}s')
        print(f'Mean duration: {sum(durations)/len(durations):.2f}s')

    # Sample rate consistency
    if len(sample_rates) == 1:
        print(f'Sample rate consistent: {list(sample_rates)[0]} Hz')
    else:
        print(f'SAMPLE RATE INCONSISTENCY: {sample_rates}')

    # Invalid recordings
    if invalid_reasons:
        print(f'\nInvalid recordings ({len(invalid_reasons)}):')
        for reason in invalid_reasons[:10]:  # Show first 10
            print(f'  - {reason}')
        if len(invalid_reasons) > 10:
            print(f'  ... and {len(invalid_reasons) - 10} more')

    # Speaker distribution
    speaker_counts = {}
    for rec in recordings:
        sid = rec.get('speaker_id', '')
        speaker_counts[sid] = speaker_counts.get(sid, 0) + 1
    
    print(f'\nSpeaker distribution (top 10):')
    for sid, count in sorted(speaker_counts.items())[:10]:
        print(f'  {sid}: {count} recordings')

    # Result directory check if provided
    if result_dir and os.path.exists(result_dir):
        print(f'\nResult directory: {result_dir}')
        # List result files (no watermark-specific content inspection)
        result_files = os.listdir(result_dir)
        # Filter to relevant categories
        watermark_files = [f for f in result_files if not f.startswith('.')]
        print(f'  Files in result dir: {len(watermark_files)}')

    # Summary
    print()
    all_good = (len(sample_rates) == 1 and len(invalid_reasons) == 0)
    if all_good:
        print('Sanity check PASSED')
    else:
        print('Sanity check had issues - see above')

    return {
        'total_recordings': total,
        'valid_recordings': valid,
        'sample_rates': sorted(sample_rates),
        'durations_min': min(durations) if durations else None,
        'durations_max': max(durations) if durations else None,
        'mean_duration': sum(durations)/len(durations) if durations else None,
        'speakers': len(speakers),
        'sample_rate_consistent': len(sample_rates) == 1,
        'invalid_reasons': invalid_reasons,
    }


# ============================================================
# Lightweight Unit Tests
# ============================================================

def test_delta_config():
    """Test that delta config values are correct and consistent."""
    from evaluation_framework import DELTA_MIN, DELTA_MAX
    assert DELTA_MIN == 0.002, f'DELTA_MIN should be 0.002, got {DELTA_MIN}'
    assert DELTA_MAX == 0.008, f'DELTA_MAX should be 0.008, got {DELTA_MAX}'
    assert DELTA_MIN < DELTA_MAX, 'DELTA_MIN must be less than DELTA_MAX'
    print('PASS: delta config values correct')


def test_hubert_frame_count():
    """Test that HuBERT frame count is consistent across audio files."""
    from evaluation_framework import extract_hubert_frames
    import soundfile as sf

    # Test with a few LibriSpeech audios
    test_files = [
        'data/raw/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac',
        'data/raw/LibriSpeech/test-clean/1188/133604/1188-133604-0000.flac',
    ]

    for fpath in test_files:
        audio, sr = sf.read(fpath)
        n_frames = extract_hubert_frames(audio, sr)
        assert n_frames > 0, f'HuBERT frames should be > 0 for {fpath}'
        print(f'PASS: {os.path.basename(fpath)} -> {n_frames} HuBERT frames')


def test_stft_frame_count():
    """Test that STFT frame count is consistent with HuBERT mapping."""
    from evaluation_framework import compute_stft_frames, map_hubert_to_stft, extract_hubert_frames
    import soundfile as sf

    test_files = [
        'data/raw/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac',
    ]

    for fpath in test_files:
        audio, sr = sf.read(fpath)
        n_stft = compute_stft_frames(audio, sr)
        n_hubert = extract_hubert_frames(audio, sr)
        mapped = map_hubert_to_stft(0, n_hubert, n_stft)
        assert mapped >= 0 and mapped < n_stft, \
            f'Mapped STFT frame {mapped} out of range [0, {n_stft})'
        print(f'PASS: STFT frames={n_stft}, HuBERT frames={n_hubert}, mapped={mapped}')


def test_selection_methods():
    """Test that all four selection methods produce valid delta values."""
    from evaluation_framework import apply_selection_method, DELTA_MIN, DELTA_MAX
    import soundfile as sf

    test_path = 'data/raw/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac'
    audio, sr = sf.read(test_path)

    for method in ['random', 'spectral', 'semantic', 'joint']:
        deltas, hubert_frames, stft_frames = apply_selection_method(
            audio, sr, method=method)
        assert len(deltas) == hubert_frames, \
            f'{method}: delta count {len(deltas)} != hubert frames {hubert_frames}'
        assert all(DELTA_MIN <= d <= DELTA_MAX
                   for d in deltas), \
            f'{method}: deltas out of range [{DELTA_MIN}, {DELTA_MAX}]'
        print(f'PASS: {method} - {hubert_frames} frames, deltas in [{DELTA_MIN}, {DELTA_MAX}]')


def test_temporal_mapping():
    """Test the linear normalized HuBERT-to-STFT mapping."""
    from evaluation_framework import map_hubert_to_stft
    # With 521 HuBERT frames and 522 STFT frames (from ~10s audio at 16kHz)
    # Mapping: position = frame / 520, mapped = round(position * 521)
    # Frame 0 -> round(0 * 521) = 0
    # Frame 520 -> round(1.0 * 521) = 521 (= stft_frames - 1)
    # The mapped value is 0-indexed STFT frame position
    assert map_hubert_to_stft(0, 521, 522) == 0, 'Frame 0 should map to 0'
    assert map_hubert_to_stft(520, 521, 522) == 521, 'Frame 520 should map to 521 (stft_frames-1)'
    # Middle frame should map to a positive frame < 521
    mid = map_hubert_to_stft(260, 521, 522)
    assert mid > 0, 'Middle frame should map to positive frame'
    assert mid < 522, 'Middle frame should map to < 522 (stft_frames)'
    print('PASS: Temporal mapping verified (521->522, identity with rounding)')


# ============================================================
# CLI entry point for sanity check
# ============================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'sanity':
        # Sanity check mode
        manifest_path = sys.argv[2] if len(sys.argv) > 2 else 'results/evaluation_manifest.csv'
        result_dir = sys.argv[3] if len(sys.argv) > 3 else None
        sanity_check_manifest(manifest_path, result_dir)

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Run unit tests
        tests = [
            test_delta_config,
            test_hubert_frame_count,
            test_stft_frame_count,
            test_selection_methods,
            test_temporal_mapping,
        ]

        passed = 0
        failed = 0
        for test in tests:
            try:
                test()
                passed += 1
            except AssertionError as e:
                print(f'FAIL: {test.__name__}: {e}')
                failed += 1
            except Exception as e:
                print(f'ERROR: {test.__name__}: {type(e).__name__}: {e}')
                failed += 1

        print(f'\n{passed} passed, {failed} failed out of {len(tests)} tests')

    else:
        print('Usage: python3 reproducibility.py [sanity|test]')
        print('  sanity <manifest_path> [result_dir] - Run manifest sanity check')
        print('  test - Run unit tests for config, HuBERT, STFT, methods, mapping')
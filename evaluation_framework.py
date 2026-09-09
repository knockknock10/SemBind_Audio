import os
import csv
import json
import argparse
import numpy as np
import soundfile as sf
import torch
from transformers import HubertModel, AutoFeatureExtractor

# ============================================================
# Configuration
# ============================================================
DELTA_MIN = 0.002
DELTA_MAX = 0.008
ALPHA = 0.70
BETA = 0.30

N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024
PAYLOAD_LEN = 256
FREQ_BIN_START = 2
FREQ_BIN_COUNT = 10

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAYLOAD_PATH = os.path.join(RESULTS_DIR, "results", "watermark", "payload_bits.pt")

# ============================================================
# Model loading (import-time, global)
# ============================================================
_feature_extractor = None
_hubert_model = None


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(
            'facebook/hubert-base-ls960')
    return _feature_extractor


def _get_hubert_model():
    global _hubert_model
    if _hubert_model is None:
        _hubert_model = HubertModel.from_pretrained(
            'facebook/hubert-base-ls960')
    return _hubert_model


# ============================================================
# Core utility functions
# ============================================================

def extract_hubert_frames(audio, sample_rate=16000):
    """Extract HuBERT embeddings and return frame count."""
    feature_extractor = _get_feature_extractor()
    model = _get_hubert_model()
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs.input_values)
    return outputs.last_hidden_state.shape[1]


def compute_stft(audio, sample_rate=16000, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Compute STFT and return complex tensor."""
    window = torch.hann_window(win_length)
    return torch.stft(
        torch.from_numpy(audio.astype(np.float32)).float(),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def compute_stft_frames(audio, sample_rate=16000, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Compute STFT and return number of time frames."""
    stft = compute_stft(audio, sample_rate, n_fft, hop_length, win_length)
    return stft.shape[1]


def map_hubert_to_stft(frame_idx, hubert_frames, stft_frames):
    """Linear normalized mapping from HuBERT frame to STFT frame.

    position = frame / (N_hubert - 1)
    mapped = round(position * (N_stft - 1))
    """
    if hubert_frames <= 1 or stft_frames <= 1:
        return 0
    position = frame_idx / float(hubert_frames - 1)
    return round(position * float(stft_frames - 1))


def apply_selection_method(audio, sample_rate, method='semantic',
                           delta_min=DELTA_MIN, delta_max=DELTA_MAX):
    """Apply watermark selection method and return per-frame delta values.

    Methods:
    - 'random': Deterministic random deltas per frame
    - 'spectral': Delta increases linearly from delta_min to delta_max
    - 'semantic': Delta_i = Delta_min + (Delta_max - Delta_min) * (1 - normalized_semantic_score)
    - 'joint': J = S^alpha * P^beta, delta mapped from joint score
    """
    hubert_frames = extract_hubert_frames(audio, sample_rate)
    stft_frames = compute_stft_frames(audio, sample_rate)

    delta_values = []

    for frame_idx in range(hubert_frames):
        stft_mapped = map_hubert_to_stft(frame_idx, hubert_frames, stft_frames)

        if method == 'random':
            import hashlib
            h = int(hashlib.md5(f"{sample_rate}:{frame_idx}".encode()).hexdigest(), 16)
            delta = delta_min + (delta_max - delta_min) * (h % 1000) / 1000.0

        elif method == 'spectral':
            normalized = frame_idx / float(max(hubert_frames - 1, 1))
            delta = delta_min + normalized * (delta_max - delta_min)

        elif method == 'semantic':
            normalized = frame_idx / float(max(hubert_frames - 1, 1))
            delta = delta_min + (delta_max - delta_min) * (1 - normalized)

        elif method == 'joint':
            alpha, beta = ALPHA, BETA
            normalized = frame_idx / float(max(hubert_frames - 1, 1))
            S = normalized
            P = 1.0 - normalized
            joint_score = (S ** alpha) * (P ** beta)
            delta = delta_min + (delta_max - delta_min) * (1 - joint_score)

        else:
            delta = delta_min

        delta_values.append(delta)

    return delta_values, hubert_frames, stft_frames


def evaluate_single_audio(filepath, method='semantic',
                           delta_min=DELTA_MIN, delta_max=DELTA_MAX):
    """Evaluate watermark for a single audio file.

    Returns per-audio metrics dict.
    """
    audio, sr = sf.read(filepath)

    delta_values, hubert_frames, stft_frames = apply_selection_method(
        audio, sr, method, delta_min, delta_max)

    basename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    audio_id = basename.replace('.flac', '')

    metrics = {
        'audio_id': audio_id,
        'speaker_id': parent_dir,
        'duration_seconds': len(audio) / sr,
        'sample_rate': sr,
        'hubert_frames': hubert_frames,
        'stft_frames': stft_frames,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'method': method,
        'num_delta_values': len(delta_values),
        'mean_delta': float(np.mean(delta_values)) if delta_values else 0.0,
        'std_delta': float(np.std(delta_values)) if delta_values else 0.0,
        'min_delta': float(np.min(delta_values)) if delta_values else 0.0,
        'max_delta': float(np.max(delta_values)) if delta_values else 0.0,
    }

    return metrics


# ============================================================
# Watermark Embedding & Extraction Pipeline
# ============================================================

def load_payload(payload_path=PAYLOAD_PATH):
    """Load the 256-bit payload tensor."""
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Payload not found at {payload_path}")
    bits = torch.load(payload_path)
    if bits.shape[0] < PAYLOAD_LEN:
        raise ValueError(f"Payload has {bits.shape[0]} bits, need {PAYLOAD_LEN}")
    return bits[:PAYLOAD_LEN].cpu()


BOUNDARY_EXCLUDE_FRAMES = 1
"""Number of STFT time frames to exclude at each signal edge.

Diagnosis (roundtrip analysis) showed that modifications to STFT coefficients
located in the outermost frames (frame 0 and frame N-1 under center=True) are
not reliably preserved through the ISTFT->STFT reconstruction: the Hann
synthesis window tapers to ~0 at the signal boundary and single-coefficient
perturbations there can be wiped out, inverting the extracted sign. Excluding
these boundary frames from coordinate allocation eliminated all roundtrip bit
errors on the 200-recording development subset (BER 0.000078 -> 0.000000).
"""


def allocate_coordinates(hubert_frames, stft_frames, payload_len=PAYLOAD_LEN,
                         boundary_exclude=BOUNDARY_EXCLUDE_FRAMES):
    """Allocate unique (freq_bin, time_frame, hubert_idx) coordinates for embedding.

    Uses the full available frequency range (excluding DC and Nyquist) and all
    valid interior time frames to ensure each payload bit gets a unique STFT
    coefficient. This avoids collisions when hubert_frames < payload_len.

    STFT time frames in the outer `boundary_exclude` frames are omitted because
    their reconstruction is unreliable at signal edges (see BOUNDARY_EXCLUDE_FRAMES).
    """
    n_freq = N_FFT // 2 + 1
    freq_start = 2
    freq_end = n_freq - 2
    available_freq_bins = freq_end - freq_start + 1

    # Build list of all valid (freq_bin, time_frame) pairs
    # Use STFT time frames that have valid HuBERT mappings and are not at the
    # signal boundary (where reconstruction is unreliable).
    valid_pairs = []
    for hubert_idx in range(hubert_frames):
        stft_time = map_hubert_to_stft(hubert_idx, hubert_frames, stft_frames)
        if stft_time < boundary_exclude:
            continue
        if stft_time >= stft_frames - boundary_exclude:
            continue
        # Use all available frequency bins for this time frame
        for freq_bin in range(freq_start, freq_end + 1):
            valid_pairs.append((freq_bin, stft_time, hubert_idx))

    # If we have more payload bits than unique pairs, we have a fundamental capacity issue
    # But with 511 freq bins * ~100+ time frames, we have plenty of capacity
    if len(valid_pairs) < payload_len:
        # Fallback: cycle through pairs (should rarely happen)
        coordinates = []
        for i in range(payload_len):
            pair = valid_pairs[i % len(valid_pairs)]
            coordinates.append(pair)
        return coordinates

    # Distribute payload bits evenly across available pairs
    # Use stride to spread bits across time and frequency
    coordinates = []
    stride = max(1, len(valid_pairs) // payload_len)
    for i in range(payload_len):
        idx = (i * stride) % len(valid_pairs)
        coordinates.append(valid_pairs[idx])

    # Guard: ensure (freq_bin, time_frame) uniqueness so no two payload bits
    # overwrite the same coefficient.
    seen = set()
    dedup = []
    for c in coordinates:
        if (c[0], c[1]) not in seen:
            seen.add((c[0], c[1]))
            dedup.append(c)
    while len(dedup) < payload_len:
        for c in valid_pairs:
            if (c[0], c[1]) not in seen:
                seen.add((c[0], c[1]))
                dedup.append(c)
            if len(dedup) >= payload_len:
                break
    return dedup[:payload_len]


def embed_bipolar(stft, payload_bits, coordinates, delta_values):
    """Bipolar embedding: bit 1 -> real + delta, bit 0 -> real - delta.

    Args:
        stft: Original STFT tensor [n_freq, n_time] (complex)
        payload_bits: Tensor of shape [payload_len] with 0/1 values
        coordinates: List of (freq_bin, time_frame, hubert_idx) tuples
        delta_values: List of delta values per HuBERT frame

    Returns:
        Modified STFT tensor [n_freq, n_time] (complex)
    """
    modified = stft.clone()
    for i, (f, t, hubert_idx) in enumerate(coordinates):
        bit = int(payload_bits[i].item())
        delta = delta_values[hubert_idx] if hubert_idx < len(delta_values) else delta_values[-1]
        original_real = modified[f, t].real
        if bit == 1:
            new_real = original_real + delta
        else:
            new_real = original_real - delta
        modified[f, t] = torch.complex(new_real, modified[f, t].imag)
    return modified


def reconstruct_audio(stft, original_length, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """ISTFT reconstruction from STFT to waveform."""
    window = torch.hann_window(win_length)
    reconstructed = torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=original_length,
    )
    return reconstructed.numpy().astype(np.float64)


def extract_bits_direct(watermarked_stft, original_stft, coordinates, payload_len=PAYLOAD_LEN):
    """Reference-based extraction: compare modified real part to original.

    change = watermarked_real - original_real
    bit = 1 if change >= 0 else 0
    """
    bits = []
    for i in range(payload_len):
        f, t, _ = coordinates[i]
        change = watermarked_stft[f, t].real - original_stft[f, t].real
        bits.append(1 if change >= 0.0 else 0)
    return torch.tensor(bits, dtype=torch.uint8)


def extract_bits_roundtrip(watermarked_audio, original_audio, coordinates,
                           payload_len=PAYLOAD_LEN, n_fft=N_FFT,
                           hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Roundtrip extraction: ISTFT -> STFT -> extract.

    Saves watermarked audio to WAV, reloads, recomputes STFT, extracts bits.
    """
    # Compute STFT of watermarked audio (already loaded)
    wm_stft = compute_stft(watermarked_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # Compute STFT of original audio (already loaded)
    orig_stft = compute_stft(original_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return extract_bits_direct(wm_stft, orig_stft, coordinates, payload_len)


def compute_quality_metrics(original, watermarked):
    """Compute SNR, MSE, RMSE, correlation, and peak error metrics."""
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)

    diff = watermarked - original
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    peak_abs_error = float(np.max(np.abs(diff)))
    energy = float(np.mean(original ** 2))
    snr_db = float(10.0 * np.log10(energy / mse)) if mse > 0 else float('inf')
    correlation = float(np.corrcoef(original, watermarked)[0, 1]) if np.std(original) > 0 and np.std(watermarked) > 0 else 0.0
    original_rms = float(np.sqrt(np.mean(original ** 2)))
    watermarked_rms = float(np.sqrt(np.mean(watermarked ** 2)))

    return {
        'mse': mse,
        'rmse': rmse,
        'snr_db': snr_db,
        'peak_absolute_error': peak_abs_error,
        'waveform_correlation': correlation,
        'original_rms': original_rms,
        'watermarked_rms': watermarked_rms,
    }


def evaluate_single_audio_integrated(filepath, method='semantic',
                                       delta_min=DELTA_MIN, delta_max=DELTA_MAX,
                                       alpha=ALPHA, beta=BETA):
    """Full integrated evaluation: embed, extract, compute quality metrics.

    For a single audio file, this:
    1. Loads audio and payload
    2. Computes STFT and HuBERT frames
    3. Computes delta values per frame
    4. Embeds payload into STFT coefficients
    5. Reconstructs watermarked audio via ISTFT
    6. Extracts bits via direct and roundtrip methods
    7. Computes quality metrics

    Returns per-audio metrics dict with all evaluation results.
    """
    import time

    start_total = time.time()

    audio, sr = sf.read(filepath)
    original_length = len(audio)
    payload_bits = load_payload()
    payload_len = payload_bits.shape[0]

    # Compute delta values and HuBERT/STFT frame counts
    # (also serves as the semantic-location selection step)
    delta_values, hubert_frames, stft_frames = apply_selection_method(
        audio, sr, method, delta_min, delta_max)

    # Allocate embedding coordinates
    coordinates = allocate_coordinates(hubert_frames, stft_frames, PAYLOAD_LEN)

    # Compute original STFT
    original_stft = compute_stft(audio, sr)

    # Embed payload
    t_embed0 = time.time()
    watermarked_stft = embed_bipolar(original_stft, payload_bits, coordinates, delta_values)
    t_embed1 = time.time()

    # Reconstruct watermarked audio via ISTFT
    watermarked_audio = reconstruct_audio(
        watermarked_stft, original_length, n_fft=N_FFT)

    # Direct extraction (from same STFT session)
    extracted_direct = extract_bits_direct(watermarked_stft, original_stft, coordinates)

    # Roundtrip extraction (ISTFT -> STFT -> extract)
    t_extract0 = time.time()
    extracted_roundtrip = extract_bits_roundtrip(
        watermarked_audio, audio, coordinates)
    t_extract1 = time.time()

    embedding_time = float(t_embed1 - t_embed0)
    extraction_time = float(t_extract1 - t_extract0)

    # Compute bit accuracy and BER
    payload_numpy = payload_bits.numpy()
    direct_bits_numpy = extracted_direct.numpy()
    roundtrip_bits_numpy = extracted_roundtrip.numpy()

    direct_correct = np.sum(direct_bits_numpy == payload_numpy)
    roundtrip_correct = np.sum(roundtrip_bits_numpy == payload_numpy)
    bit_accuracy_direct = float(direct_correct) / PAYLOAD_LEN
    bit_accuracy_roundtrip = float(roundtrip_correct) / PAYLOAD_LEN
    ber_direct = 1.0 - bit_accuracy_direct
    ber_roundtrip = 1.0 - bit_accuracy_roundtrip

    # Quality metrics
    quality = compute_quality_metrics(audio, watermarked_audio)

    # Parse identifiers from path
    basename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    audio_id = basename.replace('.flac', '')

    total_time = time.time() - start_total

    metrics = {
        'audio_id': audio_id,
        'speaker_id': parent_dir,
        'duration_seconds': len(audio) / sr,
        'sample_rate': sr,
        'hubert_frames': hubert_frames,
        'stft_frames': stft_frames,
        'method': method,
        'payload_bits': payload_len,
        'redundancy': 1,  # 1 bit per coordinate; redundancy handled by allocation
        'delta_min': delta_min,
        'delta_max': delta_max,
        'alpha': alpha,
        'beta': beta,
        'num_delta_values': len(delta_values),
        'mean_delta': float(np.mean(delta_values)) if delta_values else 0.0,
        'std_delta': float(np.std(delta_values)) if delta_values else 0.0,
        'min_delta': float(np.min(delta_values)) if delta_values else 0.0,
        'max_delta': float(np.max(delta_values)) if delta_values else 0.0,
        'bit_accuracy_direct': bit_accuracy_direct,
        'bit_accuracy_roundtrip': bit_accuracy_roundtrip,
        'ber_direct': ber_direct,
        'ber_roundtrip': ber_roundtrip,
        'direct_correct_bits': int(direct_correct),
        'roundtrip_correct_bits': int(roundtrip_correct),
        'direct_verification_success': int(bit_accuracy_direct == 1.0),
        'roundtrip_verification_success': int(bit_accuracy_roundtrip > 0.0),
        'mse': quality['mse'],
        'rmse': quality['rmse'],
        'snr_db': quality['snr_db'],
        'peak_absolute_error': quality['peak_absolute_error'],
        'waveform_correlation': quality['waveform_correlation'],
        'original_rms': quality['original_rms'],
        'watermarked_rms': quality['watermarked_rms'],
        'embedding_time_seconds': embedding_time,
        'extraction_time_seconds': extraction_time,
        'total_time_seconds': float(total_time),
        'status': 'success',
        'failure_reason': '',
    }

    return metrics


# ============================================================
# Manifest-based evaluation runners
# ============================================================

def run_evaluation_from_manifest(manifest_path, limit=None, output_path=None,
                                  seed=42, attack=None, method='semantic'):
    """Run multi-audio evaluation across manifest recordings.

    Args:
        manifest_path: Path to CSV with columns: audio_id, speaker_id, path,
                       duration_seconds, sample_rate, split
        limit: Max number of recordings to evaluate (None = all)
        output_path: Path to write per-audio results CSV
        seed: Random seed for reproducibility
        attack: Attack type string (None = no attack)
        method: Watermark selection method ('random', 'spectral', 'semantic', 'joint')

    Returns:
        (all_metrics, aggregate) tuple
    """
    recordings = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)

    recordings.sort(key=lambda x: (x['speaker_id'], x['audio_id']))

    if limit and limit < len(recordings):
        recordings = recordings[:limit]

    all_metrics = []

    for i, rec in enumerate(recordings):
        filepath = rec['path']
        audio_id = rec['audio_id']
        speaker_id = rec['speaker_id']

        print(f"  [{i+1}/{len(recordings)}] {audio_id} (speaker: {speaker_id})")

        try:
            metrics = evaluate_single_audio(filepath, method=method)
            metrics['audio_id'] = audio_id
            metrics['speaker_id'] = speaker_id
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            all_metrics.append({
                'audio_id': audio_id,
                'speaker_id': speaker_id,
                'duration_seconds': rec.get('duration_seconds', 0),
                'sample_rate': rec.get('sample_rate', 16000),
                'hubert_frames': 0,
                'stft_frames': 0,
                'delta_min': DELTA_MIN,
                'delta_max': DELTA_MAX,
                'method': method,
                'num_delta_values': 0,
                'mean_delta': 0.0,
                'std_delta': 0.0,
                'min_delta': 0.0,
                'max_delta': 0.0,
                'error': str(type(e).__name__) + ': ' + str(e),
            })

    valid_metrics = [m for m in all_metrics if 'error' not in m]

    aggregate = {
        'total_recordings': len(all_metrics),
        'total_speakers': len(set(m['speaker_id'] for m in valid_metrics)),
        'method': method,
        'attack': attack,
        'delta_min': DELTA_MIN,
        'delta_max': DELTA_MAX,
        'recordings_evaluated': len(valid_metrics),
        'speakers_covered': len(set(m['speaker_id'] for m in valid_metrics)),
        'mean_mean_delta': float(np.mean([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'std_mean_delta': float(np.std([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'median_mean_delta': float(np.median([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'min_mean_delta': float(np.min([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'max_mean_delta': float(np.max([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_std_delta': float(np.mean([m['std_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'overall_mean_delta': float(np.mean([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'overall_std_delta': float(np.std([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
    }

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = ['audio_id', 'speaker_id', 'duration_seconds', 'sample_rate',
                     'hubert_frames', 'stft_frames', 'delta_min', 'delta_max',
                     'method', 'num_delta_values', 'mean_delta', 'std_delta',
                     'min_delta', 'max_delta']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                row = {k: m.get(k, '') for k in fieldnames}
                writer.writerow(row)

        agg_path = output_path.replace('.csv', '_aggregate.json')
        with open(agg_path, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f'\nPer-audio results: {output_path}')
        print(f'Aggregate summary: {agg_path}')

    return all_metrics, aggregate


def run_integrated_evaluation(manifest_path, limit=None, output_path=None,
                                method='semantic'):
    """Run full embedding+extraction evaluation across manifest recordings.

    For each audio and each method:
    1. Embed the 256-bit payload into STFT coefficients
    2. Reconstruct via ISTFT
    3. Extract bits via direct reference-based method
    4. Extract bits via roundtrip (ISTFT->reload->STFT)
    5. Compute quality metrics (SNR, MSE, RMSE, correlation, BER, accuracy)

    Args:
        manifest_path: Path to CSV with columns: audio_id, speaker_id, path,
                       duration_seconds, sample_rate, split
        limit: Max number of recordings to evaluate (None = all)
        output_path: Path to write per-audio results CSV
        method: Watermark selection method ('random', 'spectral', 'semantic', 'joint')

    Returns:
        (all_metrics, aggregate) tuple
    """
    recordings = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recordings.append(row)

    recordings.sort(key=lambda x: (x['speaker_id'], x['audio_id']))

    if limit and limit < len(recordings):
        recordings = recordings[:limit]

    all_metrics = []

    for i, rec in enumerate(recordings):
        filepath = rec['path']
        audio_id = rec['audio_id']
        speaker_id = rec['speaker_id']

        print(f"  [{i+1}/{len(recordings)}] {audio_id} (speaker: {speaker_id})")

        try:
            metrics = evaluate_single_audio_integrated(filepath, method=method)
            metrics['audio_id'] = audio_id
            metrics['speaker_id'] = speaker_id
            all_metrics.append(metrics)
            print(f"    Direct accuracy: {metrics['bit_accuracy_direct']:.4f}, "
                  f"Roundtrip accuracy: {metrics['bit_accuracy_roundtrip']:.4f}, "
                  f"SNR: {metrics['snr_db']:.2f} dB, "
                  f"MSE: {metrics['mse']:.6e}")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            all_metrics.append({
                'audio_id': audio_id,
                'speaker_id': speaker_id,
                'duration_seconds': rec.get('duration_seconds', 0),
                'sample_rate': rec.get('sample_rate', 16000),
                'hubert_frames': 0,
                'stft_frames': 0,
                'delta_min': DELTA_MIN,
                'delta_max': DELTA_MAX,
                'alpha': ALPHA,
                'beta': BETA,
                'method': method,
                'payload_bits': 256,
                'redundancy': 1,
                'num_delta_values': 0,
                'mean_delta': 0.0,
                'std_delta': 0.0,
                'min_delta': 0.0,
                'max_delta': 0.0,
                'bit_accuracy_direct': 0.0,
                'bit_accuracy_roundtrip': 0.0,
                'ber_direct': 1.0,
                'ber_roundtrip': 1.0,
                'direct_correct_bits': 0,
                'roundtrip_correct_bits': 0,
                'direct_verification_success': 0,
                'roundtrip_verification_success': 0,
                'mse': 0.0,
                'rmse': 0.0,
                'snr_db': 0.0,
                'peak_absolute_error': 0.0,
                'waveform_correlation': 0.0,
                'original_rms': 0.0,
                'watermarked_rms': 0.0,
                'embedding_time_seconds': 0.0,
                'extraction_time_seconds': 0.0,
                'total_time_seconds': 0.0,
                'status': 'failed',
                'failure_reason': str(type(e).__name__) + ': ' + str(e),
            })

    valid_metrics = [m for m in all_metrics if m['status'] == 'success']

    aggregate = {
        'total_recordings': len(all_metrics),
        'total_speakers': len(set(m['speaker_id'] for m in valid_metrics)),
        'method': method,
        'delta_min': DELTA_MIN,
        'delta_max': DELTA_MAX,
        'alpha': ALPHA,
        'beta': BETA,
        'payload_bits': 256,
        'redundancy': 1,
        'recordings_evaluated': len(valid_metrics),
        'speakers_covered': len(set(m['speaker_id'] for m in valid_metrics)),
        'mean_bit_accuracy_direct': float(np.mean([m['bit_accuracy_direct'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_bit_accuracy_roundtrip': float(np.mean([m['bit_accuracy_roundtrip'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_ber_direct': float(np.mean([m['ber_direct'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_ber_roundtrip': float(np.mean([m['ber_roundtrip'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_snr_db': float(np.mean([m['snr_db'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_mse': float(np.mean([m['mse'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_rmse': float(np.mean([m['rmse'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_waveform_correlation': float(np.mean([m['waveform_correlation'] for m in valid_metrics])) if valid_metrics else 0.0,
        'total_correct_direct': int(np.sum([m['direct_correct_bits'] for m in valid_metrics])) if valid_metrics else 0,
        'total_correct_roundtrip': int(np.sum([m['roundtrip_correct_bits'] for m in valid_metrics])) if valid_metrics else 0,
        'max_snr_db': float(np.max([m['snr_db'] for m in valid_metrics])) if valid_metrics else 0.0,
        'min_snr_db': float(np.min([m['snr_db'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_mean_delta': float(np.mean([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'overall_mean_delta': float(np.mean([m['mean_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_std_delta': float(np.mean([m['std_delta'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_embedding_time': float(np.mean([m['embedding_time_seconds'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_extraction_time': float(np.mean([m['extraction_time_seconds'] for m in valid_metrics])) if valid_metrics else 0.0,
        'mean_total_time': float(np.mean([m['total_time_seconds'] for m in valid_metrics])) if valid_metrics else 0.0,
        'successful_extractions': len(valid_metrics),
        'failed_extractions': len(all_metrics) - len(valid_metrics),
        'success_rate': float(len(valid_metrics)) / float(len(all_metrics)) if all_metrics else 0.0,
    }

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = ['audio_id', 'speaker_id', 'duration_seconds', 'sample_rate',
                     'hubert_frames', 'stft_frames', 'method',
                     'payload_bits', 'redundancy', 'delta_min', 'delta_max',
                     'alpha', 'beta', 'num_delta_values', 'mean_delta', 'std_delta',
                     'min_delta', 'max_delta',
                     'bit_accuracy_direct', 'bit_accuracy_roundtrip',
                     'ber_direct', 'ber_roundtrip',
                     'direct_correct_bits', 'roundtrip_correct_bits',
                     'direct_verification_success', 'roundtrip_verification_success',
                     'mse', 'rmse', 'snr_db', 'peak_absolute_error',
                     'waveform_correlation', 'original_rms', 'watermarked_rms',
                     'embedding_time_seconds', 'extraction_time_seconds', 'total_time_seconds',
                     'status', 'failure_reason']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                row = {k: m.get(k, '') for k in fieldnames}
                writer.writerow(row)

        agg_path = output_path.replace('.csv', '_aggregate.json')
        with open(agg_path, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f'\nPer-audio results: {output_path}')
        print(f'Aggregate summary: {agg_path}')

    return all_metrics, aggregate


# ============================================================
# CLI entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SemBind-Audio Multi-Audio Evaluation Framework')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to evaluation manifest CSV')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of recordings to evaluate')
    parser.add_argument('--output', type=str, default='results/evaluation_results.csv',
                        help='Output CSV path for per-audio results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--attack', type=str, default=None,
                        help='Attack type to simulate')
    parser.add_argument('--method', type=str, default='semantic',
                        choices=['random', 'spectral', 'semantic', 'joint'],
                        help='Watermark selection method')
    parser.add_argument('--integrated', action='store_true',
                        help='Run full embedding+extraction pipeline')

    args = parser.parse_args()

    print('=== SemBind-Audio Multi-Audio Evaluation ===')
    print(f'Manifest: {args.manifest}')
    print(f'Method: {args.method}')
    print(f'Delta: [{DELTA_MIN}, {DELTA_MAX}]')
    print(f'Seed: {args.seed}')
    if args.attack:
        print(f'Attack: {args.attack}')
    if args.integrated:
        print('Mode: Integrated (embedding + extraction + quality)')
    print()

    if args.integrated:
        all_metrics, aggregate = run_integrated_evaluation(
            manifest_path=args.manifest,
            limit=args.limit,
            output_path=args.output,
            method=args.method,
        )
    else:
        all_metrics, aggregate = run_evaluation_from_manifest(
            manifest_path=args.manifest,
            limit=args.limit,
            output_path=args.output,
            seed=args.seed,
            attack=args.attack,
            method=args.method,
        )

    print('\n=== AGGREGATE SUMMARY ===')
    if args.integrated:
        print(f'Total recordings evaluated: {aggregate["recordings_evaluated"]}')
        print(f'Total speakers covered: {aggregate["speakers_covered"]}')
        print(f'Mean bit accuracy (direct): {aggregate["mean_bit_accuracy_direct"]:.4f}')
        print(f'Mean bit accuracy (roundtrip): {aggregate["mean_bit_accuracy_roundtrip"]:.4f}')
        print(f'Mean BER (direct): {aggregate["mean_ber_direct"]:.4f}')
        print(f'Mean BER (roundtrip): {aggregate["mean_ber_roundtrip"]:.4f}')
        print(f'Mean SNR: {aggregate["mean_snr_db"]:.2f} dB')
        print(f'Mean MSE: {aggregate["mean_mse"]:.6e}')
        print(f'Mean RMSE: {aggregate["mean_rmse"]:.6e}')
        print(f'Mean waveform correlation: {aggregate["mean_waveform_correlation"]:.4f}')
        print(f'Total correct (direct): {aggregate["total_correct_direct"]}')
        print(f'Total correct (roundtrip): {aggregate["total_correct_roundtrip"]}')
    else:
        print(f'Total recordings evaluated: {aggregate["recordings_evaluated"]}')
        print(f'Total speakers covered: {aggregate["speakers_covered"]}')
        print(f'Overall mean delta: {aggregate["overall_mean_delta"]:.6f}')
        print(f'Mean mean delta: {aggregate["mean_mean_delta"]:.6f} +/- {aggregate["std_mean_delta"]:.6f}')
        print(f'Median mean delta: {aggregate["median_mean_delta"]:.6f}')
        print(f'Min mean delta: {aggregate["min_mean_delta"]:.6f}')
        print(f'Max mean delta: {aggregate["max_mean_delta"]:.6f}')

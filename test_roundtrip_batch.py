import torch
import numpy as np
import soundfile as sf
import os
import csv

# Load the dev subset manifest
manifest_path = 'results/evaluation_dev_subset.csv'
recordings = []
with open(manifest_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        recordings.append(row)

recordings.sort(key=lambda x: (x['speaker_id'], x['audio_id']))

# Configuration
DELTA_MIN = 0.002
DELTA_MAX = 0.008
PAYLOAD_LEN = 256

from evaluation_framework import apply_selection_method, allocate_coordinates, reconstruct_audio, extract_bits_direct

results_direct = []
results_roundtrip = []

for rec in recordings[:20]:  # Test first 20
    filepath = rec['path']
    audio, sr = sf.read(filepath)
    audio_np = audio.astype(np.float32)
    
    # Compute STFT
    stft = torch.stft(torch.from_numpy(audio_np), n_fft=1024, hop_length=320, win_length=1024, center=True, return_complex=True)
    
    # Delta values
    delta_values, hubert_frames, stft_frames = apply_selection_method(audio_np, sr, 'joint', DELTA_MIN, DELTA_MAX)
    
    # Coordinates
    coordinates = allocate_coordinates(hubert_frames, stft_frames, PAYLOAD_LEN)
    
    # Payload
    payload = torch.randint(0, 2, (PAYLOAD_LEN,)).to(torch.int64)
    
    # Embed
    def embed_bipolar(stft, payload_bits, coordinates, delta_values):
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
    
    watermarked_stft = embed_bipolar(stft, payload, coordinates, delta_values)
    
    # Direct extraction
    extracted_direct = extract_bits_direct(watermarked_stft, stft, coordinates)
    direct_accuracy = (extracted_direct == payload).float().mean().item()
    
    # Roundtrip
    watermarked_audio_np = reconstruct_audio(watermarked_stft, len(audio_np))
    wm_stft = torch.stft(torch.from_numpy(watermarked_audio_np), n_fft=1024, hop_length=320, win_length=1024, center=True, return_complex=True)
    orig_stft = torch.stft(torch.from_numpy(audio_np), n_fft=1024, hop_length=320, win_length=1024, center=True, return_complex=True)
    extracted_roundtrip = extract_bits_direct(wm_stft, orig_stft, coordinates)
    roundtrip_accuracy = (extracted_roundtrip == payload).float().mean().item()
    
    results_direct.append(direct_accuracy)
    results_roundtrip.append(roundtrip_accuracy)
    
    print(f"{rec['audio_id']}: direct={direct_accuracy:.4f}, roundtrip={roundtrip_accuracy:.4f}, diff={direct_accuracy-roundtrip_accuracy:.4f}")

# Summary
print(f'\n=== Summary (first 20 recordings) ===')
print(f'Direct mean accuracy: {np.mean(results_direct):.4f}')
print(f'Direct std accuracy: {np.std(results_direct):.4f}')
print(f'Roundtrip mean accuracy: {np.mean(results_roundtrip):.4f}')
print(f'Roundtrip std accuracy: {np.std(results_roundtrip):.4f}')
print(f'Mean accuracy drop: {np.mean([d-r for d,r in zip(results_direct, results_roundtrip)]):.4f}')
print(f'Records with 100% direct: {sum(1 for d in results_direct if d == 1.0)}/{len(results_direct)}')
print(f'Records with 100% roundtrip: {sum(1 for r in results_roundtrip if r == 1.0)}/{len(results_roundtrip)}')
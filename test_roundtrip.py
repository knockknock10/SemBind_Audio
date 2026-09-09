import torch
import numpy as np
import soundfile as sf

# Load a single audio from the dev subset
audio, sr = sf.read('data/raw/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac')
audio_np = audio.astype(np.float32)
audio_tensor = torch.from_numpy(audio_np)
original_length = len(audio_np)

# Compute original STFT
stft = torch.stft(torch.from_numpy(audio_np), n_fft=1024, hop_length=320, win_length=1024, center=True, return_complex=True)

# Embed all 256 bits using the framework's method
from evaluation_framework import apply_selection_method, allocate_coordinates

delta_values, hubert_frames, stft_frames = apply_selection_method(audio_np, sr, 'joint', 0.002, 0.008)
coordinates = allocate_coordinates(hubert_frames, stft_frames, 256)

# Embed payload
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

payload = torch.randint(0, 2, (256,)).to(torch.int64)
watermarked_stft = embed_bipolar(stft, payload, coordinates, delta_values)

# Direct extraction
from evaluation_framework import extract_bits_direct
extracted_direct = extract_bits_direct(watermarked_stft, stft, coordinates)
direct_accuracy = (extracted_direct == payload).float().mean().item()
direct_ber = 1 - direct_accuracy

print(f'Direct accuracy: {direct_accuracy:.4f}')
print(f'Direct BER: {direct_ber:.4f}')

# Roundtrip
from evaluation_framework import reconstruct_audio
watermarked_audio_np = reconstruct_audio(watermarked_stft, original_length)
watermarked_audio_tensor = torch.from_numpy(watermarked_audio_np)

# Compute STFT of watermarked and original for roundtrip
def compute_stft_tensor(audio, n_fft=1024, hop_length=320, win_length=1024):
    return torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True, return_complex=True)

wm_stft = compute_stft_tensor(watermarked_audio_tensor)
orig_stft = compute_stft_tensor(audio_tensor)

from evaluation_framework import extract_bits_direct
extracted_roundtrip = extract_bits_direct(wm_stft, orig_stft, coordinates)
roundtrip_accuracy = (extracted_roundtrip == payload).float().mean().item()
roundtrip_ber = 1 - roundtrip_accuracy

print(f'Roundtrip accuracy: {roundtrip_accuracy:.4f}')
print(f'Roundtrip BER: {roundtrip_ber:.4f}')
print(f'Accuracy drop: {direct_accuracy - roundtrip_accuracy:.4f}')
print(f'Direct correct: {torch.sum(extracted_direct == payload).item()}/256')
print(f'Roundtrip correct: {torch.sum(extracted_roundtrip == payload).item()}/256')

# Check individual bit differences
diff_mask = extracted_direct != extracted_roundtrip
print(f'Bits that differ direct vs roundtrip: {torch.sum(diff_mask).item()}')
if torch.sum(diff_mask).item() > 0:
    diff_indices = torch.where(diff_mask)[0]
    print(f'First few differing bit indices: {diff_indices[:10].tolist()}')
    for idx in diff_indices[:5]:
        print(f'  Bit {idx}: direct={extracted_direct[idx].item()}, roundtrip={extracted_roundtrip[idx].item()}, payload={payload[idx].item()}')
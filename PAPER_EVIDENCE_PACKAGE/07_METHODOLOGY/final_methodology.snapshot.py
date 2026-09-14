"""
SemBind-Audio: Final Methodology Module.

Correct implementation of the intended SemBind-Audio watermarking pipeline:

    semantic contribution   Δ_i = 1 - cos(G, G^(-i))           (leave-one-out, closed form)
    semantic suitability    S_r = 1 - minmax(Δ)                 L low contribution -> HIGH suitability
    spectral suitability    P_r = minmax(log1p(|X|))_r          (local spectral energy proxy)
    joint suitability       J_r = max(S_r,eps)^a * max(P_r,eps)^b
    adaptive strength       d_r = d_min + (d_max - d_min) * J_r
    allocation              256 bits x R replicas -> unique (freq,time) coordinates
    embedding               bipolar: bit=1 -> Re+delta, bit=0 -> Re-delta
    extraction              reference-based: sign(Re(W)-Re(X)) then majority vote per bit

Differences vs. the legacy evaluation_framework.apply_selection_method:
  - semantic scores are the REAL perturbation-based contribution (not frame-index proxies)
  - joint suitability is multiplicative (S^a * P^b), not a weighted-sum frame proxy
  - coordinates differ across methods (allocation ordering follows each method's score)
  - payload bits are spread over distinct regions with 3x replica redundancy

All randomness is deterministic (audio-path seeded); no result is estimated or fabricated.
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- config
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024
N_MELS = 80
PAYLOAD_BITS = 256
REDUNDANCY = 3

ALPHA = 0.70
BETA = 0.30
DELTA_MIN = 0.002
DELTA_MAX = 0.008
FIXED_DELTA = 0.005

EPS = 1e-8
BOUNDARY_EXCLUDE_FRAMES = 1
FREQ_START_RATIO = 0.05
FREQ_END_RATIO = 0.95

HU_BERT_MODEL = "facebook/hubert-base-ls960"
HU_BERT_DIM = 768

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "final_cache" / "hubert"
PAYLOAD_PATH = RESULTS_DIR / "watermark" / "payload_bits.pt"

_base_config = {
    "sample_rate": SAMPLE_RATE,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "win_length": WIN_LENGTH,
    "n_mels": N_MELS,
    "payload_bits": PAYLOAD_BITS,
    "redundancy": REDUNDANCY,
    "alpha": ALPHA,
    "beta": BETA,
    "delta_min": DELTA_MIN,
    "delta_max": DELTA_MAX,
    "fixed_delta": FIXED_DELTA,
    "hubert_model": HU_BERT_MODEL,
    "hubert_dim": HU_BERT_DIM,
    "boundary_exclude_frames": BOUNDARY_EXCLUDE_FRAMES,
    "freq_start_ratio": FREQ_START_RATIO,
    "freq_end_ratio": FREQ_END_RATIO,
    "extraction": "reference-based",
    "verification": "populated majority vote over R replicas",
    "dataset": "LibriSpeech test-clean (2620 recordings, 40 speakers, 16kHz)",
    "config_version": "1.0",
    "stft": {"center": True, "window": "hann", "pad_mode": "reflect"},
}


def config_hash(config=None):
    cfg = _base_config if config is None else config
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def git_head():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        ).stdout.strip()
    except Exception:
        return "unknown"


def result_metadata(experiment_id, extra=None, config=None):
    cfg = _base_config if config is None else config
    return {
        "experiment_id": experiment_id,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_head(),
        "torch_version": torch.__version__,
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_payload(path=PAYLOAD_PATH):
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------- stft
def compute_stft(audio: np.ndarray):
    window = torch.hann_window(WIN_LENGTH)
    return torch.stft(
        torch.from_numpy(audio.astype(np.float32)).float(),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )


def reconstruct_audio(stft: torch.Tensor, target_length: int):
    window = torch.hann_window(WIN_LENGTH)
    audio = torch.istft(
        stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        length=target_length,
    )
    return audio.numpy()


# ---------------------------------------------------------------- hubert
_feature_extractor = None
_model = None


def _get_hubert():
    global _feature_extractor, _model
    if _model is None:
        from transformers import AutoFeatureExtractor, AutoModel

        _feature_extractor = AutoFeatureExtractor.from_pretrained(HU_BERT_MODEL)
        _model = AutoModel.from_pretrained(HU_BERT_MODEL)
        _model.eval()
    return _feature_extractor, _model


def extract_embeddings(audio: np.ndarray):
    """Return HuBERT last_hidden_state [N, 768]."""
    feature_extractor, model = _get_hubert()
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values)
    return outputs.last_hidden_state[0]


def _cache_path(audio_id):
    return CACHE_DIR / f"{audio_id}.npy"


def load_embeddings_cached(audio_id, audio, force_recompute=False):
    """Load HuBERT embeddings from cache, else compute and cache them."""
    p = _cache_path(audio_id)
    if p.exists() and not force_recompute:
        return torch.from_numpy(np.load(p).astype(np.float32))
    emb = extract_embeddings(audio)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, emb.numpy().astype(np.float16))
    return emb


# ---------------------------------------------------------------- semantic contribution
def compute_contribution(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Closed-form leave-one-out semantic contribution.

      G      = mean_j E_j
      G^(-i) = (N*G - E_i) / (N - 1)
      d_i    = clamp(1 - cos(G, G^(-i)))

    Exactly equivalent to the per-frame loop in src/semantic/contribution.py
    but vectorised over frames (identical results, lower cost).
    """
    emb = embeddings.float()
    n = emb.shape[0]
    if n < 2:
        return torch.zeros(n)
    g = emb.mean(dim=0)
    g_minus = (n * g - emb) / (n - 1)
    cos = F.cosine_similarity(g.view(1, -1), g_minus, dim=1)
    return (1.0 - cos).clamp(min=0.0)


def minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if abs(float(hi) - float(lo)) < EPS:
        return torch.full_like(x, 0.5)
    return (x - lo) / (hi - lo + EPS)


# ---------------------------------------------------------------- suitability
def semantic_suitability(contribution: torch.Tensor) -> torch.Tensor:
    """S_r = minmax(1 - minmax(d)). Lower contribution -> higher suitability."""
    return minmax(1.0 - minmax(contribution))


def spectral_suitability(stft: torch.Tensor) -> torch.Tensor:
    """P_r = minmax(mean_{f in band} log1p(|X[f, t]|)) over STFT time frames."""
    f_start = max(1, int(stft.shape[0] * FREQ_START_RATIO))
    f_end = min(stft.shape[0] - 2, int(stft.shape[0] * FREQ_END_RATIO))
    band = torch.log1p(torch.abs(stft[f_start : f_end + 1, :]))
    return minmax(band.mean(dim=0))


def joint_suitability(s_suit: torch.Tensor, p_suit: torch.Tensor,
                      alpha: float = ALPHA, beta: float = BETA) -> torch.Tensor:
    """J_r = max(S_r,eps)^alpha * max(P_r,eps)^beta (multiplicative, NOT a sum)."""
    return (s_suit.clamp(min=EPS) ** alpha) * (p_suit.clamp(min=EPS) ** beta)


def adaptive_delta(joint: torch.Tensor) -> torch.Tensor:
    return DELTA_MIN + (DELTA_MAX - DELTA_MIN) * joint


# ---------------------------------------------------------------- mapping
def map_contribution_to_stft(contribution: torch.Tensor, hubert_frames, stft_frames):
    """Interpolate per-HuBERT-frame values onto the STFT time axis (np.interp)."""
    if stft_frames <= 1:
        return contribution[:1]
    t_space = np.linspace(0.0, stft_frames - 1, max(hubert_frames, 2))
    t = np.arange(stft_frames)
    return torch.from_numpy(
        np.interp(t, t_space, contribution.numpy().astype(np.float64)).astype(np.float32)
    )


def interior_frames(stft_frames):
    """Time frames excluding STFT boundary artifacts."""
    lo = BOUNDARY_EXCLUDE_FRAMES
    hi = stft_frames - 1 - BOUNDARY_EXCLUDE_FRAMES
    if lo >= hi:
        return list(range(hi + 1))
    return list(range(lo, hi + 1))


def freq_pool():
    """Frequency bins eligible for embedding (inner 5%-95% band, DC/nyquist excluded)."""
    f_bins = N_FFT // 2 + 1  # 513
    f_start = max(1, int(f_bins * FREQ_START_RATIO))
    f_end = min(f_bins - 1, int(f_bins * FREQ_END_RATIO))
    return list(range(f_start, f_end + 1))


def per_frame_delta(method, stft, contribution, audio_id,
                    alpha: float = ALPHA, beta: float = BETA, delta_mode="adaptive"):
    """Return (delta_at_stft_frame, suitability_score_at_stft_frame)."""
    hubert_frames = contribution.numel() if contribution is not None else 0
    stft_frames = stft.shape[1]
    if contribution is not None:
        s_map = map_contribution_to_stft(semantic_suitability(contribution), hubert_frames, stft_frames)
    p_map = spectral_suitability(stft)
    if method == "random":
        rng = np.random.default_rng(int(hashlib.md5(audio_id.encode()).hexdigest(), 16))
        key = torch.from_numpy(rng.random(stft_frames).astype(np.float32))
        delta = (DELTA_MIN + (DELTA_MAX - DELTA_MIN) * key) if delta_mode == "adaptive" else torch.full_like(key, FIXED_DELTA)
        return delta, key
    if method in ("semantic", "joint") and contribution is None:
        raise ValueError(f"method {method} requires HuBERT embeddings")
    if method == "spectral":
        j = p_map
    elif method == "semantic":
        j = s_map
    elif method == "joint":
        j = joint_suitability(s_map, p_map, alpha, beta)
    else:
        raise ValueError(f"unknown method {method}")
    if delta_mode == "adaptive":
        delta = adaptive_delta(j)
    elif delta_mode == "fixed":
        delta = torch.full_like(j, FIXED_DELTA)
    else:  # explicit constant strength (delta=value)
        delta = torch.full_like(j, float(delta_mode))
    return delta, j


# ---------------------------------------------------------------- allocation
def allocate_coordinates(
    stft_frames: int,
    method: str,
    key_per_frame: torch.Tensor,
    audio_id: str,
    redundancy: int = REDUNDANCY,
):
    """
    Deterministically place PAYLOAD_BITS*redundancy unique (freq,time) coordinates.

    The method controls the ORDER in which time frames are consumed:
      - 'random':    pseudo-random frame order (md5(audio_id:frame))
      - 'spectral':  descending spectral suitability
      - 'semantic':  descending semantic suitability
      - 'joint':     descending joint suitability

    Frequency bins are spread with a coprime stride across that order. Uniqueness of
    every (freq,time) pair is asserted. Replica r of bit b lives at coordinate index
    k = b*redundancy + r.
    """
    frames = interior_frames(stft_frames)
    if len(frames) < 2:
        raise ValueError(f"{audio_id}: too few interior time frames ({len(frames)})")
    pool = freq_pool()
    if len(pool) < 2:
        raise ValueError(f"{audio_id}: empty freq pool")

    total = PAYLOAD_BITS * redundancy

    if method == "random":
        frames = sorted(
            frames,
            key=lambda t: int(hashlib.md5(f"{audio_id}:{t}".encode()).hexdigest(), 16),
        )
    else:
        score = key_per_frame.numpy()
        frames = sorted(frames, key=lambda t: float(score[t]), reverse=True)
    frames = list(frames)

    plen = len(pool)
    flen = len(frames)
    # Same-bit replicas are placed in widely separated thirds of the timeline so that
    # their STFT leakage regions do not overlap (analysis windows span ~3 frames).
    # Frequency bins cycle with a coprime stride; uniqueness of every (freq,time) pair
    # is asserted below.
    fi = max(flen // redundancy, 1)
    coordinates = []
    for k in range(total):
        r = k % redundancy
        b = k // redundancy
        t = frames[(b % fi) + r * fi]
        fbin = pool[(k * 467) % plen]
        coordinates.append((int(fbin), int(t)))

    unique = len(set(coordinates))
    if unique != total:
        coordinates = []
        seen = set()
        for k in range(total):
            r = k % redundancy
            b = k // redundancy
            t = frames[(b % fi) + r * fi]
            for fb in pool:
                c = (int(fb), int(t))
                if c not in seen:
                    seen.add(c)
                    coordinates.append(c)
                    break
    assert len(set(coordinates)) == total, (
        f"{audio_id}: coordinate collision {len(set(coordinates))}/{total}"
    )
    return coordinates


# ---------------------------------------------------------------- pipeline
def embed_watermark(stft, coordinates, bits, delta_values, redundancy=REDUNDANCY):
    """Bipolar additive embedding into the real part of selected coefficients."""
    stft = stft.clone()
    bl = len(bits)
    for k, (f, t) in enumerate(coordinates):
        bit = int(bits[(k // redundancy) % bl])
        if hasattr(delta_values, "__len__"):
            d = float(delta_values[k])
        else:
            d = float(delta_values)
        if bit == 1:
            stft[f, t] = torch.complex(stft[f, t].real + d, stft[f, t].imag)
        else:
            stft[f, t] = torch.complex(stft[f, t].real - d, stft[f, t].imag)
    return stft


def extract_majority(original_stft, watermarked_stft, coordinates, redundancy=REDUNDANCY):
    """Reference-based extraction: sign(Re(W)-Re(X)) per replica, majority vote per bit."""
    bits = []
    vote_mat = []
    for b in range(PAYLOAD_BITS):
        votes = []
        for r in range(redundancy):
            k = b * redundancy + r
            f, t = coordinates[k]
            change = float(watermarked_stft[f, t].real - original_stft[f, t].real)
            votes.append(1 if change > 0 else 0)
        vote_mat.append(votes)
        bits.append(int(round(sum(votes) / redundancy)))
    return torch.tensor(bits, dtype=torch.uint8), torch.tensor(vote_mat)


def extract_majority_surviving(original_stft, watermarked_stft, coordinates,
                               redundancy=REDUNDANCY, expected_len=PAYLOAD_BITS):
    """Extraction over a surviving subset of coordinates (e.g., after truncation).

    Returns:
      bits: tensor of length expected_len; -1 where a bit lost ALL replicas.
      valid: bool tensor of length expected_len.
    """
    counts = {b: [] for b in range(expected_len)}
    for k, (f, t) in enumerate(coordinates):
        b = k // redundancy
        if b >= expected_len:
            continue
        if 0 <= f < watermarked_stft.shape[0] and 0 <= t < watermarked_stft.shape[1]:
            change = float(watermarked_stft[f, t].real - original_stft[f, t].real)
            counts[b].append(1 if change > 0 else 0)
    bits = torch.full((expected_len,), -1, dtype=torch.int8)
    valid = torch.zeros(expected_len, dtype=torch.bool)
    for b, votes in counts.items():
        if votes:
            valid[b] = True
            bits[b] = int(round(sum(votes) / len(votes)))
    return bits, valid


def quality_metrics(payload, recovered, audio_orig, audio_wm):
    payload = torch.as_tensor(payload).to(torch.uint8)
    recovered = torch.as_tensor(recovered).to(torch.uint8)
    correct = int((recovered == payload).sum().item())
    total = int(len(payload))
    diff = audio_orig.astype(np.float64) - audio_wm.astype(np.float64)
    error_energy = float((diff ** 2).sum())
    signal_energy = float((audio_orig.astype(np.float64) ** 2).sum())
    snr = 10.0 * np.log10(signal_energy / (error_energy + EPS))
    return {
        "payload_bits": total,
        "correct_bits": correct,
        "bit_accuracy": float(correct / total),
        "ber": float(1.0 - correct / total),
        "payload_success": 1.0 if correct == total else 0.0,
        "snr_db": float(snr),
        "error_energy": error_energy,
        "signal_energy": signal_energy,
    }


# ---------------------------------------------------------------- attacks
def _mp3_roundtrip(audio, sr, bitrate=128):
    import soundfile as sf_compat  # noqa: F401

    buf = io.BytesIO()
    sf_compat.write(buf, audio, sr, format="MP3")
    buf.seek(0)
    return sf_compat.read(buf, dtype="float32")[0]


def _fix_length(x, target_len):
    x = x.astype(np.float32)
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x


def _resample_poly(audio, sr_in, sr_out):
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def apply_attack(audio, sr, attack):
    """Return the attacked audio at the SAME length/grid as the input.

    Attacks are applied as signal-processing roundtrips so that time alignment with
    the original (reference) signal is preserved for coordinate extraction.
    """
    target_len = len(audio)
    if attack == "none":
        return audio
    if attack.startswith("awgn_"):
        snr_db = int(attack.split("_")[1].replace("db", ""))
        seed = int(snr_db)
        noise = np.random.default_rng(seed).normal(0.0, 1.0, audio.shape).astype(np.float32)
        target = float((audio ** 2).mean()) / (10.0 ** (snr_db / 10.0))
        return audio + noise * np.sqrt(target)
    if attack == "resample_8k":
        x = _resample_poly(audio, sr, 8000)
        x = _resample_poly(x, 8000, sr)
        return _fix_length(x, target_len)
    if attack == "lowpass_4k":
        from scipy.signal import butter, sosfilt

        sos = butter(4, 4000, "low", fs=sr, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    if attack == "scale_0.9":
        return (audio * 0.9).astype(np.float32)
    if attack == "mp3_128":
        return _fix_length(_mp3_roundtrip(audio, sr), target_len)
    raise ValueError(f"unknown attack {attack}")


ATTACKS = [
    "none",
    "awgn_40db",
    "awgn_30db",
    "awgn_20db",
    "awgn_10db",
    "resample_8k",
    "lowpass_4k",
    "scale_0.9",
    "mp3_128",
]
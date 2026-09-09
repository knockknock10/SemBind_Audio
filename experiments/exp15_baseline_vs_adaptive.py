"""
SemBind-Audio
Experiment 15: Baseline vs Semantic-Adaptive Ablation

Purpose
-------
Compare the fixed-strength baseline watermark against the
semantic-adaptive watermark under the same experimental conditions.

Baseline:
    Fixed embedding strength alpha = 0.001

SemBind:
    Region-dependent semantic-adaptive embedding strength

The experiment evaluates:
    - MSE
    - RMSE
    - SNR
    - Peak absolute error
    - Waveform correlation
    - RMS
    - Peak amplitude
    - Bit accuracy
    - Bit error rate

The purpose is to determine whether semantic-adaptive embedding
provides measurable benefits over fixed-strength embedding.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASELINE_AUDIO = (
    PROJECT_ROOT
    / "results"
    / "watermark_embedding"
    / "watermarked_audio.wav"
)

ADAPTIVE_AUDIO = (
    PROJECT_ROOT
    / "results"
    / "adaptive_embedding"
    / "adaptive_watermarked_audio.wav"
)

BASELINE_EXTRACTION = (
    PROJECT_ROOT
    / "results"
    / "watermark_extraction"
    / "extraction_summary.json"
)

ADAPTIVE_EXTRACTION = (
    PROJECT_ROOT
    / "results"
    / "adaptive_extraction"
    / "extraction_summary.json"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ablation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_audio(path: Path):
    """
    Load audio as float64 mono waveform.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{path}"
        )

    audio, sample_rate = sf.read(
        path,
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    # Convert stereo to mono if necessary.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    return audio, sample_rate


def align_audio(
    original,
    watermarked,
):
    """
    Make two waveforms equal in length.
    """

    length = min(
        len(original),
        len(watermarked),
    )

    return (
        original[:length],
        watermarked[:length],
    )


def calculate_metrics(
    original,
    watermarked,
):
    """
    Calculate objective waveform-level quality metrics.
    """

    original, watermarked = align_audio(
        original,
        watermarked,
    )

    difference = (
        watermarked - original
    )

    mse = np.mean(
        difference ** 2
    )

    rmse = np.sqrt(mse)

    signal_power = np.mean(
        original ** 2
    )

    noise_power = np.mean(
        difference ** 2
    )

    if noise_power > 0:
        snr = 10.0 * np.log10(
            signal_power / noise_power
        )
    else:
        snr = float("inf")

    peak_error = np.max(
        np.abs(difference)
    )

    original_rms = np.sqrt(
        np.mean(
            original ** 2
        )
    )

    watermarked_rms = np.sqrt(
        np.mean(
            watermarked ** 2
        )
    )

    original_peak = np.max(
        np.abs(original)
    )

    watermarked_peak = np.max(
        np.abs(watermarked)
    )

    # Correlation.
    original_centered = (
        original - np.mean(original)
    )

    watermarked_centered = (
        watermarked - np.mean(watermarked)
    )

    denominator = (
        np.linalg.norm(original_centered)
        *
        np.linalg.norm(watermarked_centered)
    )

    if denominator > 0:
        correlation = (
            np.dot(
                original_centered,
                watermarked_centered,
            )
            / denominator
        )
    else:
        correlation = 0.0

    return {
        "samples": int(len(original)),
        "mse": float(mse),
        "rmse": float(rmse),
        "snr_db": float(snr),
        "peak_absolute_error": float(
            peak_error
        ),
        "original_rms": float(
            original_rms
        ),
        "watermarked_rms": float(
            watermarked_rms
        ),
        "rms_difference": float(
            abs(
                original_rms
                -
                watermarked_rms
            )
        ),
        "original_peak": float(
            original_peak
        ),
        "watermarked_peak": float(
            watermarked_peak
        ),
        "waveform_correlation": float(
            correlation
        ),
    }


def load_extraction_summary(
    path: Path,
):
    """
    Load extraction statistics produced by
    Exp10 or Exp14.
    """

    if not path.exists():
        return {
            "total_bits": None,
            "correct_bits": None,
            "incorrect_bits": None,
            "bit_accuracy": None,
            "bit_error_rate": None,
        }

    with open(
        path,
        "r",
    ) as f:
        data = json.load(f)

    return {
        "total_bits": data.get(
            "total_bits",
            data.get(
                "payload_length"
            ),
        ),
        "correct_bits": data.get(
            "correct_bits"
        ),
        "incorrect_bits": data.get(
            "incorrect_bits"
        ),
        "bit_accuracy": data.get(
            "bit_accuracy"
        ),
        "bit_error_rate": data.get(
            "bit_error_rate"
        ),
    }


def main():

    print("=" * 64)
    print(
        "SemBind-Audio : Experiment 15"
    )
    print(
        "Baseline vs Semantic-Adaptive Ablation"
    )
    print("=" * 64)

    # --------------------------------------------------------
    # Load original audio.
    #
    # The original audio is reconstructed from the same
    # preprocessing pipeline used by the experiments.
    # --------------------------------------------------------

    from src.dataset.audio_dataset import AudioDataset
    from src.preprocessing.audio_loader import preprocess

    dataset = AudioDataset()

    sample = dataset.load(0)

    original_audio = preprocess(
        sample["audio"],
        sample["sample_rate"],
    )

    original_audio = np.asarray(
        original_audio,
        dtype=np.float64,
    )

    original_sample_rate = int(
        sample["sample_rate"]
    )

    print()
    print(
        f"Original Sample Rate : "
        f"{original_sample_rate}"
    )

    print(
        f"Original Samples     : "
        f"{len(original_audio)}"
    )

    # --------------------------------------------------------
    # Load baseline.
    # --------------------------------------------------------

    baseline_audio, baseline_sr = load_audio(
        BASELINE_AUDIO
    )

    # --------------------------------------------------------
    # Load adaptive.
    # --------------------------------------------------------

    adaptive_audio, adaptive_sr = load_audio(
        ADAPTIVE_AUDIO
    )

    print()
    print(
        f"Baseline Sample Rate : "
        f"{baseline_sr}"
    )

    print(
        f"Adaptive Sample Rate : "
        f"{adaptive_sr}"
    )

    # --------------------------------------------------------
    # Validate sample rates.
    # --------------------------------------------------------

    if baseline_sr != original_sample_rate:
        raise ValueError(
            "Baseline sample rate does not "
            "match original audio."
        )

    if adaptive_sr != original_sample_rate:
        raise ValueError(
            "Adaptive sample rate does not "
            "match original audio."
        )

    # --------------------------------------------------------
    # Calculate metrics.
    # --------------------------------------------------------

    baseline_metrics = calculate_metrics(
        original_audio,
        baseline_audio,
    )

    adaptive_metrics = calculate_metrics(
        original_audio,
        adaptive_audio,
    )

    # --------------------------------------------------------
    # Load extraction results.
    # --------------------------------------------------------

    baseline_extraction = (
        load_extraction_summary(
            BASELINE_EXTRACTION
        )
    )

    adaptive_extraction = (
        load_extraction_summary(
            ADAPTIVE_EXTRACTION
        )
    )

    # --------------------------------------------------------
    # Print baseline.
    # --------------------------------------------------------

    print()
    print("-" * 64)
    print(
        "BASELINE : Fixed Strength"
    )
    print("-" * 64)

    print(
        f"MSE                  : "
        f"{baseline_metrics['mse']:.12e}"
    )

    print(
        f"RMSE                 : "
        f"{baseline_metrics['rmse']:.12e}"
    )

    print(
        f"SNR                  : "
        f"{baseline_metrics['snr_db']:.6f} dB"
    )

    print(
        f"Peak Absolute Error  : "
        f"{baseline_metrics['peak_absolute_error']:.12e}"
    )

    print(
        f"Waveform Correlation : "
        f"{baseline_metrics['waveform_correlation']:.12f}"
    )

    print(
        f"RMS Difference       : "
        f"{baseline_metrics['rms_difference']:.12e}"
    )

    print(
        f"Bit Accuracy         : "
        f"{baseline_extraction['bit_accuracy']}"
    )

    print(
        f"Bit Error Rate       : "
        f"{baseline_extraction['bit_error_rate']}"
    )

    # --------------------------------------------------------
    # Print adaptive.
    # --------------------------------------------------------

    print()
    print("-" * 64)
    print(
        "SEMBIND : Semantic-Adaptive"
    )
    print("-" * 64)

    print(
        f"MSE                  : "
        f"{adaptive_metrics['mse']:.12e}"
    )

    print(
        f"RMSE                 : "
        f"{adaptive_metrics['rmse']:.12e}"
    )

    print(
        f"SNR                  : "
        f"{adaptive_metrics['snr_db']:.6f} dB"
    )

    print(
        f"Peak Absolute Error  : "
        f"{adaptive_metrics['peak_absolute_error']:.12e}"
    )

    print(
        f"Waveform Correlation : "
        f"{adaptive_metrics['waveform_correlation']:.12f}"
    )

    print(
        f"RMS Difference       : "
        f"{adaptive_metrics['rms_difference']:.12e}"
    )

    print(
        f"Bit Accuracy         : "
        f"{adaptive_extraction['bit_accuracy']}"
    )

    print(
        f"Bit Error Rate       : "
        f"{adaptive_extraction['bit_error_rate']}"
    )

    snr_improvement = (
        adaptive_metrics["snr_db"]
        -
        baseline_metrics["snr_db"]
    )

    mse_reduction = (
        (
            baseline_metrics["mse"]
            -
            adaptive_metrics["mse"]
        )
        /
        baseline_metrics["mse"]
        *
        100.0
    )

    rmse_reduction = (
        (
            baseline_metrics["rmse"]
            -
            adaptive_metrics["rmse"]
        )
        /
        baseline_metrics["rmse"]
        *
        100.0
    )

    peak_error_reduction = (
        (
            baseline_metrics[
                "peak_absolute_error"
            ]
            -
            adaptive_metrics[
                "peak_absolute_error"
            ]
        )
        /
        baseline_metrics[
            "peak_absolute_error"
        ]
        *
        100.0
    )

    correlation_improvement = (
        adaptive_metrics[
            "waveform_correlation"
        ]
        -
        baseline_metrics[
            "waveform_correlation"
        ]
    )

    print()
    print("=" * 64)
    print(
        "ABLATION COMPARISON"
    )
    print("=" * 64)

    print(
        f"SNR Improvement          : "
        f"{snr_improvement:.6f} dB"
    )

    print(
        f"MSE Reduction            : "
        f"{mse_reduction:.6f}%"
    )

    print(
        f"RMSE Reduction           : "
        f"{rmse_reduction:.6f}%"
    )

    print(
        f"Peak Error Reduction    : "
        f"{peak_error_reduction:.6f}%"
    )

    print(
        f"Correlation Improvement : "
        f"{correlation_improvement:.12f}"
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "baseline_fixed",
                **baseline_metrics,
                **{
                    f"extraction_{k}": v
                    for k, v
                    in baseline_extraction.items()
                },
            },
            {
                "method": "sembind_adaptive",
                **adaptive_metrics,
                **{
                    f"extraction_{k}": v
                    for k, v
                    in adaptive_extraction.items()
                },
            },
        ]
    )

    comparison_path = (
        RESULT_DIR
        / "baseline_vs_adaptive.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    summary = {
        "experiment": 15,
        "description": (
            "Baseline fixed-strength versus "
            "semantic-adaptive watermarking"
        ),
        "baseline": {
            "embedding_strategy": (
                "fixed strength"
            ),
            "strength": 0.001,
            **baseline_metrics,
            "extraction": baseline_extraction,
        },
        "sembind": {
            "embedding_strategy": (
                "semantic-adaptive strength"
            ),
            **adaptive_metrics,
            "extraction": adaptive_extraction,
        },
        "improvement": {
            "snr_improvement_db": (
                float(snr_improvement)
            ),
            "mse_reduction_percent": (
                float(mse_reduction)
            ),
            "rmse_reduction_percent": (
                float(rmse_reduction)
            ),
            "peak_error_reduction_percent": (
                float(peak_error_reduction)
            ),
            "correlation_improvement": (
                float(correlation_improvement)
            ),
        },
    }

    summary_path = (
        RESULT_DIR
        / "baseline_vs_adaptive.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 64)
    print(
        "Experiment 15 Complete"
    )
    print("=" * 64)

    print()
    print(
        "Results Saved"
    )

    print(
        comparison_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()
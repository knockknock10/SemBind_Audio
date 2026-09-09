"""
SemBind-Audio

Watermark Embedding Module
"""

import torch


def embed_bit(
    coefficient: torch.Tensor,
    bit: int,
    delta: float,
) -> torch.Tensor:
    """
    Embed one binary bit into the real component of a complex STFT coefficient.

    The imaginary component remains unchanged.

    bit = 1: real(X') = real(X) + delta
    bit = 0: real(X') = real(X) - delta

    Parameters
    ----------
    coefficient : torch.Tensor
        Complex STFT coefficient.
    bit : int
        Binary bit to embed (0 or 1).
    delta : float
        Embedding strength.

    Returns
    -------
    torch.Tensor
        Modified complex STFT coefficient.
    """

    original_real = float(coefficient.real.item())
    original_imag = float(coefficient.imag.item())

    if bit == 1:
        new_real = original_real + delta
    else:
        new_real = original_real - delta

    return torch.complex(
        torch.tensor(new_real, dtype=coefficient.real.dtype),
        torch.tensor(original_imag, dtype=coefficient.imag.dtype),
    )
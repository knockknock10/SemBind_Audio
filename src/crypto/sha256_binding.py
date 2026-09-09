"""
SemBind-Audio

Cryptographic Payload Binding Module

The 256-bit payload is the SHA-256 cryptographic digest of a canonical,
sort-key-serialized JSON string that binds:

    1. semantic fingerprint of the source audio
    2. ownership identity
    3. timestamp / provenance information

Embedding the digest (rather than the raw metadata) means:
  * a single flipped bit causes a completely different digest (avalanche), and
  * the extracted bits are directly comparable to SHA-256(candidate metadata),
    so verification checks the intended metadata without requiring a
    reversible recovery of the original payload text (which is impossible for
    a cryptographic hash).

Extraction "reconstruction" therefore yields the 256-bit digest hex string,
not the original payload text. Verification re-hashes a candidate payload
string and compares it to the digest hex recovered from the extracted bits.
"""

import hashlib
import json
from datetime import datetime, timezone


def generate_semantic_fingerprint(embedding) -> str:
    """
    Generate a SHA-256 fingerprint from a HuBERT embedding vector.

    The fingerprint is derived from the embedding's numerical values,
    providing a content-derived identifier for the source audio.

    Parameters
    ----------
    embedding : torch.Tensor
        1-D tensor of HuBERT embedding coefficients.

    Returns
    -------
    str
        Hex-encoded SHA-256 fingerprint.
    """
    import torch
    embedding_bytes = (
        embedding.detach()
        .cpu()
        .float()
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(embedding_bytes).hexdigest()


def serialize_payload(payload: dict) -> str:
    """Canonical, deterministic serialization of a payload dict."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def bits_to_hex(bit_tensor) -> str:
    """Recover the 64-char hex digest from a 256-bit tensor (LSB-order as stored)."""
    bit_str = "".join(str(int(b)) for b in bit_tensor.detach().cpu().tolist())
    return hex(int(bit_str, 2))[2:].zfill(64)


def hex_to_bits(hex_str: str):
    """Convert a 64-char hex digest into a 256-bit tensor."""
    import torch
    return torch.tensor(
        [int(c) for c in bin(int(hex_str, 16))[2:].zfill(256)],
        dtype=torch.uint8,
    )


def generate_payload(
    project: str,
    owner_id: str,
    timestamp: str,
    semantic_fingerprint: str,
) -> dict:
    """
    Generate the cryptographically bound payload dictionary.

    Returns the payload dict, its canonical string, its hash, and the
    256-bit tensor (the SHA-256 digest of the canonical string).
    """
    import torch
    payload = {
        "project": project,
        "owner_id": owner_id,
        "timestamp": timestamp,
        "semantic_fingerprint": semantic_fingerprint,
    }
    payload_string = serialize_payload(payload)
    payload_hash = hashlib.sha256(payload_string.encode("utf-8")).hexdigest()
    bit_tensor = hex_to_bits(payload_hash)

    return {
        "payload": payload,
        "payload_string": payload_string,
        "payload_hash": payload_hash,
        "payload_bits": bit_tensor,
    }


def reconstruct_hash_from_bits(bit_tensor) -> str:
    """
    Recover the 64-char SHA-256 digest hex string from the embedded 256 bits.

    This is the only metadata recoverable from the bits (the original payload
    text itself is not recoverable, since SHA-256 is one-way).
    """
    return bits_to_hex(bit_tensor)


def vault_payload_string_to_dict(payload_string: str) -> dict:
    """Parse a canonical payload string back into a dict (for diagnostics)."""
    return json.loads(payload_string)


def verify_payload_bits(bit_tensor, expected_payload: dict) -> dict:
    """
    Verify extracted bits against expected payload metadata.

    Computes SHA-256(candidate payload string) and compares it to the digest
    recovered from the extracted bits. This directly checks the intended
    metadata (owner, timestamp, semantic fingerprint), not merely syntactic
    validity of the bits.

    Returns a result dict with a boolean match flag.
    """
    candidate_string = serialize_payload(expected_payload)
    candidate_hash = hashlib.sha256(candidate_string.encode("utf-8")).hexdigest()
    extracted_hash = bits_to_hex(bit_tensor)
    match = (extracted_hash == candidate_hash)
    return {
        "verified": match,
        "extracted_hash": extracted_hash,
        "expected_hash": candidate_hash,
        "candidate_payload_string": candidate_string,
    }


def verify_cryptographic_boundary(
    expected_payload: dict,
    extracted_payload: dict,
) -> bool:
    """
    Backward-compatible wrapper comparing two payload dicts field-by-field.
    """
    expected_keys = sorted(expected_payload.keys())
    extracted_keys = sorted(extracted_payload.keys())
    if expected_keys != extracted_keys:
        return False
    for key in expected_keys:
        if expected_payload[key] != extracted_payload[key]:
            return False
    return True

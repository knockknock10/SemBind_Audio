from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch


RESULT_DIR = PROJECT_ROOT / "results" / "watermark"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "results"
    / "embeddings"
    / "hubert_embeddings.pt"
)

OWNER_ID = "SemBind-Audio"
TIMESTAMP = datetime.now(timezone.utc).isoformat()


def tensor_to_bytes(tensor):
    tensor = tensor.detach().cpu().float().contiguous()
    return tensor.numpy().tobytes()


def main():

    print("=" * 60)
    print("SemBind-Audio : Experiment 06")
    print("Watermark Payload & Cryptographic Binding")
    print("=" * 60)

    embeddings = torch.load(
        EMBEDDING_FILE,
        map_location="cpu",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got {embeddings.shape}"
        )

    semantic_embedding = embeddings.mean(dim=0)

    embedding_bytes = tensor_to_bytes(
        semantic_embedding
    )

    semantic_fingerprint = hashlib.sha256(
        embedding_bytes
    ).hexdigest()

    payload = {
        "project": "SemBind-Audio",
        "owner_id": OWNER_ID,
        "timestamp": TIMESTAMP,
        "semantic_fingerprint": semantic_fingerprint,
    }

    payload_string = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    payload_hash = hashlib.sha256(
        payload_string.encode("utf-8")
    ).hexdigest()

    payload_bits = bin(
        int(payload_hash, 16)
    )[2:].zfill(256)

    bit_tensor = torch.tensor(
        [int(bit) for bit in payload_bits],
        dtype=torch.uint8,
    )

    with open(
        RESULT_DIR / "watermark_payload.json",
        "w",
    ) as f:
        json.dump(
            payload,
            f,
            indent=4,
        )

    with open(
        RESULT_DIR / "payload_string.txt",
        "w",
    ) as f:
        f.write(payload_string)

    with open(
        RESULT_DIR / "semantic_fingerprint.txt",
        "w",
    ) as f:
        f.write(semantic_fingerprint)

    with open(
        RESULT_DIR / "payload_hash.txt",
        "w",
    ) as f:
        f.write(payload_hash)

    with open(
        RESULT_DIR / "payload_bits.txt",
        "w",
    ) as f:
        f.write(payload_bits)

    torch.save(
        bit_tensor,
        RESULT_DIR / "payload_bits.pt",
    )

    summary = {
        "embedding_frames": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "fingerprint_algorithm": "SHA-256",
        "payload_hash_algorithm": "SHA-256",
        "payload_bits": 256,
        "owner_id": OWNER_ID,
        "timestamp": TIMESTAMP,
    }

    with open(
        RESULT_DIR / "payload_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print(
        f"Embedding Shape : {embeddings.shape}"
    )

    print(
        f"Global Embedding Dimension : "
        f"{semantic_embedding.shape[0]}"
    )

    print()
    print("Semantic Fingerprint")
    print(semantic_fingerprint)

    print()
    print("Payload Hash")
    print(payload_hash)

    print()
    print("Payload Length")
    print(f"{len(payload_bits)} bits")

    print()
    print("Payload Bits")
    print(payload_bits)

    print()
    print("Results Saved")

    print(
        RESULT_DIR / "watermark_payload.json"
    )

    print(
        RESULT_DIR / "payload_string.txt"
    )

    print(
        RESULT_DIR / "semantic_fingerprint.txt"
    )

    print(
        RESULT_DIR / "payload_hash.txt"
    )

    print(
        RESULT_DIR / "payload_bits.txt"
    )

    print(
        RESULT_DIR / "payload_bits.pt"
    )

    print(
        RESULT_DIR / "payload_summary.json"
    )


if __name__ == "__main__":
    main()
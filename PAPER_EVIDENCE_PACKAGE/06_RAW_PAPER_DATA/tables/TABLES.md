# SemBind-Audio — Final Results Tables (2026-09-09)

## TABLE_1_clean_roundtrip_E1 — clean_roundtrip_E1
```json
{
  "joint": {
    "n": 2620,
    "payload_success_rate": 1.0,
    "roundtrip_acc": 1.0,
    "ber": 0.0,
    "snr_db": 75.37
  },
  "semantic": {
    "n": 2620,
    "payload_success_rate": 1.0,
    "roundtrip_acc": 1.0,
    "ber": 0.0,
    "snr_db": 73.3
  },
  "spectral": {
    "n": 2620,
    "payload_success_rate": 1.0,
    "roundtrip_acc": 1.0,
    "ber": 0.0,
    "snr_db": 77.16
  },
  "random": {
    "n": 2620,
    "payload_success_rate": 0.9996,
    "roundtrip_acc": 1.0,
    "ber": 1e-06,
    "snr_db": 74.25
  }
}
```

## TABLE_2_audio_quality_E2 — audio_quality_E2
```json
{
  "snr_mean_db": 74.81824533247608,
  "snr_p95_db": 79.90735361290037
}
```

## TABLE_3_fixed_vs_adaptive_E3 — fixed_vs_adaptive_E3
```json
{
  "fixed": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.2904,
    "bit_error_rate": 0.0
  },
  "adaptive": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.8182,
    "bit_error_rate": 0.0
  }
}
```

## TABLE_4_selection_ablations_E4 — selection_ablations_E4
```json
{
  "A_semantic_only": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 72.749,
    "bit_error_rate": 0.0
  },
  "B_spectral_only": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 76.6287,
    "bit_error_rate": 0.0
  },
  "C_same_distortion_fixed": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.2904,
    "bit_error_rate": 0.0
  },
  "D_random": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 73.8048,
    "bit_error_rate": 0.0
  }
}
```

## TABLE_5_alpha_beta_sweep_E5 — alpha_beta_sweep_E5
```json
{
  "1.0": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 72.749,
    "bit_error_rate": 0.0
  },
  "0.75": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.5807,
    "bit_error_rate": 0.0
  },
  "0.7": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.8182,
    "bit_error_rate": 0.0
  },
  "0.5": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 75.5713,
    "bit_error_rate": 0.0
  },
  "0.25": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 76.2135,
    "bit_error_rate": 0.0
  },
  "0.0": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 76.6287,
    "bit_error_rate": 0.0
  }
}
```

## TABLE_6_strength_sweep_E6 — strength_sweep_E6
```json
{
  "0.0005": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 93.8713,
    "bit_error_rate": 0.0
  },
  "0.001": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 88.1644,
    "bit_error_rate": 0.0
  },
  "0.002": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 82.226,
    "bit_error_rate": 0.0
  },
  "0.005": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.2904,
    "bit_error_rate": 0.0
  },
  "0.008": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 70.2107,
    "bit_error_rate": 0.0
  },
  "0.01": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 68.2732,
    "bit_error_rate": 0.0
  }
}
```

## TABLE_7_redundancy_E7 — redundancy_E7
```json
{
  "1": {
    "n": 500,
    "roundtrip_accuracy": 0.9999,
    "snr_db": 78.887,
    "bit_error_rate": 0.0001
  },
  "2": {
    "n": 500,
    "roundtrip_accuracy": 0.9994,
    "snr_db": 76.4598,
    "bit_error_rate": 0.0006
  },
  "3": {
    "n": 500,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.8182,
    "bit_error_rate": 0.0
  }
}
```

## TABLE_8_crypto_E8 — crypto_E8
```json
{
  "tp": 500,
  "fp": 0,
  "tn": 500,
  "fn": 0,
  "n": 500,
  "truncation_rejected": 500
}
```

## TABLE_9_awgn_robustness_E9 — awgn_robustness_E9
```json
{
  "none": {
    "n": 500,
    "bit_accuracy": 1.0,
    "ber": 0.0,
    "payload_success": 1.0,
    "snr_db": 74.8182
  },
  "awgn_40db": {
    "n": 500,
    "bit_accuracy": 0.6248,
    "ber": 0.3752,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "awgn_30db": {
    "n": 500,
    "bit_accuracy": 0.541,
    "ber": 0.459,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "awgn_20db": {
    "n": 500,
    "bit_accuracy": 0.5124,
    "ber": 0.4876,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "awgn_10db": {
    "n": 500,
    "bit_accuracy": 0.5044,
    "ber": 0.4956,
    "payload_success": 0.0,
    "snr_db": 74.8182
  }
}
```

## TABLE_10_attack_robustness_E10 — attack_robustness_E10
```json
{
  "none": {
    "n": 500,
    "bit_accuracy": 1.0,
    "ber": 0.0,
    "payload_success": 1.0,
    "snr_db": 74.8182
  },
  "resample_8k": {
    "n": 500,
    "bit_accuracy": 0.7371,
    "ber": 0.2629,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "lowpass_4k": {
    "n": 500,
    "bit_accuracy": 0.5034,
    "ber": 0.4966,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "scale_0.9": {
    "n": 500,
    "bit_accuracy": 0.7941,
    "ber": 0.2059,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "mp3_128": {
    "n": 500,
    "bit_accuracy": 0.632,
    "ber": 0.368,
    "payload_success": 0.0,
    "snr_db": 74.8182
  },
  "awgn_20db": {
    "n": 500,
    "bit_accuracy": 0.5124,
    "ber": 0.4876,
    "payload_success": 0.0,
    "snr_db": 74.8182
  }
}
```

## TABLE_11_duration_bins_E11 — duration_bins_E11
```json
{
  "<4s": {
    "n": 770,
    "roundtrip_accuracy": 1.0,
    "snr_db": 72.5393029621022,
    "payload_success_rate": 1.0
  },
  "4-8s": {
    "n": 991,
    "roundtrip_accuracy": 1.0,
    "snr_db": 74.97662218366567,
    "payload_success_rate": 1.0
  },
  "8-12s": {
    "n": 428,
    "roundtrip_accuracy": 1.0,
    "snr_db": 77.30318503782678,
    "payload_success_rate": 1.0
  },
  "12-20s": {
    "n": 339,
    "roundtrip_accuracy": 1.0,
    "snr_db": 79.08286213420153,
    "payload_success_rate": 1.0
  },
  ">20s": {
    "n": 92,
    "roundtrip_accuracy": 1.0,
    "snr_db": 80.71406132593397,
    "payload_success_rate": 1.0
  }
}
```

## TABLE_12_per_speaker_E12 — per_speaker_E12
```json
{
  "speakers": 40,
  "worst_success": 1.0,
  "mean_snr_db": 75.73,
  "mean_success": 1.0
}
```

## TABLE_13_paired_stats_E13 — paired_stats_E13
```json
[
  {
    "a": "joint",
    "b": "semantic",
    "t": 203.00782526361075,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": 2.0728394676337274
  },
  {
    "a": "joint",
    "b": "spectral",
    "t": -126.4286795254889,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": -1.789363042310768
  },
  {
    "a": "joint",
    "b": "random",
    "t": 71.93424120070598,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": 1.1222147424604003
  },
  {
    "a": "semantic",
    "b": "spectral",
    "t": -213.86973379492247,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": -3.8622025099444954
  },
  {
    "a": "random",
    "b": "semantic",
    "t": 81.17585243015984,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": 0.9506247251733271
  },
  {
    "a": "random",
    "b": "spectral",
    "t": -193.10140092074164,
    "p": 0.0,
    "delta_mean_snr_A_minus_B": -2.9115777847711684
  }
]
```

## TABLE_14_compute_cost_E14 — compute_cost_E14
```json
{
  "joint": {
    "total_s": 40.6
  },
  "semantic": {
    "total_s": 40.7
  },
  "spectral": {
    "total_s": 37.4
  },
  "random": {
    "total_s": 38.4
  }
}
```

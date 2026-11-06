# SemBind-Audio

SemBind-Audio is a research project exploring whether information about the content of an audio recording can help choose where to place an audio watermark.

The current implementation uses HuBERT to get frame-level audio representations and an STFT representation of the waveform. It uses those features to score candidate time-frequency locations, then embeds a 256-bit payload with three copies of each bit.

This is still an experimental method. The clean-audio results are good, but the current embedding method does not hold up when the audio is changed. I have kept that result in the project notes rather than presenting clean extraction as proof of robustness.

## How the current method works

1. Load the audio at 16 kHz.
2. Extract frame-level representations with `facebook/hubert-base-ls960`.
3. Calculate an STFT using a Hann window, a 1024-point FFT and a hop length of 320 samples.
4. Calculate semantic suitability from a leave-one-out contribution score.
5. Calculate spectral suitability from the local STFT energy.
6. Combine the two scores multiplicatively and use the result to set the embedding strength.
7. Place the payload at distinct time-frequency coordinates, with three replicas per bit.
8. Recover the bits with a reference-based extraction step and majority voting across the replicas.

The current extraction method needs the original, unwatermarked audio as a reference. It is not a blind watermark detector.

## Experiments

The final experiment set recorded in `RESULT_STATUS.md` uses the LibriSpeech test-clean set: 2,620 recordings from 40 speakers.

| Test | Recorded result |
| --- | --- |
| Clean extraction — joint, semantic and spectral selection | 100% payload success |
| Clean extraction — random selection | 99.96% payload success |
| SNR across the four clean-selection methods | 73.30–77.16 dB |
| Authentication check | 500 true positives, 500 true negatives, no false positives or false negatives |

These results describe the tested setup. They should not be read as evidence that the watermark survives normal audio processing.

## What did not work

The robustness experiments exposed a major limitation. Payload recovery fell to 0% under the tested mild AWGN at 40 dB, MP3 compression, resampling, low-pass filtering and amplitude changes. Bit accuracy under these attacks was around 0.50–0.79.

The current method changes individual STFT coefficients by small amounts. Those changes are easy to disturb when the audio is processed. This is the main problem that needs to be addressed next.

Audio quality was evaluated using SNR. MUSHRA and PESQ listening-quality tests were not run.

## Code and experiment notes

- `final_methodology.py` contains the current embedding and extraction method.
- `RESULT_STATUS.md` identifies the final experiment set and separates it from older results.
- `evaluation_framework.py` is an older evaluation path; its outputs are not the source for the final results.
- `PAPER_EVIDENCE_PACKAGE/` contains material collected for the paper.

The repository does not currently provide a validated Conda environment or a cross-platform lockfile. `requirements.txt` is a pinned environment capture; it may need a platform-appropriate PyTorch/torchaudio installation before it works on another machine. A starting point for an isolated install is:\n\n\`\`\`sh\npython -m venv .venv\nsource .venv/bin/activate\npython -m pip install --upgrade pip\npython -m pip install -r requirements.txt\n\`\`\`\n\nThis setup has not been verified across operating systems. The results and code should be checked together before using the numbers in another report or presentation.

## Current status

The clean-reference experiments and the authentication check are recorded. The attack results show that the current embedding approach is not robust. Improving that is still open work, not a solved part of the project.

# baseline-comparison-v0.1

baseline-comparison-v0.1 compares deterministic mappings using implemented descriptive metrics only. It does not measure objective musicality, consonance, emotional fit, or listener preference.

Pipeline version: `corpus-pipeline-v0.1`

Strata include the pilot fixture (core cells + orthographic challenge) as well as gutenberg_validation, gutenberg_holdout, wordlist_validation, and wordlist_holdout — the frequency-ranked control is fit on the Gutenberg train partition only and evaluated here against material it never saw.

## Metric versions

- `register_center`: `register_center_v0.1`
- `pitch_span`: `pitch_span_v0.1`
- `interval_contour`: `interval_contour_v0.1`
- `directional_balance`: `directional_balance_v0.1`
- `repetition_index`: `repetition_index_v0.1`

## Results by mapping and stratum (macro = per-item average, micro = pooled)

### lape-26-en-general-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.86 | 12.58 | 0.135 | 0.049 | 23/18/2 |
| short_negative | 12 | 61.52 | 15.17 | 0.316 | 0.021 | 27/18/1 |
| short_neutral | 12 | 60.73 | 15.25 | 0.285 | 0.042 | 26/18/2 |
| medium_positive | 12 | 60.46 | 16.50 | 0.117 | 0.070 | 33/33/5 |
| medium_negative | 12 | 61.05 | 14.58 | 0.137 | 0.028 | 40/30/2 |
| medium_neutral | 12 | 60.75 | 14.58 | 0.195 | 0.000 | 39/37/0 |
| long_positive | 12 | 60.51 | 16.58 | 0.025 | 0.037 | 57/50/4 |
| long_negative | 12 | 60.17 | 15.33 | 0.062 | 0.044 | 57/49/5 |
| long_neutral | 12 | 60.48 | 16.42 | 0.119 | 0.045 | 57/48/5 |
| orthographic_challenge | 12 | 61.12 | 13.92 | 0.015 | 0.035 | 47/40/3 |
| gutenberg_validation | 200 | 60.39 | 14.61 | 0.091 | 0.027 | 656/575/32 |
| gutenberg_holdout | 200 | 60.27 | 14.07 | 0.048 | 0.037 | 630/580/46 |
| wordlist_validation | 200 | 60.37 | 15.39 | 0.070 | 0.031 | 790/678/44 |
| wordlist_holdout | 200 | 60.59 | 15.22 | 0.046 | 0.022 | 793/692/34 |

### sequential-chromatic-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 57.85 | 18.25 | 0.249 | 0.049 | 22/19/2 |
| short_negative | 12 | 59.85 | 20.50 | 0.300 | 0.021 | 29/16/1 |
| short_neutral | 12 | 57.79 | 18.83 | 0.184 | 0.042 | 24/20/2 |
| medium_positive | 12 | 59.30 | 18.42 | 0.033 | 0.070 | 32/34/5 |
| medium_negative | 12 | 59.04 | 19.25 | 0.017 | 0.028 | 40/30/2 |
| medium_neutral | 12 | 58.60 | 18.25 | -0.011 | 0.000 | 39/37/0 |
| long_positive | 12 | 58.39 | 19.75 | 0.098 | 0.037 | 60/47/4 |
| long_negative | 12 | 60.21 | 18.17 | 0.024 | 0.044 | 55/51/5 |
| long_neutral | 12 | 59.03 | 20.17 | -0.011 | 0.045 | 52/53/5 |
| orthographic_challenge | 12 | 57.91 | 18.17 | -0.095 | 0.035 | 44/43/3 |
| gutenberg_validation | 200 | 59.26 | 17.43 | 0.029 | 0.027 | 631/600/32 |
| gutenberg_holdout | 200 | 58.95 | 17.40 | 0.013 | 0.037 | 616/594/46 |
| wordlist_validation | 200 | 59.01 | 18.65 | 0.037 | 0.031 | 730/738/44 |
| wordlist_holdout | 200 | 58.66 | 18.55 | 0.034 | 0.022 | 733/752/34 |

### frequency-ranked-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.63 | 12.33 | -0.401 | 0.049 | 12/29/2 |
| short_negative | 12 | 58.67 | 13.25 | -0.031 | 0.021 | 20/25/1 |
| short_neutral | 12 | 59.46 | 12.25 | -0.133 | 0.042 | 21/23/2 |
| medium_positive | 12 | 59.99 | 13.83 | -0.055 | 0.070 | 33/33/5 |
| medium_negative | 12 | 59.82 | 13.67 | -0.160 | 0.028 | 29/41/2 |
| medium_neutral | 12 | 60.36 | 15.58 | -0.050 | 0.000 | 35/41/0 |
| long_positive | 12 | 60.15 | 14.75 | -0.038 | 0.037 | 55/52/4 |
| long_negative | 12 | 60.52 | 14.08 | 0.019 | 0.044 | 53/53/5 |
| long_neutral | 12 | 60.19 | 15.83 | -0.066 | 0.045 | 51/54/5 |
| orthographic_challenge | 12 | 60.13 | 14.42 | -0.075 | 0.035 | 41/46/3 |
| gutenberg_validation | 200 | 60.54 | 13.69 | 0.032 | 0.027 | 648/583/32 |
| gutenberg_holdout | 200 | 60.34 | 13.11 | 0.061 | 0.037 | 614/596/46 |
| wordlist_validation | 200 | 60.34 | 14.29 | -0.036 | 0.031 | 724/744/44 |
| wordlist_holdout | 200 | 60.31 | 13.99 | -0.041 | 0.022 | 717/768/34 |

### circle-of-fifths-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 58.12 | 19.08 | 0.148 | 0.049 | 23/18/2 |
| short_negative | 12 | 60.58 | 20.42 | 0.183 | 0.021 | 24/21/1 |
| short_neutral | 12 | 58.44 | 20.17 | 0.211 | 0.042 | 24/20/2 |
| medium_positive | 12 | 59.99 | 20.58 | 0.010 | 0.070 | 30/36/5 |
| medium_negative | 12 | 59.29 | 20.33 | -0.026 | 0.028 | 32/38/2 |
| medium_neutral | 12 | 59.04 | 19.00 | 0.006 | 0.000 | 37/39/0 |
| long_positive | 12 | 59.00 | 21.58 | 0.110 | 0.037 | 56/51/4 |
| long_negative | 12 | 60.91 | 20.08 | 0.016 | 0.044 | 55/51/5 |
| long_neutral | 12 | 59.26 | 21.08 | -0.024 | 0.045 | 48/57/5 |
| orthographic_challenge | 12 | 57.92 | 20.08 | -0.053 | 0.035 | 40/47/3 |
| gutenberg_validation | 200 | 59.71 | 18.91 | 0.005 | 0.027 | 604/627/32 |
| gutenberg_holdout | 200 | 59.36 | 18.76 | 0.027 | 0.037 | 601/609/46 |
| wordlist_validation | 200 | 59.34 | 20.21 | 0.012 | 0.031 | 718/750/44 |
| wordlist_holdout | 200 | 59.04 | 20.27 | 0.026 | 0.022 | 733/752/34 |

### random-seed-026-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 57.93 | 15.58 | -0.372 | 0.049 | 15/26/2 |
| short_negative | 12 | 56.33 | 15.58 | -0.265 | 0.021 | 19/26/1 |
| short_neutral | 12 | 57.35 | 16.83 | -0.181 | 0.042 | 21/23/2 |
| medium_positive | 12 | 56.87 | 18.58 | -0.124 | 0.070 | 29/37/5 |
| medium_negative | 12 | 56.38 | 16.50 | -0.102 | 0.028 | 31/39/2 |
| medium_neutral | 12 | 58.16 | 18.08 | -0.029 | 0.000 | 38/38/0 |
| long_positive | 12 | 57.25 | 18.25 | -0.090 | 0.037 | 52/55/4 |
| long_negative | 12 | 56.34 | 19.50 | -0.056 | 0.044 | 49/57/5 |
| long_neutral | 12 | 56.66 | 19.92 | -0.120 | 0.045 | 49/56/5 |
| orthographic_challenge | 12 | 56.72 | 18.83 | -0.075 | 0.035 | 41/46/3 |
| gutenberg_validation | 200 | 57.20 | 18.55 | -0.123 | 0.027 | 548/683/32 |
| gutenberg_holdout | 200 | 56.97 | 17.81 | -0.170 | 0.037 | 521/689/46 |
| wordlist_validation | 200 | 57.86 | 18.64 | -0.113 | 0.031 | 685/783/44 |
| wordlist_holdout | 200 | 57.52 | 18.52 | -0.112 | 0.022 | 663/822/34 |

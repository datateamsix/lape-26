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
| short_positive | 12 | 61.74 | 13.92 | 0.227 | 0.062 | 23/20/3 |
| short_negative | 12 | 60.39 | 12.58 | 0.460 | 0.090 | 24/13/3 |
| short_neutral | 12 | 61.62 | 15.33 | 0.206 | 0.042 | 24/19/2 |
| medium_positive | 12 | 61.05 | 15.08 | 0.077 | 0.012 | 36/37/1 |
| medium_negative | 12 | 60.91 | 17.50 | 0.124 | 0.036 | 39/33/3 |
| medium_neutral | 12 | 60.84 | 13.67 | 0.057 | 0.066 | 36/29/5 |
| long_positive | 12 | 59.67 | 16.58 | 0.028 | 0.056 | 60/39/6 |
| long_negative | 12 | 60.79 | 16.92 | 0.144 | 0.021 | 58/46/2 |
| long_neutral | 12 | 59.88 | 17.50 | 0.162 | 0.009 | 62/50/1 |
| orthographic_challenge | 12 | 61.07 | 16.58 | 0.111 | 0.096 | 44/38/8 |
| gutenberg_validation | 200 | 60.39 | 14.61 | 0.091 | 0.027 | 656/575/32 |
| gutenberg_holdout | 200 | 60.27 | 14.07 | 0.048 | 0.037 | 630/580/46 |
| wordlist_validation | 200 | 60.81 | 14.69 | 0.064 | 0.026 | 774/706/41 |
| wordlist_holdout | 200 | 60.53 | 14.88 | 0.055 | 0.029 | 757/685/42 |

### sequential-chromatic-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.28 | 19.50 | 0.231 | 0.062 | 25/18/3 |
| short_negative | 12 | 59.71 | 15.17 | -0.083 | 0.090 | 19/18/3 |
| short_neutral | 12 | 59.82 | 18.08 | 0.427 | 0.042 | 29/14/2 |
| medium_positive | 12 | 58.27 | 19.08 | 0.075 | 0.012 | 40/33/1 |
| medium_negative | 12 | 59.25 | 19.92 | 0.003 | 0.036 | 36/36/3 |
| medium_neutral | 12 | 58.70 | 15.50 | 0.038 | 0.066 | 32/33/5 |
| long_positive | 12 | 59.20 | 19.42 | 0.072 | 0.056 | 54/45/6 |
| long_negative | 12 | 59.42 | 19.83 | -0.001 | 0.021 | 54/50/2 |
| long_neutral | 12 | 59.27 | 22.00 | -0.047 | 0.009 | 55/57/1 |
| orthographic_challenge | 12 | 59.18 | 19.25 | 0.098 | 0.096 | 43/39/8 |
| gutenberg_validation | 200 | 59.26 | 17.43 | 0.029 | 0.027 | 631/600/32 |
| gutenberg_holdout | 200 | 58.95 | 17.40 | 0.013 | 0.037 | 616/594/46 |
| wordlist_validation | 200 | 58.46 | 18.57 | 0.049 | 0.026 | 737/743/41 |
| wordlist_holdout | 200 | 58.30 | 18.68 | 0.015 | 0.029 | 724/718/42 |

### frequency-ranked-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.86 | 12.92 | -0.064 | 0.062 | 18/25/3 |
| short_negative | 12 | 59.75 | 12.83 | 0.201 | 0.090 | 21/16/3 |
| short_neutral | 12 | 59.71 | 13.08 | -0.152 | 0.042 | 19/24/2 |
| medium_positive | 12 | 60.08 | 16.33 | -0.048 | 0.012 | 35/38/1 |
| medium_negative | 12 | 59.40 | 14.17 | 0.010 | 0.036 | 35/37/3 |
| medium_neutral | 12 | 60.03 | 12.67 | -0.009 | 0.066 | 34/31/5 |
| long_positive | 12 | 60.29 | 16.17 | -0.046 | 0.056 | 49/50/6 |
| long_negative | 12 | 60.38 | 15.92 | -0.025 | 0.021 | 51/53/2 |
| long_neutral | 12 | 59.46 | 16.25 | -0.084 | 0.009 | 50/62/1 |
| orthographic_challenge | 12 | 60.98 | 16.33 | -0.097 | 0.096 | 38/44/8 |
| gutenberg_validation | 200 | 60.54 | 13.69 | 0.032 | 0.027 | 648/583/32 |
| gutenberg_holdout | 200 | 60.34 | 13.11 | 0.061 | 0.037 | 614/596/46 |
| wordlist_validation | 200 | 60.12 | 14.12 | -0.030 | 0.026 | 735/745/41 |
| wordlist_holdout | 200 | 60.28 | 13.51 | -0.029 | 0.029 | 706/736/42 |

### circle-of-fifths-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.05 | 20.25 | 0.245 | 0.062 | 26/17/3 |
| short_negative | 12 | 61.26 | 14.58 | -0.024 | 0.090 | 17/20/3 |
| short_neutral | 12 | 59.70 | 18.58 | 0.313 | 0.042 | 26/17/2 |
| medium_positive | 12 | 58.26 | 19.83 | 0.066 | 0.012 | 35/38/1 |
| medium_negative | 12 | 59.24 | 19.33 | -0.008 | 0.036 | 33/39/3 |
| medium_neutral | 12 | 58.83 | 18.83 | 0.045 | 0.066 | 39/26/5 |
| long_positive | 12 | 59.64 | 21.17 | 0.082 | 0.056 | 51/48/6 |
| long_negative | 12 | 59.92 | 20.75 | -0.028 | 0.021 | 48/56/2 |
| long_neutral | 12 | 59.12 | 23.00 | -0.067 | 0.009 | 50/62/1 |
| orthographic_challenge | 12 | 59.48 | 21.50 | 0.078 | 0.096 | 46/36/8 |
| gutenberg_validation | 200 | 59.71 | 18.91 | 0.005 | 0.027 | 604/627/32 |
| gutenberg_holdout | 200 | 59.36 | 18.76 | 0.027 | 0.037 | 601/609/46 |
| wordlist_validation | 200 | 58.77 | 20.26 | 0.035 | 0.026 | 730/750/41 |
| wordlist_holdout | 200 | 58.54 | 20.20 | 0.019 | 0.029 | 692/750/42 |

### random-seed-026-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 56.80 | 15.75 | -0.230 | 0.062 | 19/24/3 |
| short_negative | 12 | 57.27 | 15.42 | -0.252 | 0.090 | 18/19/3 |
| short_neutral | 12 | 58.57 | 15.58 | -0.278 | 0.042 | 20/23/2 |
| medium_positive | 12 | 56.47 | 18.08 | -0.090 | 0.012 | 35/38/1 |
| medium_negative | 12 | 56.83 | 20.08 | -0.172 | 0.036 | 28/44/3 |
| medium_neutral | 12 | 56.49 | 16.50 | -0.019 | 0.066 | 27/38/5 |
| long_positive | 12 | 57.65 | 19.50 | -0.062 | 0.056 | 46/53/6 |
| long_negative | 12 | 56.39 | 19.08 | -0.024 | 0.021 | 47/57/2 |
| long_neutral | 12 | 57.73 | 20.25 | -0.068 | 0.009 | 51/61/1 |
| orthographic_challenge | 12 | 57.19 | 17.67 | -0.127 | 0.096 | 37/45/8 |
| gutenberg_validation | 200 | 57.20 | 18.55 | -0.123 | 0.027 | 548/683/32 |
| gutenberg_holdout | 200 | 56.97 | 17.81 | -0.170 | 0.037 | 521/689/46 |
| wordlist_validation | 200 | 57.36 | 18.35 | -0.056 | 0.026 | 675/805/41 |
| wordlist_holdout | 200 | 57.44 | 18.34 | -0.072 | 0.029 | 655/787/42 |

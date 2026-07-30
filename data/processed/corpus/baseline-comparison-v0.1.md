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
| short_positive | 12 | 60.70 | 12.92 | 0.182 | 0.042 | 25/17/2 |
| short_negative | 12 | 59.88 | 15.58 | 0.264 | 0.028 | 24/17/1 |
| short_neutral | 12 | 61.31 | 12.25 | 0.022 | 0.028 | 18/23/1 |
| medium_positive | 12 | 60.89 | 13.67 | 0.117 | 0.026 | 37/37/2 |
| medium_negative | 12 | 59.83 | 15.58 | 0.042 | 0.029 | 37/33/2 |
| medium_neutral | 12 | 60.68 | 16.17 | 0.044 | 0.026 | 34/35/2 |
| long_positive | 12 | 59.68 | 18.08 | 0.040 | 0.033 | 54/51/4 |
| long_negative | 12 | 60.45 | 17.42 | 0.127 | 0.027 | 57/48/3 |
| long_neutral | 12 | 61.12 | 17.67 | 0.117 | 0.021 | 57/48/2 |
| orthographic_challenge | 12 | 61.25 | 14.58 | 0.145 | 0.039 | 51/33/3 |
| gutenberg_validation | 200 | 60.39 | 14.61 | 0.091 | 0.027 | 656/575/32 |
| gutenberg_holdout | 200 | 60.27 | 14.07 | 0.048 | 0.037 | 630/580/46 |
| wordlist_validation | 200 | 60.92 | 15.02 | 0.054 | 0.036 | 781/693/50 |
| wordlist_holdout | 200 | 60.49 | 15.01 | 0.055 | 0.040 | 798/695/60 |

### sequential-chromatic-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 59.42 | 18.50 | 0.132 | 0.042 | 22/20/2 |
| short_negative | 12 | 59.45 | 16.50 | 0.111 | 0.028 | 23/18/1 |
| short_neutral | 12 | 59.12 | 14.75 | 0.123 | 0.028 | 22/19/1 |
| medium_positive | 12 | 57.51 | 18.50 | 0.052 | 0.026 | 37/37/2 |
| medium_negative | 12 | 59.80 | 17.83 | 0.038 | 0.029 | 38/32/2 |
| medium_neutral | 12 | 59.56 | 18.17 | 0.049 | 0.026 | 38/31/2 |
| long_positive | 12 | 60.07 | 20.00 | 0.070 | 0.033 | 60/45/4 |
| long_negative | 12 | 59.08 | 20.42 | 0.093 | 0.027 | 57/48/3 |
| long_neutral | 12 | 58.70 | 20.42 | 0.070 | 0.021 | 55/50/2 |
| orthographic_challenge | 12 | 58.32 | 19.17 | 0.010 | 0.039 | 43/41/3 |
| gutenberg_validation | 200 | 59.26 | 17.43 | 0.029 | 0.027 | 631/600/32 |
| gutenberg_holdout | 200 | 58.95 | 17.40 | 0.013 | 0.037 | 616/594/46 |
| wordlist_validation | 200 | 58.71 | 18.66 | 0.035 | 0.036 | 742/732/50 |
| wordlist_holdout | 200 | 58.61 | 18.32 | 0.016 | 0.040 | 723/770/60 |

### frequency-ranked-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 60.06 | 12.08 | 0.176 | 0.042 | 23/19/2 |
| short_negative | 12 | 59.58 | 13.17 | -0.078 | 0.028 | 20/21/1 |
| short_neutral | 12 | 60.11 | 12.75 | -0.071 | 0.028 | 21/20/1 |
| medium_positive | 12 | 60.70 | 13.83 | -0.094 | 0.026 | 31/43/2 |
| medium_negative | 12 | 59.34 | 14.17 | -0.104 | 0.029 | 29/41/2 |
| medium_neutral | 12 | 59.77 | 13.08 | -0.144 | 0.026 | 30/39/2 |
| long_positive | 12 | 60.34 | 15.58 | -0.071 | 0.033 | 53/52/4 |
| long_negative | 12 | 59.81 | 15.75 | -0.027 | 0.027 | 47/58/3 |
| long_neutral | 12 | 60.55 | 15.58 | -0.003 | 0.021 | 49/56/2 |
| orthographic_challenge | 12 | 59.07 | 15.17 | -0.082 | 0.039 | 46/38/3 |
| gutenberg_validation | 200 | 60.54 | 13.69 | 0.032 | 0.027 | 648/583/32 |
| gutenberg_holdout | 200 | 60.34 | 13.11 | 0.061 | 0.037 | 614/596/46 |
| wordlist_validation | 200 | 60.46 | 14.54 | -0.026 | 0.036 | 723/751/50 |
| wordlist_holdout | 200 | 60.75 | 13.71 | -0.020 | 0.040 | 749/744/60 |

### circle-of-fifths-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 60.06 | 17.17 | 0.085 | 0.042 | 23/19/2 |
| short_negative | 12 | 60.25 | 16.75 | 0.138 | 0.028 | 22/19/1 |
| short_neutral | 12 | 59.66 | 15.25 | 0.219 | 0.028 | 25/16/1 |
| medium_positive | 12 | 58.05 | 21.42 | 0.039 | 0.026 | 34/40/2 |
| medium_negative | 12 | 60.54 | 17.67 | 0.008 | 0.029 | 34/36/2 |
| medium_neutral | 12 | 59.76 | 19.67 | -0.010 | 0.026 | 30/39/2 |
| long_positive | 12 | 60.23 | 20.92 | 0.030 | 0.033 | 56/49/4 |
| long_negative | 12 | 59.31 | 21.00 | 0.058 | 0.027 | 51/54/3 |
| long_neutral | 12 | 59.44 | 20.58 | 0.045 | 0.021 | 50/55/2 |
| orthographic_challenge | 12 | 59.31 | 20.08 | 0.008 | 0.039 | 45/39/3 |
| gutenberg_validation | 200 | 59.71 | 18.91 | 0.005 | 0.027 | 604/627/32 |
| gutenberg_holdout | 200 | 59.36 | 18.76 | 0.027 | 0.037 | 601/609/46 |
| wordlist_validation | 200 | 59.05 | 20.38 | 0.038 | 0.036 | 715/759/50 |
| wordlist_holdout | 200 | 59.12 | 19.83 | 0.005 | 0.040 | 700/793/60 |

### random-seed-026-v0.1

| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |
|---|---:|---:|---:|---:|---:|---|
| short_positive | 12 | 58.05 | 17.92 | -0.404 | 0.042 | 13/29/2 |
| short_negative | 12 | 56.30 | 13.42 | -0.067 | 0.028 | 20/21/1 |
| short_neutral | 12 | 57.26 | 15.17 | -0.073 | 0.028 | 21/20/1 |
| medium_positive | 12 | 56.61 | 17.50 | -0.096 | 0.026 | 34/40/2 |
| medium_negative | 12 | 57.11 | 19.50 | -0.123 | 0.029 | 29/41/2 |
| medium_neutral | 12 | 56.86 | 16.17 | -0.025 | 0.026 | 33/36/2 |
| long_positive | 12 | 56.98 | 18.58 | -0.062 | 0.033 | 53/52/4 |
| long_negative | 12 | 56.67 | 19.83 | -0.053 | 0.027 | 54/51/3 |
| long_neutral | 12 | 56.84 | 20.00 | -0.090 | 0.021 | 50/55/2 |
| orthographic_challenge | 12 | 57.90 | 19.25 | -0.218 | 0.039 | 36/48/3 |
| gutenberg_validation | 200 | 57.20 | 18.55 | -0.123 | 0.027 | 548/683/32 |
| gutenberg_holdout | 200 | 56.97 | 17.81 | -0.170 | 0.037 | 521/689/46 |
| wordlist_validation | 200 | 57.58 | 18.79 | -0.063 | 0.036 | 687/787/50 |
| wordlist_holdout | 200 | 57.31 | 18.89 | -0.115 | 0.040 | 682/811/60 |

# Data Sources and Licensing

LAPE-26 separates project code from external data licenses.

## Rules

Every external dataset must have a manifest containing:

- Dataset ID and version
- Source and retrieval date
- License and redistribution terms
- Checksum
- Preparation script
- Language and locale
- Known biases and exclusions

## Copyrighted lyrics

Copyrighted lyric text must not be committed unless the source license explicitly permits redistribution and research use. Restricted local corpora belong in `data/restricted/`, which is ignored by Git.

Public experiments may publish aggregate frequencies or derived statistics only when the dataset license permits those outputs.

## Listener data

Public listener data must be anonymous, consented, minimized, and documented. Raw free-text responses should be reviewed for accidental personal information before release.

## Approved corpus datasets (Phase 2)

Five datasets are approved for the corpus and stimulus pipeline. Each has a
manifest under `data/manifests/`. Brown, Names, CMUdict, and NLTK's
punkt/punkt_tab tokenizer are explicitly excluded — sentence splitting uses
this project's own deterministic splitter instead
(`python/lape26/corpus/nltk_adapter.py`) to avoid depending on a 6th,
unreviewed NLTK resource.

| Dataset | Role | License | Redistribution | Manifest |
|---|---|---|---|---|
| Gutenberg (NLTK sample) | Natural text, sentences, boundaries, character/bigram frequency | Public Domain (US) | Yes — derived stats only | `data/manifests/gutenberg.yaml` |
| words | Candidate vocabulary, orthographic stress cases | Public Domain / unrestricted | Yes — selected words only | `data/manifests/words.yaml` |
| WordNet | Dictionary validation, part-of-speech, morphological grouping | WordNet License (permissive) | Yes — per-item validation results only | `data/manifests/wordnet.yaml` |
| Opinion Lexicon | Positive/negative candidate labels, confirmed against VADER | CC BY 4.0 (Copyright 2011 Bing Liu) | Yes, with attribution | `data/manifests/opinion-lexicon.yaml` |
| VADER | Sentiment scoring / polarity confirmation | MIT License | Yes | `data/manifests/vader.yaml` |

See `data/corpus/README.md` for dataset roles and exclusion rules in detail.

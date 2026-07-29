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

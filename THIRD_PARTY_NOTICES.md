# Third-Party Notices

The Word Explorer loads Tone.js 15.1.22 from UNPKG at runtime. Tone.js remains the property of its contributors and is governed by its own project license and notices. LAPE-26 does not redistribute the Tone.js source in this Phase 0 package.

External datasets and future dependencies must be documented before inclusion.

## Corpus pipeline datasets (Phase 2)

- **NLTK Gutenberg Sample Corpus** — Project Gutenberg texts, Public Domain (US), bundled via NLTK (`nltk.download('gutenberg')`).
- **NLTK Words Corpus** — derived from the Unix words list, public domain / unrestricted, bundled via NLTK (`nltk.download('words')`).
- **Princeton WordNet** (via NLTK) — WordNet License. "This software and database is being provided to you, the LICENSEE, by Princeton University under the following license. ... Princeton University makes no representations about the suitability of the licensed software, database or documentation for any purpose." Bundled via NLTK (`nltk.download('wordnet')`).
- **Opinion Lexicon** — Minqing Hu and Bing Liu, "Mining and Summarizing Customer Reviews", KDD 2004. **Creative Commons Attribution 4.0 International (CC BY 4.0), Copyright (C) 2011 Bing Liu**, per NLTK's own data index. Bundled via NLTK (`nltk.download('opinion_lexicon')`).
- **VADER Sentiment Lexicon** — Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-14. MIT License. Bundled via NLTK (`nltk.download('vader_lexicon')`).

Full licensing and redistribution details for each are in `data/manifests/`.

Note: the NLTK Python *library* version is pinned in `requirements-research.txt`;
the separately-distributed `nltk_data` *resources* above are not
independently versioned by NLTK, so no specific resource version number is
claimed anywhere in this project's manifests.

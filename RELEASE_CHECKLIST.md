# Experimental Release Checklist

## Before publishing

- [ ] Replace `REPLACE-ME` repository URLs
- [ ] Confirm author/contact metadata
- [ ] Run `make test`
- [ ] Open the Word Explorer in a real browser and verify audio
- [ ] Confirm mapping checksum
- [ ] Review external dependency and data licenses
- [ ] Enable GitHub Actions
- [ ] Enable branch protection for `main`
- [ ] Require CI before merge
- [ ] Create tag `v0.1.0-experimental`
- [ ] Mark release as prerelease

## Release notes must state

- The mapping is experimental
- The project explores musicality rather than proving natural word melodies
- Analysis scores are heuristic and versioned
- Tone.js audio currently requires an internet connection

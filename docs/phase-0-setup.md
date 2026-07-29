# Phase 0 GitHub Setup

After creating an empty GitHub repository:

```bash
git init
git branch -M main
git add .
git commit -m "chore: establish LAPE-26 phase 0"
git remote add origin https://github.com/YOUR-ACCOUNT/lape-26.git
git push -u origin main
```

Then:

1. Replace `REPLACE-ME` in `CITATION.cff`.
2. Enable GitHub Actions.
3. Enable GitHub Pages with **GitHub Actions** as the source.
4. Protect `main` and require the `validate-and-test` check.
5. Add repository topics such as `music`, `language`, `creative-coding`, and `text-to-music`.
6. Run the experimental release checklist.

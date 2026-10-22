# Steam Game Map

A map of every Steam game with 10+ reviews (~56.7k). Search games you already like, combine
several, and get recommendations reranked by review quality and obscurity.

Idea: [@SyntheticBeef](https://x.com/SyntheticBeef/status/2112904858718367750) ·
map video: [@re_skob](https://x.com/re_skob/status/2113043120128483349).

## How it works

1. **Data**: `FronkonGames/steam-games-dataset` (MIT) — Steam store + SteamSpy, with user-tag vote counts.
2. **Vectors** (`build.py`):
   - vote-weighted user tags (+ genres) → TF-IDF → SVD 128
   - name + short description → `all-MiniLM-L6-v2` (384)
   - concatenated (tags weighted higher) → PCA 64 → int8 per game
3. **Map**: UMAP (cosine) of the full vector → 2D; region names come from k-means
   clusters labelled by their most over-represented tags.
4. **Recommendations** (in the browser, `site/app.js`):
   - **Blend**: cosine to the normalized average of your picks
   - **Like all**: soft-minimum of the similarity to each pick (must be close to every one)
   - **Like any**: maximum similarity to any pick
   - **Avoid** chips subtract a penalty for being close to that game
   - the 500 nearest games form the pool; the sliders rerank only inside it:
     `score = sim + quality·(wilson − 0.78)·0.7 + obscurity·(0.42 − popularity)·0.5`
     where `wilson` is the 95% lower bound of the positive-review share and
     `popularity` is log review count scaled to 0–1.
   - filters: min reviews, max price, release year.
   - state lives in the URL hash, so searches can be shared:
     `#like=367520,646570&mode=all&o=60`

## Run

```bash
python3 build.py                    # needs data/raw/games.json (970 MB) from the dataset
python3 -m http.server 8765 --directory site
```

The site is static: `site/index.html`, `style.css`, `app.js`, `data/meta.json`, `data/games.bin`.

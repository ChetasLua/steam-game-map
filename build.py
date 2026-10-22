"""Build the Steam game map: tag + description embeddings -> UMAP 2D -> compact site data.

Input:  data/raw/games.json  (FronkonGames/steam-games-dataset, MIT)
Output: site/data/meta.json, site/data/games.bin
"""
import json, math, re, struct, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent
RAW = ROOT / "data/raw/games.json"
OUT = ROOT / "site/data"
CACHE = ROOT / "data/cache"
MIN_REVIEWS = 10
DIM = 64

# Colour families: first family whose tag ranks highest in a game's tag votes wins.
FAMILIES = [
    ("Visual Novel", ["Visual Novel", "Dating Sim", "Otome", "Interactive Fiction", "Choose Your Own Adventure"]),
    ("Horror", ["Horror", "Psychological Horror", "Survival Horror"]),
    ("Shooter", ["FPS", "Shooter", "Third-Person Shooter", "Shoot 'Em Up", "Bullet Hell", "Twin Stick Shooter", "Arena Shooter"]),
    ("Strategy", ["Strategy", "RTS", "Turn-Based Strategy", "Grand Strategy", "4X", "Tower Defense", "Wargame", "Card Game", "Deckbuilding", "Board Game"]),
    ("RPG", ["RPG", "JRPG", "Action RPG", "CRPG", "Party-Based RPG", "Dungeon Crawler", "Roguelike", "Roguelite", "Souls-like"]),
    ("Simulation", ["Simulation", "Farming Sim", "Life Sim", "City Builder", "Management", "Automobile Sim", "Building", "Colony Sim", "Base Building", "Crafting", "Survival", "Sandbox"]),
    ("Puzzle", ["Puzzle", "Logic", "Hidden Object", "Point & Click", "Escape Room", "Word Game", "Match 3"]),
    ("Platformer", ["Platformer", "2D Platformer", "3D Platformer", "Precision Platformer", "Metroidvania", "Puzzle Platformer"]),
    ("Racing & Sports", ["Racing", "Sports", "Football (Soccer)", "Basketball", "Golf", "Driving", "Fishing"]),
    ("Action", ["Action", "Hack and Slash", "Fighting", "Beat 'em up", "Action-Adventure", "Arcade"]),
    ("Adventure", ["Adventure", "Exploration", "Walking Simulator", "Story Rich", "Mystery", "Detective"]),
    ("Casual", ["Casual", "Clicker", "Idle", "Cute", "Family Friendly", "Relaxing", "Party Game"]),
]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def tag_votes(g):
    t = g.get("tags")
    if isinstance(t, dict):
        return {k: float(v) for k, v in t.items() if v}
    if isinstance(t, list):  # unweighted fallback: rank-decayed votes
        return {k: 1.0 / (1 + i * 0.15) for i, k in enumerate(t)}
    return {}


def clean(s, n=600):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"&[a-z#0-9]+;", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:n]


def year_of(s):
    m = re.search(r"(19|20)\d\d", s or "")
    return int(m.group(0)) if m else 0


def wilson(pos, n, z=1.96):
    if n == 0:
        return 0.0
    p = pos / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def main():
    log("loading", RAW)
    raw = json.loads(RAW.read_text())
    log("games in dump:", len(raw))

    games = []
    for appid, g in raw.items():
        pos, neg = int(g.get("positive") or 0), int(g.get("negative") or 0)
        if pos + neg < MIN_REVIEWS or not g.get("name"):
            continue
        tv = tag_votes(g)
        base = max(tv.values(), default=0.0) or 1.0
        for gen in g.get("genres") or []:  # genres as weak tags, helps sparse-tag games
            tv[gen] = tv.get(gen, 0.0) + base * 0.25
        if not tv:
            continue
        desc = clean(g.get("short_description")) or clean(g.get("about_the_game"), 400)
        games.append(dict(
            appid=int(appid), name=g["name"].strip(), pos=pos, neg=neg, tv=tv,
            desc=desc, year=year_of(g.get("release_date")), price=float(g.get("price") or 0),
            dev=", ".join((g.get("developers") or [])[:2]),
        ))
    del raw
    n = len(games)
    log("kept", n, "games with >=", MIN_REVIEWS, "reviews and tags")

    # --- tag TF-IDF (vote-weighted) ---
    tag_df = Counter(t for g in games for t in g["tv"])
    tags = [t for t, c in tag_df.most_common() if c >= 5]
    tix = {t: i for i, t in enumerate(tags)}
    rows, cols, vals = [], [], []
    for r, g in enumerate(games):
        mx = max(g["tv"].values())
        for t, v in g["tv"].items():
            if t in tix:
                rows.append(r); cols.append(tix[t])
                vals.append(math.sqrt(v / mx) * math.log(n / tag_df[t]))
    T = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(tags)), dtype=np.float32)
    T = normalize(T)
    log("tag matrix", T.shape)
    tag_vec = TruncatedSVD(128, random_state=0).fit_transform(T)
    tag_vec = normalize(tag_vec)

    # --- description embeddings ---
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"text_{n}.npy"
    if cache.exists():
        txt = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
        texts = [f"{g['name']}. {g['desc']}" for g in games]
        log("embedding", n, "descriptions")
        txt = model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        np.save(cache, txt)
    log("text emb", txt.shape)

    # Tags carry genre/mechanics; text carries theme/setting. Tags dominate.
    combo = np.hstack([tag_vec * 0.8, txt * 0.6]).astype(np.float32)
    combo = normalize(combo)

    vec = PCA(DIM, random_state=0).fit_transform(combo)
    vec = normalize(vec).astype(np.float32)

    # --- 2D map ---
    import umap
    log("UMAP on", combo.shape)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.45, spread=1.4, metric="cosine", random_state=42,
                   low_memory=True).fit_transform(combo)
    xy = (xy - xy.mean(0)) / xy.std(0)
    log("UMAP done")

    # --- colour families ---
    fam_names = [f for f, _ in FAMILIES] + ["Other"]
    fam = np.zeros(n, np.uint8)
    for i, g in enumerate(games):
        best, bestv = len(FAMILIES), -1.0
        for fi, (_, members) in enumerate(FAMILIES):
            v = max((g["tv"].get(t, 0.0) for t in members), default=0.0)
            if v > bestv * 1.15:  # earlier families win ties
                best, bestv = fi, v
        fam[i] = best if bestv > 0 else len(FAMILIES)

    # --- region labels: k-means on the map, named by most distinctive tags ---
    K = 70
    km = KMeans(K, n_init=4, random_state=0).fit(xy)
    glob = np.asarray(T.mean(0)).ravel() + 1e-6
    regions = []
    for k in range(K):
        idx = np.where(km.labels_ == k)[0]
        if len(idx) < 60:
            continue
        loc = np.asarray(T[idx].mean(0)).ravel()
        lift = loc * np.log(loc / glob + 1)
        top = [tags[j] for j in np.argsort(-lift)[:6] if tags[j] not in ("Indie", "Singleplayer", "Casual", "Action", "Adventure")][:2]
        cx, cy = xy[idx].mean(0)
        regions.append(dict(x=round(float(cx), 3), y=round(float(cy), 3), n=int(len(idx)), label=" · ".join(top)))

    # --- per-game top tags (by votes) ---
    top_tags = np.full((n, 6), 0xFFFF, np.uint16)
    for i, g in enumerate(games):
        tt = [t for t, _ in sorted(g["tv"].items(), key=lambda kv: -kv[1]) if t in tix][:6]
        top_tags[i, :len(tt)] = [tix[t] for t in tt]

    # --- write ---
    OUT.mkdir(parents=True, exist_ok=True)
    pos = np.array([g["pos"] for g in games], np.uint32)
    neg = np.array([g["neg"] for g in games], np.uint32)
    year = np.array([g["year"] for g in games], np.uint16)
    price = np.array([min(65535, round(g["price"] * 100)) for g in games], np.uint16)
    q8 = np.clip(np.round(vec * 127), -127, 127).astype(np.int8)

    with open(OUT / "games.bin", "wb") as f:
        f.write(struct.pack("<4sII", b"SGM1", n, DIM))
        for a in (xy.astype(np.float32), pos, neg, year, price, top_tags, fam, q8):
            b = np.ascontiguousarray(a).tobytes()
            f.write(b)
            f.write(b"\0" * ((-len(b)) % 4))  # keep 4-byte alignment for typed arrays

    meta = dict(
        n=n, dim=DIM, tags=tags, families=fam_names, regions=regions,
        appid=[g["appid"] for g in games], name=[g["name"] for g in games],
        source="FronkonGames/steam-games-dataset (MIT), Steam + SteamSpy",
    )
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    # descriptions only feed the hover card, so they load after the map is up
    (OUT / "desc.json").write_text(json.dumps({"desc": [g["desc"][:220] for g in games]}, ensure_ascii=False, separators=(",", ":")))
    log("wrote", OUT / "games.bin", (OUT / "games.bin").stat().st_size, "bytes;",
        "meta.json", (OUT / "meta.json").stat().st_size, "bytes")

    # sanity: nearest neighbours of a few anchors in the 64-d space
    names = {g["name"]: i for i, g in enumerate(games)}
    for probe in ["Stardew Valley", "Hollow Knight", "Factorio", "Slay the Spire", "Outer Wilds"]:
        if probe in names:
            s = vec @ vec[names[probe]]
            nn = [games[j]["name"] for j in np.argsort(-s)[1:7]]
            log(probe, "->", nn)


if __name__ == "__main__":
    main()

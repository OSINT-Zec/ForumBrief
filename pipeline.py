import os, re, json, time, argparse, unicodedata, datetime as dt
from pathlib import Path
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# --------- utils ----------
def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path("out").mkdir(exist_ok=True)

def normalize(txt: str) -> str:
    import regex as re
    if not txt: return ""
    txt = unicodedata.normalize("NFKC", txt).strip().lower()
    txt = re.sub(r"https?://\S+", " ", txt)
    txt = re.sub(r"^edit:.*$", " ", txt, flags=re.MULTILINE)
    txt = re.sub(r"`{1,3}.*?`{1,3}", " ", txt, flags=re.DOTALL)
    txt = re.sub(r"\s+", " ", txt)
    return txt

# --------- collect ----------
def cmd_collect(args):
    load_dotenv()
    cfg = load_cfg(args.config)
    ensure_dirs()

    bag = []
    since = int((dt.datetime.utcnow() - dt.timedelta(days=30*cfg["time_window_months"])).timestamp())

    # --- Reddit via PRAW ---
    import praw
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT","one-time-topic-harvest/0.1")
    )

    def build_query(must, must_not):
        q = " ".join([f'({m})' for m in must or []])
        if must_not:
            q += " " + " ".join([f'-{x}' for x in must_not])
        return q

    q = build_query(cfg["reddit_query"]["must"], cfg["reddit_query"]["must_not"])
    min_score = int(cfg["reddit_query"].get("min_score", 10))
    min_comments = int(cfg["reddit_query"].get("min_comments", 3))
    max_posts = int(cfg["max_posts"])

    for sr in cfg["subreddits"]:
        sub = reddit.subreddit(sr)
        for post in sub.search(q, sort="top", time_filter="all", limit=None):
            if post.created_utc < since:
                continue
            if post.score < min_score or post.num_comments < min_comments:
                continue
            try:
                post.comments.replace_more(limit=0)
                top_comments = [c.body for c in post.comments[:cfg["grab_top_n_comments"]]]
            except Exception:
                top_comments = []
            bag.append({
                "source":"reddit",
                "subreddit":sr,
                "title":post.title or "",
                "selftext":post.selftext or "",
                "comments":top_comments,
                "score":int(post.score),
                "num_comments":int(post.num_comments),
                "url":f"https://www.reddit.com{post.permalink}",
                "created_utc":float(post.created_utc),
                "author":str(post.author) if post.author else "deleted"
            })
            if len(bag) >= max_posts:
                break
        time.sleep(0.8)

    # --- StackExchange (optional) ---
    if cfg.get("stackexchange_sites"):
        import requests
        se_key = os.getenv("STACKEXCHANGE_KEY","")
        for site in cfg["stackexchange_sites"]:
            params = {
                "order":"desc","sort":"votes","site":site,"pagesize":50,
                "filter":"withbody","key":se_key
            }
            for query in cfg["stackexchange_query"].get("intitle", []):
                page=1
                while True:
                    resp = requests.get("https://api.stackexchange.com/2.3/search/advanced",
                                        params={**params, "intitle":query, "page":page})
                    if resp.status_code!=200: break
                    data = resp.json()
                    for item in data.get("items", []):
                        if item.get("score",0) < cfg["stackexchange_query"].get("min_score",1):
                            continue
                        body = item.get("body","")
                        bag.append({
                            "source":"stackexchange",
                            "site":site,
                            "title": item.get("title",""),
                            "selftext": body,
                            "comments": [],
                            "score": item.get("score",0),
                            "num_comments": item.get("answer_count",0),
                            "url": item.get("link",""),
                            "created_utc": 0,
                            "author": item.get("owner",{}).get("display_name","")
                        })
                    if not data.get("has_more"): break
                    page += 1
                    time.sleep(0.3)

    with open(args.out, "w") as f:
        for x in bag:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[collect] saved {len(bag)} rows → {args.out}")

# --------- clean ----------
def cmd_clean(args):
    ensure_dirs()
    out = open(args.out,"w")
    kept=0
    with open(args.inp) as f:
        for line in f:
            o = json.loads(line)
            blob = " ".join([o.get("title",""), o.get("selftext","")] + o.get("comments",[]))
            blob = normalize(blob)
            if len(blob) < 280:   # configurable minimal useful length
                continue
            o["blob"] = blob
            out.write(json.dumps(o, ensure_ascii=False) + "\n")
            kept+=1
    out.close()
    print(f"[clean] kept {kept} docs → {args.out}")

# --------- dedupe ----------
def cmd_dedupe(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [json.loads(l) for l in open(args.inp)]
    texts = [d["blob"] for d in docs]
    if not docs:
        Path(args.out).write_text("")
        print("[dedupe] no docs")
        return
    tfidf = TfidfVectorizer(max_df=0.6, min_df=3).fit_transform(texts)
    sim = cosine_similarity(tfidf, dense_output=False)
    keep, removed = [], set()
    for i in range(sim.shape[0]):
        if i in removed: continue
        dup_idx = sim[i].nonzero()[1]
        dup_idx = [j for j in dup_idx if j>i and sim[i,j] >= 0.85]
        for j in dup_idx: removed.add(j)
        keep.append(i)
    kept_docs = [docs[i] for i in keep]
    with open(args.out,"w") as f:
        for x in kept_docs:
            f.write(json.dumps(x, ensure_ascii=False)+"\n")
    print(f"[dedupe] {len(kept_docs)} kept / {len(docs)} input → {args.out}")

# --------- cluster ----------
def cmd_cluster(args):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import CountVectorizer

    cfg = load_cfg("config.yaml")
    items = [json.loads(l) for l in open(args.inp)]
    texts = [it["blob"] for it in items]
    if not items:
        Path(args.out).write_text("{}"); print("[cluster] no docs"); return

    model_name = cfg["clustering"]["model"]
    th = float(cfg["clustering"]["distance_threshold"])
    print(f"[cluster] encoding with {model_name} ...")
    st = SentenceTransformer(model_name)
    emb = st.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    print(f"[cluster] Agglomerative threshold={th}")
    clu = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric="cosine", linkage="average")
    labels = clu.fit_predict(emb)
    uniq = sorted(set(labels))

    def top_terms(idxs):
        """Return up to 10 salient 1–2 gram terms for a cluster, robust to tiny clusters."""
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np, re
        from collections import Counter

        cluster_texts = [texts[i] for i in idxs]
        # Adaptive min_df: tiny clusters → 1, otherwise 2
        min_df = 1 if len(cluster_texts) < 5 else 2

        # Try vectorizer first
        vec = CountVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.9,
            stop_words="english",
            strip_accents="unicode",
            max_features=2000,
        )
        try:
            X = vec.fit_transform(cluster_texts)
            if X.shape[1] > 0:
                freqs = np.asarray(X.sum(axis=0)).ravel()
                terms = np.array(vec.get_feature_names_out())
                top = [t for t, _ in sorted(zip(terms, freqs), key=lambda x: -x[1])[:10]]
                return top
        except ValueError:
            pass  # fall through to manual fallback

        # Fallback: simple token freq (keeps ASCII + German letters)
        tokens = []
        for t in cluster_texts:
            tokens += re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", t)
        counter = Counter([w.lower() for w in tokens if len(w) >= 3])
        # remove a few generic words that slip past english stoplist
        boring = {"people","thing","things","time","year","years","day","days",
                  "life","work","job","jobs","city","cities","germany","german",
                  "berlin","move","moved","moving"}
        terms = [w for w, _ in counter.most_common(30) if w not in boring][:10]
        return terms or ["general"]

    # make theme patterns case-insensitive
    theme_map = cfg["themes"]
    theme_regex = {k: re.compile(v, flags=re.IGNORECASE) for k, v in theme_map.items()}

    def assign_theme(terms):
        bag = " ".join(terms)
        for theme, pat in theme_regex.items():
            if pat.search(bag):
                return theme
        return "other"

    clusters = {}
    for cid in uniq:
        idxs = [i for i,l in enumerate(labels) if l==cid]
        terms = top_terms(idxs)
        theme = assign_theme(terms)
        clusters[str(cid)] = {"idxs": idxs, "top_terms": terms, "theme": theme}

    json.dump(clusters, open(args.out,"w"), ensure_ascii=False, indent=2)
    print(f"[cluster] {len(uniq)} clusters → {args.out}")

# --------- summarize ----------
def cmd_summarize(args):
    """
    Summarize clusters into compact, actionable bullets using OpenAI.
    - Loads OPENAI_API_KEY from .env automatically.
    - Skips tiny clusters (cluster_min_size) and caps total (max_clusters).
    - Ranks clusters by (size, avg engagement) before summarizing.
    """
    import os, json, time
    import numpy as np
    from tqdm import tqdm
    from dotenv import load_dotenv
    from openai import OpenAI
    from openai import OpenAIError

    # --- setup & config ---
    load_dotenv()
    cfg = load_cfg("config.yaml")
    topic = cfg.get("topic", "[TOPIC]")
    s_cfg = cfg.get("summarization", {}) or {}
    model = s_cfg.get("model", "gpt-4o-mini")
    per_item_bullets = int(s_cfg.get("per_item_bullets", 2))
    reduce_bullets   = int(s_cfg.get("reduce_bullets", 6))
    temperature      = float(s_cfg.get("temperature", 0.2))
    cluster_min_size = int(s_cfg.get("cluster_min_size", 3))
    max_clusters     = int(s_cfg.get("max_clusters", 150))
    per_cluster_k    = int(s_cfg.get("per_cluster_representatives", 12))  # optional knob

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env or export it in your shell.")
    client = OpenAI(api_key=api_key)

    # --- load data ---
    items = [json.loads(l) for l in open(args.inp)]
    clusters = json.load(open(args.clusters))
    if not items or not clusters:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({}, open(args.out, "w"))
        print("[summarize] no items or clusters; wrote empty output")
        return

    # --- helpers ---
    def pick_representatives(idxs, k=per_cluster_k):
        # top by (score, num_comments) as a cheap proxy for signal
        srt = sorted(
            idxs,
            key=lambda i: (items[i].get("score", 0), items[i].get("num_comments", 0)),
            reverse=True,
        )
        return [items[i] for i in srt[:k]]

    def cluster_stats(idxs):
        size = len(idxs)
        if size == 0:
            return (0, 0.0)
        eng = [(items[i].get("score", 0) + items[i].get("num_comments", 0)) for i in idxs]
        avg_eng = float(sum(eng)) / size if size else 0.0
        return (size, avg_eng)

    # order by (size, avg engagement), filter by min size, cap to max_clusters
    ordered = sorted(
        clusters.items(),
        key=lambda kv: cluster_stats(kv[1]["idxs"]),
        reverse=True,
    )
    target = [(cid, meta) for cid, meta in ordered if len(meta["idxs"]) >= cluster_min_size][:max_clusters]
    if not target:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({}, open(args.out, "w"))
        print("[summarize] nothing passed filters; wrote empty output")
        return

    def map_item_to_bullets(txt, theme):
        prompt = (
            f"You are compressing forum tips about '{topic}', theme='{theme}'.\n"
            f"Produce {per_item_bullets} concise, highly actionable bullets from the text.\n"
            "Avoid redundancy and fluff. Use imperative voice. No preambles."
        )
        # trim to keep request small; chars≈tokens/≈4 is fine as a heuristic
        payload = txt[:5000]
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt + "\n\nTEXT:\n" + payload}],
        ).choices[0].message.content

    def reduce_bullets(joined, theme):
        prompt = (
            f"Merge and deduplicate all bullets about '{topic}', theme='{theme}'.\n"
            f"Return exactly {reduce_bullets} surgical, actionable tips in bullet form.\n"
            "Rank each with [F?/U?] where F=frequency (1–5) and U=usefulness (1–5).\n"
            "End with a short 'Mistakes to Avoid' list. No preambles."
        )
        return client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt + "\n\nBULLETS:\n" + joined[:12000]}],
        ).choices[0].message.content

    # --- main loop with light backoff on rate limits ---
    out = {}
    for cid, meta in tqdm(target, desc="[summarize] clusters"):
        reps = pick_representatives(meta["idxs"])
        theme = meta.get("theme", "other")
        maps = []
        for it in reps:
            txt = it.get("blob", "")
            if not txt:
                continue
            for attempt in range(3):
                try:
                    maps.append(map_item_to_bullets(txt, theme))
                    break
                except OpenAIError as e:
                    # light exponential backoff (rate limit/transient)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise
        joined = "\n".join(maps)
        # reduce
        for attempt in range(3):
            try:
                reduced = reduce_bullets(joined, theme)
                break
            except OpenAIError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

        out[cid] = {
            "theme": theme,
            "top_terms": meta.get("top_terms", []),
            "size": len(meta.get("idxs", [])),
            "summary": reduced,
        }

    # --- write output ---
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[summarize] wrote {args.out} ({len(out)} clusters)")

# --------- export ----------
def cmd_export(args):
    import pandas as pd
    cluster_summ = json.load(open(args.cluster_summ))
    clusters = json.load(open(args.clusters))
    items = [json.loads(l) for l in open(args.inp)]

    # Build simple CSV of “tips” by parsing lines that look like bullets
    rows = []
    for cid, meta in cluster_summ.items():
        theme = meta["theme"]
        text = meta["summary"]
        for line in text.splitlines():
            line=line.strip()
            if not line or not (line.startswith("-") or line.startswith("*")):
                continue
            tip = line.lstrip("-* ").strip()
            rows.append({"theme": theme, "tip": tip, "cluster_id": cid})

    df = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(exist_ok=True)
    df.to_csv(args.csv, index=False)

    # Markdown report
    md = ["# Forum Brief — One-time Summary\n"]
    themes = {}
    for cid, meta in cluster_summ.items():
        themes.setdefault(meta["theme"], []).append((cid, meta))
    for theme, lst in sorted(themes.items(), key=lambda x: x[0]):
        md.append(f"## {theme.title()}\n")
        for cid, meta in lst:
            md.append(f"**Cluster {cid}** — top terms: *{', '.join(meta['top_terms'][:8])}*\n\n")
            md.append(meta["summary"]+"\n\n")
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"[export] wrote {args.md} and {args.csv}")

# --------- main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("collect"); s.add_argument("--config", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("clean");   s.add_argument("--inp", required=True);   s.add_argument("--out", required=True)
    s = sub.add_parser("dedupe");  s.add_argument("--inp", required=True);   s.add_argument("--out", required=True)
    s = sub.add_parser("cluster"); s.add_argument("--inp", required=True);   s.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--inp", required=True); s.add_argument("--clusters", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("export")
    s.add_argument("--cluster_summ", required=True); s.add_argument("--clusters", required=True)
    s.add_argument("--inp", required=True); s.add_argument("--md", required=True); s.add_argument("--csv", required=True)

    args = p.parse_args()
    if args.cmd=="collect": cmd_collect(args)
    elif args.cmd=="clean": cmd_clean(args)
    elif args.cmd=="dedupe": cmd_dedupe(args)
    elif args.cmd=="cluster": cmd_cluster(args)
    elif args.cmd=="summarize": cmd_summarize(args)
    elif args.cmd=="export": cmd_export(args)
    else:
        print("Use a valid subcommand (collect|clean|dedupe|cluster|summarize|export)")

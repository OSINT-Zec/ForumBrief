# Forum Brief — One-Time Summarization Pipeline

This project collects **Reddit / StackExchange** discussions on a topic, cleans and deduplicates them, clusters posts by theme, and produces a **ranked, summarized brief** you can read in one sitting.

---

## Features

* One-time data grab (not continuous monitoring)
* Deduplicates and clusters similar advice
* Ranks by **frequency** and **usefulness**
* Outputs **Markdown summary** + **CSV of tips**
* Configurable keywords, subreddits, and cluster sizes
* Uses OpenAI GPT models for final summarization

---

## Requirements

* Python 3.11+
* Reddit API credentials
* OpenAI API key (for summarization step)

Install everything with:

```bash
make install
```

---

## Quickstart

1. **Set environment variables**

Copy the template and add your keys:

```bash
cp .env.example .env
```

Edit `.env` to include:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

---

2. **Configure topic & subreddits**

Edit `config.yaml`:

```yaml
topic: "moving to Germany"
subreddits:
  - germany
  - berlin
  - expats
  - IWantOut
time_window_months: 24
max_posts: 1800
```

---

3. **Run the full pipeline**

```bash
make run
```

This runs:

* `make collect` → grab Reddit/StackExchange posts
* `make clean` → normalize text, strip noise
* `make dedupe` → remove near-duplicates
* `make cluster` → embed & cluster posts
* `make summarize` → map/reduce summarization with GPT
* `make export` → produce final Markdown + CSV

---

4. **Outputs**

* `out/summary.md` → human-readable report by theme/cluster
* `out/tips.csv` → all distilled tips with scores

Open `summary.md` in any Markdown viewer for the final brief.

---

## Useful Make Commands

| Command          | What it does                           |
| ---------------- | -------------------------------------- |
| `make run`       | Run entire pipeline (collect → export) |
| `make collect`   | Fetch posts/comments only              |
| `make clean`     | Normalize, strip noise                 |
| `make dedupe`    | Remove duplicates                      |
| `make cluster`   | Cluster and label posts                |
| `make summarize` | Summarize clusters with OpenAI         |
| `make export`    | Write `summary.md` + `tips.csv`        |
| `make wipe`      | Delete all data/output artifacts       |

---

## Configurable knobs

* `distance_threshold` → larger = fewer, bigger clusters
* `cluster_min_size` → skip very small clusters
* `max_clusters` → limit how many clusters to summarize
* `per_item_bullets` / `reduce_bullets` → bullet length

---

## Example Output Snippet

```markdown
## Housing — Cluster 4
- Always request a **Schufa** before signing a rental contract [F:5, U:5]
- Start flat-hunting **2–3 months early**, competition is tough [F:4, U:5]
- Use **WG-Gesucht** and **ImmoScout** as main portals [F:4, U:4]

Mistakes to Avoid:
- Paying deposits before visiting in person
- Ignoring neighborhood commute times
```

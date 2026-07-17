# Clippie Court Search

**Semantic highlight search for youth basketball, powered by Qdrant.**

Sports World Cup Hackathon 2026 -- built on top of [Clippie](https://clippie.app), a computer-vision pipeline that watches youth basketball game film, detects made baskets with a custom-trained YOLO model, and auto-cuts highlight clips.

## What it does

Type a play in plain English -- "clutch three pointer in a close game", "steal that led to a bucket" -- and get back **real game clips**, ranked by meaning, not keywords. Every result is an actual highlight cut by Clippie's CV pipeline from real 2026 grassroots games.

## How Qdrant is used

1. `build_index.py` pulls 1,750+ attributed highlight clips from Clippie's production Firestore (player, jersey number, play type, matchup, live score at the moment of the clip).
2. Each clip's game context is rendered into a natural-language sentence, including derived context like "close game, clutch moment" when the score margin is tight.
3. Sentences are embedded with OpenAI `text-embedding-3-small` (1536-dim) and upserted into a **Qdrant** collection (cosine distance), with the full clip payload (video URL, thumbnail, score snapshot) stored alongside the vector.
4. `serve.py` embeds each incoming query and runs a Qdrant vector search; the UI renders ranked clips with thumbnails that link straight to the video.

This is the "multi-modal sports data + video retrieval" use case end to end: CV events in, semantic video retrieval out.

## Run it

```bash
docker run -d -p 6333:6333 qdrant/qdrant
python3 build_index.py   # Firestore -> embeddings -> Qdrant
python3 serve.py         # UI + /api/search on :8787
```

## Stack

- **Qdrant** -- vector store + search (Docker, REST API, zero client deps)
- OpenAI text-embedding-3-small -- embeddings
- Custom YOLO v2 model (rim/backboard/net/ball) -- the upstream CV pipeline that creates the clips
- Firebase/Firestore -- Clippie's production clip store
- Python stdlib server + single-page UI

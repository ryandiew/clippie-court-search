#!/usr/bin/env python3
"""Clippie Court Search -- index builder.

Pulls real highlight clips from Clippie's Firestore, turns each clip's game
context into a natural-language description, embeds it (OpenAI
text-embedding-3-small), and upserts everything into a Qdrant collection so
plays are searchable by meaning, not just exact fields.
"""
import json
import os
import urllib.request

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/firebase/service-account.json"),
)
from google.cloud import firestore  # noqa: E402

QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "clippie_clips"
EMBED_MODEL = "text-embedding-3-small"
DIM = 1536

EVENT_LABELS = {
    "1g": "made free throw",
    "2g": "made two point basket",
    "3g": "made three pointer",
    "3PT": "made three pointer",
    "2PT": "made two point basket",
    "FT": "made free throw",
    "STL": "steal",
    "BLK": "block",
    "AST": "assist",
    "REB": "rebound",
    "DUNK": "dunk",
}


def openai_key():
    for line in open(os.path.expanduser("~/.openclaw-ryanos/.env")):
        if line.startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no OPENAI_API_KEY")


def http_json(url, payload=None, headers=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def embed(texts, key):
    out = http_json(
        "https://api.openai.com/v1/embeddings",
        {"model": EMBED_MODEL, "input": texts},
        {"Authorization": f"Bearer {key}"},
    )
    return [d["embedding"] for d in out["data"]]


def clip_text(d):
    """One natural-language sentence per clip -- this is what gets embedded."""
    ev = EVENT_LABELS.get((d.get("eventType") or "").strip(), (d.get("eventType") or "").strip())
    player = (d.get("playerName") or "").strip()
    num = (d.get("playerNumber") or "").strip()
    team = (d.get("teamName") or d.get("homeTeamName") or "").strip()
    home, away = (d.get("homeTeamName") or "").strip(), (d.get("awayTeamName") or "").strip()
    hs, as_ = str(d.get("homeScoreAtClip") or ""), str(d.get("awayScoreAtClip") or "")

    bits = []
    who = " ".join(x for x in [player, f"#{num}" if num else ""] if x).strip()
    bits.append(f"{who or 'A player'} {ev or 'highlight play'}")
    if team:
        bits.append(f"for {team}")
    if home and away:
        bits.append(f"in {home} vs {away}")
    if hs and as_:
        bits.append(f"with the score {hs}-{as_}")
        try:
            if abs(int(hs) - int(as_)) <= 4:
                bits.append("in a close game, clutch moment")
        except ValueError:
            pass
    cap = (d.get("caption") or "").strip()
    if cap and cap.lower() != "highlight":
        bits.append(f'-- "{cap}"')
    return " ".join(bits)


def main():
    key = openai_key()
    db = firestore.Client(project="scdemo-c11e0")

    clips = []
    for doc in db.collection("highlights").stream():
        d = doc.to_dict() or {}
        if not d.get("videoURL"):
            continue
        # Rich docs first: named player or typed event makes a searchable clip
        if not ((d.get("playerName") or "").strip() or (d.get("eventType") or "").strip()):
            continue
        clips.append(
            {
                "id": doc.id,
                "text": clip_text(d),
                "payload": {
                    "player": (d.get("playerName") or "").strip(),
                    "number": (d.get("playerNumber") or "").strip(),
                    "event": (d.get("eventType") or "").strip(),
                    "team": (d.get("teamName") or "").strip(),
                    "matchup": f"{d.get('homeTeamName','')} vs {d.get('awayTeamName','')}".strip(" vs"),
                    "score": f"{d.get('homeScoreAtClip','')}-{d.get('awayScoreAtClip','')}".strip("-"),
                    "video": d.get("videoURL"),
                    "thumb": d.get("thumbnailURL") or "",
                    "clipId": doc.id,
                },
            }
        )
    print(f"clips with player/event context: {len(clips)}")
    if not clips:
        raise SystemExit("nothing to index")

    # Fresh collection
    try:
        http_json(f"{QDRANT}/collections/{COLLECTION}", method="DELETE")
    except Exception:
        pass
    http_json(
        f"{QDRANT}/collections/{COLLECTION}",
        {"vectors": {"size": DIM, "distance": "Cosine"}},
        method="PUT",
    )

    for i in range(0, len(clips), 128):
        batch = clips[i : i + 128]
        vecs = embed([c["text"] for c in batch], key)
        points = [
            {"id": i + j + 1, "vector": vecs[j], "payload": {**batch[j]["payload"], "text": batch[j]["text"]}}
            for j in range(len(batch))
        ]
        http_json(f"{QDRANT}/collections/{COLLECTION}/points?wait=true", {"points": points}, method="PUT")
        print(f"indexed {min(i+128, len(clips))}/{len(clips)}")

    info = http_json(f"{QDRANT}/collections/{COLLECTION}")
    print("qdrant points:", info["result"]["points_count"])


if __name__ == "__main__":
    main()

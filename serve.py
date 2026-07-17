#!/usr/bin/env python3
"""Clippie Court Search -- demo server.

GET /               -> search UI
GET /api/search?q=  -> embeds the query, vector-searches Qdrant, returns clips
"""
import json
import os
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "clippie_clips"
EMBED_MODEL = "text-embedding-3-small"
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def openai_key():
    for line in open(os.path.expanduser("~/.openclaw-ryanos/.env")):
        if line.startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no OPENAI_API_KEY")


KEY = openai_key()


def http_json(url, payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


import re

# Play-type language -> canonical event value stored in the payload.
PLAY_WORDS = [
    (("three", "threes", "3pt", "3-pt", "3 point", "triple", "from deep", "downtown"), "3PT Make"),
    (("free throw", "free throws", "foul shot", "from the line", "and one"), "Free Throw"),
    (("steal", "steals", "takeaway", "takeaways"), "Steal"),
    (("assist", "assists", "dime", "dimes", "dish"), "Assist"),
    (("block", "blocks", "swat", "rejection"), "Block"),
    (("rebound", "rebounds", "board", "boards"), "Rebound"),
    (("layup", "layups", "at the rim", "finish", "finishes", "two pointer", "2pt"), "2PT Make"),
]

# Game-state language -> computed payload tag (go_ahead / close / blowout).
MOMENT_WORDS = [
    (("go ahead", "go-ahead", "goahead", "go aheads", "lead change", "lead changing", "took the lead", "take the lead", "lead taking", "go ahead bucket"), "go_ahead"),
    (("clutch", "close game", "close games", "tie game", "tight game", "down to the wire", "nail biter", "buzzer"), "close"),
    (("blowout", "blow out", "garbage time"), "blowout"),
]

# Loaded once at startup: every team name in the collection, for query matching.
TEAMS = []          # [(match_key_lower, canonical_team)]


def age_strip(t):
    """'Bay City 17u' -> 'bay city' so a coach typing the short name still hits."""
    return re.sub(r"\b\d{1,2}u\b|\b\d{2}['’]?\b", "", t, flags=re.I).strip().lower()


def load_teams():
    names = set()
    nxt = None
    while True:
        body = {"limit": 512, "with_payload": ["team"], "with_vector": False}
        if nxt:
            body["offset"] = nxt
        r = http_json(f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)["result"]
        for p in r["points"]:
            t = (p["payload"].get("team") or "").strip()
            if t:
                names.add(t)
        nxt = r.get("next_page_offset")
        if not nxt:
            break
    # Exact team names win; add age-stripped aliases only where they don't collide
    # (so "NBBA" -> NBBA and "NBBA 15u" -> NBBA 15u, not one clobbering the other).
    seen = {}
    for t in names:
        seen[t.lower()] = t
    for t in names:
        a = age_strip(t)
        if a and a not in seen:
            seen[a] = t
    # longest keys first so "team cali 16u" wins over "team cali"
    return sorted(([k, v] for k, v in seen.items() if k), key=lambda kv: -len(kv[0]))


def parse(q):
    """Pull structured filters (team / jersey # / play type) out of a plain query.
    Returns (qdrant_filter, residual_text_for_semantic_rank)."""
    ql = " " + q.lower() + " "
    must = []
    used = []

    for key, team in TEAMS:
        if key and (" " + key + " ") in ql:
            must.append({"key": "team", "match": {"value": team}})
            used.append(key)
            ql = ql.replace(" " + key + " ", " ")
            break

    m = re.search(r"#\s*(\d{1,3})|\b(?:number|no|jersey)\s+(\d{1,3})\b", ql)
    if m:
        num = m.group(1) or m.group(2)
        must.append({"key": "number", "match": {"value": num}})
        ql = ql[: m.start()] + " " + ql[m.end():]

    for words, ev in PLAY_WORDS:
        if any((" " + w + " ") in ql for w in words):
            must.append({"key": "event", "match": {"value": ev}})
            for w in words:
                ql = ql.replace(" " + w + " ", " ")
            break

    # Game-state moments computed from the score sequence and stored as payload tags.
    for words, tag in MOMENT_WORDS:
        if any((" " + w + " ") in ql for w in words):
            must.append({"key": tag, "match": {"value": True}})
            for w in words:
                ql = ql.replace(" " + w + " ", " ")
            break

    residual = re.sub(r"\s+", " ", ql).strip(" -")
    # strip filler that carries no meaning once filters are pulled
    residual = re.sub(r"\b(show me|all|every|clips? of|highlights?|by|for|the|a|an|of)\b", " ", residual)
    residual = re.sub(r"\s+", " ", residual).strip()
    return ({"must": must} if must else None), residual


def embed(text):
    return http_json(
        "https://api.openai.com/v1/embeddings",
        {"model": EMBED_MODEL, "input": [text]},
        {"Authorization": f"Bearer {KEY}"},
    )["data"][0]["embedding"]


# Highlights only -- never surface misses or bare attempts.
NOT_HIGHLIGHT = [{"key": "event", "match": {"value": v}} for v in ("2PT Miss", "3PT Miss", "Shot Attempt")]


def with_filter(filt):
    f = dict(filt) if filt else {}
    f["must_not"] = NOT_HIGHLIGHT
    return f


def search(q, limit=24):
    filt, residual = parse(q)
    if residual:  # semantic component -> filtered vector search
        body = {"vector": embed(residual), "limit": limit, "with_payload": True, "filter": with_filter(filt)}
        res = http_json(f"{QDRANT}/collections/{COLLECTION}/points/search", body)["result"]
        # Trim the long tail: keep the natural cluster near the top match so a name
        # search returns THAT player, not a padded list of loosely-related clips.
        if res:
            top = res[0]["score"]
            # Relative cluster cut only -- no absolute floor (which zeroed out weak
            # generic residuals like "buckets"); keeps a name search tight but never empty.
            cut = top * 0.80
            res = [p for p in res if p["score"] >= cut]
        return [{**p["payload"], "sim": round(p["score"], 3)} for p in res]
    # pure structured lookup (e.g. "NBBA threes") -> filtered scroll, no vector needed
    body = {"limit": limit, "with_payload": True, "with_vector": False, "filter": with_filter(filt)}
    res = http_json(f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)["result"]
    return [{**p["payload"], "sim": None} for p in res["points"]]


# ---- Browse: teams -> roster -> clips (how a coach or parent actually navigates) ----
BROWSE = {"teams": [], "rosters": {}}  # built once at startup from the clean Qdrant set


def build_browse():
    from collections import defaultdict
    pts = []
    nxt = None
    while True:
        body = {"limit": 512, "with_payload": True, "with_vector": False}
        if nxt:
            body["offset"] = nxt
        r = http_json(f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)["result"]
        pts += r["points"]
        nxt = r.get("next_page_offset")
        if not nxt:
            break
    team_ct = defaultdict(int)
    rosters = defaultdict(lambda: defaultdict(lambda: {"name": "", "number": "", "count": 0}))
    for p in pts:
        d = p["payload"]
        team = (d.get("team") or "").strip()
        if not team:
            continue
        team_ct[team] += 1
        name = (d.get("player") or "").strip()
        num = str(d.get("number") or "").strip()
        if not (name or num):
            continue
        key = name or ("#" + num)
        e = rosters[team][key]
        e["name"] = name or e["name"]
        e["number"] = num or e["number"]
        e["count"] += 1
    teams = sorted(({"team": t, "count": c} for t, c in team_ct.items()), key=lambda x: -x["count"])
    rout = {}
    for t, players in rosters.items():
        rout[t] = sorted(players.values(), key=lambda x: -x["count"])
    return {"teams": teams, "rosters": rout}


def clips_for(team=None, number=None, name=None, event=None, moment=None, limit=60):
    must = []
    if team:
        must.append({"key": "team", "match": {"value": team}})
    if number:
        must.append({"key": "number", "match": {"value": str(number)}})
    if name:
        must.append({"key": "player", "match": {"value": name}})
    if event:
        must.append({"key": "event", "match": {"value": event}})
    if moment in ("go_ahead", "close", "blowout"):
        must.append({"key": moment, "match": {"value": True}})
    body = {"limit": limit, "with_payload": True, "with_vector": False}
    if must:
        body["filter"] = {"must": must}
    res = http_json(f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)["result"]
    return [p["payload"] for p in res["points"]]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        if u.path == "/api/teams":
            return self._send(200, json.dumps({"teams": BROWSE["teams"]}))
        if u.path == "/api/roster":
            team = qs.get("team", [""])[0]
            return self._send(200, json.dumps({"team": team, "players": BROWSE["rosters"].get(team, [])}))
        if u.path == "/api/clips":
            try:
                clips = clips_for(
                    team=qs.get("team", [None])[0],
                    number=qs.get("number", [None])[0],
                    name=qs.get("name", [None])[0],
                    event=qs.get("event", [None])[0],
                    moment=qs.get("moment", [None])[0],
                )
                return self._send(200, json.dumps({"results": clips}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))
        if u.path == "/api/search":
            q = qs.get("q", [""])[0].strip()
            if not q:
                return self._send(400, json.dumps({"error": "missing q"}))
            try:
                return self._send(200, json.dumps({"query": q, "results": search(q)}))
            except Exception as e:
                return self._send(500, json.dumps({"error": str(e)}))
        if u.path in ("/", "/index.html"):
            with open(os.path.join(WEB, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        # static assets from web/ (appicon, etc.) -- basename only, no traversal
        name = os.path.basename(u.path.lstrip("/"))
        fp = os.path.join(WEB, name)
        if name and os.path.isfile(fp):
            ctype = "image/png" if name.endswith(".png") else "application/octet-stream"
            with open(fp, "rb") as f:
                return self._send(200, f.read(), ctype)
        return self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    TEAMS = load_teams()
    BROWSE = build_browse()
    print(f"loaded {len(TEAMS)} team match-keys, {len(BROWSE['teams'])} teams")
    print(f"court-search on :{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

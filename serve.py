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


def search(q, limit=12):
    vec = http_json(
        "https://api.openai.com/v1/embeddings",
        {"model": EMBED_MODEL, "input": [q]},
        {"Authorization": f"Bearer {KEY}"},
    )["data"][0]["embedding"]
    res = http_json(
        f"{QDRANT}/collections/{COLLECTION}/points/search",
        {"vector": vec, "limit": limit, "with_payload": True},
    )
    return [{**p["payload"], "sim": round(p["score"], 3)} for p in res["result"]]


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
        if u.path == "/api/search":
            q = urllib.parse.parse_qs(u.query).get("q", [""])[0].strip()
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
    print(f"court-search on :{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

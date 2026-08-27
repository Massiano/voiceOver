"""
voiceRelay -- live mic voice-over demo.

Pattern: a broadcaster's browser records short, INDEPENDENTLY-DECODABLE webm/opus
chunks (each chunk is its own start/stop MediaRecorder session, so it carries its
own header -- a continuous timesliced recorder would emit headerless continuation
fragments that decodeAudioData() can't play standalone). Chunks are POSTed to the
server, which stamps each one with a server-clock timestamp and a fixed-delay
`play_at` target. Listeners poll for new chunk metadata, fetch the bytes, decode
them with the Web Audio API, and schedule playback at `play_at` translated into
their own local clock (via a smoothed server/local clock-offset estimate). Every
listener scheduling off the same play_at target is what keeps them in sync --
the fixed delay just has to be bigger than the typical broadcast->listener lag.

In-memory only, per room. No auth: this is a demo, not a product.
"""
import os
import re
import threading
import time

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,30}$")

BUFFER_SECONDS = float(os.environ.get("VR_BUFFER_SECONDS", "2.5"))
MAX_CHUNKS_PER_ROOM = int(os.environ.get("VR_MAX_CHUNKS_PER_ROOM", "40"))
ROOM_STALE_SECS = int(os.environ.get("VR_ROOM_STALE_SECS", "20"))
ROOM_REAP_SECS = int(os.environ.get("VR_ROOM_REAP_SECS", "600"))   # drop the room entirely after this long silent
MAX_CHUNK_BYTES = int(os.environ.get("VR_MAX_CHUNK_BYTES", str(512 * 1024)))

rooms = {}              # room -> {chunks: {seq: {seq, ts, play_at, size, data}}, seq_counter, last_seen, total_chunks}
rooms_lock = threading.Lock()


def _room_id(raw):
    rid = (raw or "").strip().lstrip("@").lower()
    return rid if ROOM_ID_RE.match(rid) else None


def _get_room(rid, create=False):
    """Caller holds rooms_lock."""
    r = rooms.get(rid)
    if r is None and create:
        r = {"chunks": {}, "seq_counter": 0, "last_seen": time.time(), "total_chunks": 0}
        rooms[rid] = r
    return r


def _reap_stale_rooms(now):
    """Caller holds rooms_lock. Drops rooms nobody has broadcast to in a long while."""
    dead = [rid for rid, r in rooms.items() if now - r["last_seen"] > ROOM_REAP_SECS]
    for rid in dead:
        del rooms[rid]


@app.route("/api/voice/broadcast/<room>", methods=["POST"])
def broadcast_chunk(room):
    rid = _room_id(room)
    if not rid:
        return jsonify({"status": "error", "message": "Room: 1-30 letters, digits, - or _."}), 400

    f = request.files.get("chunk")
    if f is None:
        return jsonify({"status": "error", "message": "Missing 'chunk' file field."}), 400
    data = f.read()
    if not data:
        return jsonify({"status": "error", "message": "Empty chunk."}), 400
    if len(data) > MAX_CHUNK_BYTES:
        return jsonify({"status": "error", "message": "Chunk too large."}), 413

    now = time.time()
    with rooms_lock:
        _reap_stale_rooms(now)
        r = _get_room(rid, create=True)
        seq = r["seq_counter"]
        r["seq_counter"] += 1
        r["total_chunks"] += 1
        r["last_seen"] = now
        play_at = now + BUFFER_SECONDS
        r["chunks"][seq] = {"seq": seq, "ts": now, "play_at": play_at, "size": len(data), "data": data}
        while len(r["chunks"]) > MAX_CHUNKS_PER_ROOM:
            oldest = next(iter(r["chunks"]))
            del r["chunks"][oldest]

    return jsonify({"status": "success", "seq": seq, "server_time_now": now,
                    "play_at": play_at, "buffer_seconds": BUFFER_SECONDS})


@app.route("/api/voice/stream/<room>", methods=["GET"])
def stream_room(room):
    """Chunk metadata newer than ?after=<seq> (bytes fetched separately, see /chunk).
    ?bootstrap=1 returns no chunks, just latest_seq -- lets a joining listener skip
    the backlog and start live instead of rapid-firing everything already buffered."""
    rid = _room_id(room)
    if not rid:
        return jsonify({"status": "error", "message": "Bad room id."}), 400

    now = time.time()
    try:
        after = int(request.args.get("after", -1))
    except (TypeError, ValueError):
        after = -1
    bootstrap = request.args.get("bootstrap") == "1"

    with rooms_lock:
        r = rooms.get(rid)
        if r is None or now - r["last_seen"] > ROOM_STALE_SECS:
            return jsonify({"status": "error", "message": "Room is offline (no live broadcaster)."}), 404
        latest_seq = max(r["chunks"]) if r["chunks"] else after
        chunks = [] if bootstrap else [
            {"seq": c["seq"], "ts": c["ts"], "play_at": c["play_at"], "size": c["size"]}
            for c in r["chunks"].values() if c["seq"] > after
        ]

    return jsonify({"status": "success", "room": rid, "server_time_now": now,
                    "buffer_seconds": BUFFER_SECONDS, "latest_seq": latest_seq, "chunks": chunks})


@app.route("/api/voice/chunk/<room>/<int:seq>", methods=["GET"])
def get_chunk(room, seq):
    rid = _room_id(room)
    if not rid:
        return jsonify({"status": "error", "message": "Bad room id."}), 400
    with rooms_lock:
        r = rooms.get(rid)
        c = r["chunks"].get(seq) if r else None
        data = c["data"] if c else None
    if data is None:
        return jsonify({"status": "error", "message": "Chunk not found (too old or wrong room)."}), 404
    resp = Response(data, mimetype="audio/webm")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/voice/rooms", methods=["GET"])
def list_rooms():
    now = time.time()
    with rooms_lock:
        _reap_stale_rooms(now)
        out = [{"room": rid, "live": now - r["last_seen"] <= ROOM_STALE_SECS,
                "age_seconds": round(now - r["last_seen"], 1), "total_chunks": r["total_chunks"]}
               for rid, r in rooms.items()]
    out.sort(key=lambda x: (-x["live"], x["room"]))
    return jsonify({"status": "success", "rooms": out, "buffer_seconds": BUFFER_SECONDS})


@app.route("/api/diag", methods=["GET"])
def diag():
    now = time.time()
    with rooms_lock:
        room_count = len(rooms)
        live_count = sum(1 for r in rooms.values() if now - r["last_seen"] <= ROOM_STALE_SECS)
    return jsonify({"service": "voiceRelay", "server_time": now, "rooms": room_count,
                    "live_rooms": live_count, "buffer_seconds": BUFFER_SECONDS,
                    "max_chunks_per_room": MAX_CHUNKS_PER_ROOM})


@app.route("/", methods=["GET"])
def serve_root():
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static_wildcard(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

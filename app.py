"""
voiceRelay -- live mic + music voice-over demo.

Mixing happens client-side on the broadcaster: mic and an optional local music
track are combined into one MediaStream via the Web Audio API before encoding,
so voice and music are in sync by construction (one shared audio clock, no
separate sync problem). The broadcaster also has a local zero-latency monitor
tap of that same pre-encode mix -- it never listens to the delayed network path.

Each chunk is its own start/stop MediaRecorder session (self-contained webm
header, independently decodable -- a continuous timesliced recorder emits
headerless continuation fragments that decodeAudioData() can't play standalone).

Sync across listeners uses deadline scheduling, the same idea multiplayer-game
netcode and watch-party tools use for heterogeneous clients: the server just
timestamps each chunk on receipt (`ts`); each listener independently measures
its own RTT/jitter and snaps to the smallest of a small shared set of buffer
tiers (BUFFER_TIERS_MS) it can reliably hit, then schedules playback at
`ts + tier`. Two listeners on the same tier compute the identical absolute
playback time and are therefore sample-accurate in sync with each other, even
though they never talk to each other. A listener whose connection can't hit
even the largest tier is flagged degraded client-side rather than forcing
everyone else's delay up to match it.

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

# Shared buffer tiers, in milliseconds. Listeners snap to the smallest tier
# their measured RTT + jitter fits inside of. Keep this list small: everyone
# on the same tier hears the same content at the same instant.
BUFFER_TIERS_MS = [int(x) for x in os.environ.get("VR_BUFFER_TIERS_MS", "800,1500,3000").split(",")]

MAX_CHUNKS_PER_ROOM = int(os.environ.get("VR_MAX_CHUNKS_PER_ROOM", "60"))
ROOM_STALE_SECS = int(os.environ.get("VR_ROOM_STALE_SECS", "20"))
ROOM_REAP_SECS = int(os.environ.get("VR_ROOM_REAP_SECS", "600"))   # drop the room entirely after this long silent
MAX_CHUNK_BYTES = int(os.environ.get("VR_MAX_CHUNK_BYTES", str(512 * 1024)))

rooms = {}              # room -> {chunks: {seq: {seq, ts, size, data}}, seq_counter, last_seen, total_chunks}
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
        r["chunks"][seq] = {"seq": seq, "ts": now, "size": len(data), "data": data}
        while len(r["chunks"]) > MAX_CHUNKS_PER_ROOM:
            oldest = next(iter(r["chunks"]))
            del r["chunks"][oldest]

    return jsonify({"status": "success", "seq": seq, "server_time_now": now})


@app.route("/api/voice/stream/<room>", methods=["GET"])
def stream_room(room):
    """Chunk metadata newer than ?after=<seq> (bytes fetched separately, see /chunk).
    ?bootstrap=1 returns no chunks, just latest_seq -- lets a joining listener skip
    the backlog and start live instead of rapid-firing everything already buffered.
    Also doubles as the RTT-measurement request: the caller times this round trip
    and reads server_time_now to estimate clock offset and jitter."""
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
            {"seq": c["seq"], "ts": c["ts"], "size": c["size"]}
            for c in r["chunks"].values() if c["seq"] > after
        ]

    return jsonify({"status": "success", "room": rid, "server_time_now": now,
                    "tiers_ms": BUFFER_TIERS_MS, "latest_seq": latest_seq, "chunks": chunks})


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
    return jsonify({"status": "success", "rooms": out, "tiers_ms": BUFFER_TIERS_MS})


@app.route("/api/diag", methods=["GET"])
def diag():
    now = time.time()
    with rooms_lock:
        room_count = len(rooms)
        live_count = sum(1 for r in rooms.values() if now - r["last_seen"] <= ROOM_STALE_SECS)
    return jsonify({"service": "voiceRelay", "server_time": now, "rooms": room_count,
                    "live_rooms": live_count, "tiers_ms": BUFFER_TIERS_MS,
                    "max_chunks_per_room": MAX_CHUNKS_PER_ROOM})


@app.route("/", methods=["GET"])
def serve_root():
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static_wildcard(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

#!/usr/bin/env python3
"""
Pose Detection & Frame Stability Web Server

Subscribes to /stable_gest/landmarks and /stable_gest/image_annotated published
by publisher.py.  Runs a TRACKING / NOT_TRACKING state machine (5 s each), buffers
validated landmark frames, builds an averaged skeleton, and serves a live web UI.
"""
import json
import math
import time
import threading

import cv2
import rospy
import numpy as np
from flask import Flask, Response, render_template_string
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ── Tunable parameters ────────────────────────────────────────────────────────
TRACKING_DURATION     = 5.0   # seconds per tracking wave
NOT_TRACKING_DURATION = 5.0   # seconds between waves
PRESENCE_THRESHOLD    = 0.5   # per-landmark minimum presence score
VISIBILITY_THRESHOLD  = 0.5   # per-landmark minimum visibility score
STABILITY_N           = 3     # neighbor half-window for stability score (future use)
STABILITY_SIGMA       = 1.0   # Gaussian σ for stability score weights (future use)
WEB_PORT              = 8080
# ─────────────────────────────────────────────────────────────────────────────

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


# ── Stability score (Gaussian-weighted neighbor distances) ────────────────────
# Uses world_landmarks for 3D distance.  Currently not called — naive average
# is used instead.  Here for when we graduate from the naive fallback.

def _frame_distance(a, b):
    """Sum of Euclidean distances between corresponding 3-D landmark vectors."""
    total = 0.0
    for la, lb in zip(a, b):
        dx, dy, dz = la['x'] - lb['x'], la['y'] - lb['y'], la['z'] - lb['z']
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _stability_score(frames, idx, N, sigma):
    """Gaussian-weighted sum of distances to the N nearest neighbors on each side."""
    score = 0.0
    for k in range(-N, N + 1):
        if k == 0:
            continue
        j = idx + k
        if 0 <= j < len(frames):
            w = math.exp(-k * k / (2.0 * sigma * sigma))
            score += _frame_distance(frames[idx], frames[j]) * w
    return score


def most_stable_frame(frames, N=STABILITY_N, sigma=STABILITY_SIGMA):
    """Return the index of the most stable frame, excluding the first/last N."""
    if len(frames) < 2 * N + 1:
        return len(frames) // 2
    scores = [_stability_score(frames, i, N, sigma) for i in range(N, len(frames) - N)]
    return N + scores.index(min(scores))


# ── Naive skeleton average ────────────────────────────────────────────────────

def naive_average_skeleton(frames):
    """Average landmark (x, y, z) across all buffered frames."""
    if not frames:
        return None
    n = len(frames[0])
    return [
        {
            'x': sum(f[i]['x'] for f in frames) / len(frames),
            'y': sum(f[i]['y'] for f in frames) / len(frames),
            'z': sum(f[i]['z'] for f in frames) / len(frames),
        }
        for i in range(n)
    ]


# ── Frame validation ──────────────────────────────────────────────────────────

REQUIRED_LANDMARKS = {11, 12, 13, 14, 15, 16}  # shoulders, elbows, wrists

def is_valid_frame(landmarks):
    """Reject frame if any core landmark is below either confidence threshold."""
    if not landmarks:
        return False
    return all(
        landmarks[i]['presence'] >= PRESENCE_THRESHOLD and
        landmarks[i]['visibility'] >= VISIBILITY_THRESHOLD
        for i in REQUIRED_LANDMARKS
        if i < len(landmarks)
    )


# ── Shared state ──────────────────────────────────────────────────────────────

class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        # State machine
        self.mode        = 'NOT_TRACKING'
        self.phase_start = time.time()
        self.cycle       = 0
        # Frame buffer (norm_landmarks lists for each accepted frame)
        self.buffer      = []
        # Display state
        self.status      = 'loading'   # 'loading' | 'no_human' | 'detected'
        self.skeleton    = None        # list of {x,y,z} or None
        # Latest annotated frame for MJPEG feed
        self.latest_bgr  = None
        # Logging counters
        self._lm_count   = 0
        self._img_count  = 0
        self._last_log   = time.time()

    # ── Called from ROS subscriber thread ────────────────────────────────────

    def on_landmarks(self, payload):
        norm_lm = payload.get('norm_landmarks', [])

        with self.lock:
            now     = time.time()
            elapsed = now - self.phase_start

            # ── State transitions ────────────────────────────────────────────
            if self.mode == 'TRACKING' and elapsed >= TRACKING_DURATION:
                if not self.buffer:
                    self.status = 'no_human'
                self.mode        = 'NOT_TRACKING'
                self.phase_start = now
                self.cycle      += 1

            elif self.mode == 'NOT_TRACKING' and elapsed >= NOT_TRACKING_DURATION:
                self.buffer      = []
                self.status      = 'loading'
                self.skeleton    = None
                self.mode        = 'TRACKING'
                self.phase_start = now

            # ── Frame ingestion (only during TRACKING) ───────────────────────
            if self.mode == 'TRACKING' and is_valid_frame(norm_lm):
                self.buffer.append(norm_lm)
                self.skeleton = naive_average_skeleton(self.buffer)
                self.status   = 'detected'

            self._lm_count += 1
            now = time.time()
            if now - self._last_log >= 1.0:
                print('[WebServer] landmarks %d/s  images %d/s  mode=%s  buffer=%d  status=%s' % (
                    self._lm_count, self._img_count,
                    self.mode, len(self.buffer), self.status,
                ), flush=True)
                self._lm_count  = 0
                self._img_count = 0
                self._last_log  = now

    def on_image(self, ros_image):
        frame = np.ndarray(
            (ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        ).copy()
        # cv2_image2ros publishes BGR packed as-is; convert only if the message
        # encoding says rgb8 (typical for ROS camera drivers).
        if getattr(ros_image, 'encoding', '') == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self.lock:
            self.latest_bgr = frame
            self._img_count += 1

    # ── Called from Flask threads ─────────────────────────────────────────────

    def snapshot(self):
        with self.lock:
            elapsed = time.time() - self.phase_start
            dur     = TRACKING_DURATION if self.mode == 'TRACKING' else NOT_TRACKING_DURATION
            return {
                'mode':            self.mode,
                'phase_remaining': round(max(0.0, dur - elapsed), 1),
                'phase_duration':  dur,
                'status':          self.status,
                'skeleton':        self.skeleton,
                'buffer_size':     len(self.buffer),
                'cycle':           self.cycle,
            }

    def get_latest_bgr(self):
        with self.lock:
            return self.latest_bgr


state = StateManager()

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pose Stability</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#0e0e0e;color:#ddd;min-height:100vh}
header{display:flex;align-items:center;gap:14px;padding:10px 18px;
       border-bottom:1px solid #222;background:#141414}
header h1{font-size:16px;color:#aaa;letter-spacing:.05em}
.badge{padding:3px 11px;border-radius:3px;font-size:12px;font-weight:bold;letter-spacing:.08em}
.badge.tracking{background:#1b5e20;color:#a5d6a7}
.badge.waiting {background:#2a2a2a;color:#666}
.meta{font-size:12px;color:#555;margin-left:auto}
.progress{height:3px;background:#1a1a1a;overflow:hidden}
.bar{height:100%;width:0%;transition:width .5s linear}
.bar.tracking{background:#2e7d32}
.bar.waiting {background:#333}
.layout{display:flex;gap:10px;padding:10px}
.pane{background:#141414;border:1px solid #222;border-radius:5px;padding:10px}
.feed-pane{flex:1;min-width:0}
.result-pane{width:340px;flex-shrink:0}
.pane-title{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
img#feed{width:100%;border-radius:3px;background:#000;display:block}
canvas#skel{width:100%;background:#0a0a0a;border-radius:3px;display:block}
.spinner{display:flex;align-items:center;justify-content:center;height:150px;
         gap:6px;color:#555;font-size:13px}
.dot{width:7px;height:7px;border-radius:50%;background:#333;
     animation:pulse 1.1s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.15s}.dot:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
.no-human{display:flex;align-items:center;justify-content:center;height:150px;
          font-size:13px;color:#b71c1c}
.hint{font-size:11px;color:#444;margin-top:6px}
</style>
</head>
<body>
<header>
  <span id="badge" class="badge waiting">NOT TRACKING</span>
  <span id="timer" style="font-size:13px;color:#666">—</span>
  <span class="meta" id="meta">cycle 0 &nbsp;·&nbsp; 0 frames</span>
  <h1 style="margin-left:auto">Pose Stability</h1>
</header>
<div class="progress"><div id="bar" class="bar waiting"></div></div>

<div class="layout">
  <div class="pane feed-pane">
    <div class="pane-title">Live Camera</div>
    <img id="feed" src="/feed" alt="camera feed">
  </div>
  <div class="pane result-pane">
    <div class="pane-title">Stable Skeleton</div>
    <div id="result">
      <div class="spinner">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        <span>waiting for tracking…</span>
      </div>
    </div>
    <div class="hint" id="hint"></div>
  </div>
</div>

<script>
const CONN = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
  [9,10],[11,12],
  [11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[27,31],[29,31],
  [24,26],[26,28],[28,30],[28,32],[30,32],
];

let canvas = null, ctx = null, lastStatus = null;

function ensureCanvas() {
  const box = document.getElementById('result');
  if (!canvas) {
    box.innerHTML = '<canvas id="skel"></canvas>';
    canvas = document.getElementById('skel');
    ctx = canvas.getContext('2d');
  }
}

function drawSkeleton(skel) {
  ensureCanvas();
  const W = canvas.offsetWidth || 320, H = Math.round(W * 1.5);
  canvas.width = W; canvas.height = H;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  const pad = 24;
  function pt(lm) {
    return [lm.x * (W - 2*pad) + pad, lm.y * (H - 2*pad) + pad];
  }

  // left-side connections in teal, right in blue, centre in grey
  function connColor(a, b) {
    const leftIds  = new Set([1,2,3,7,13,15,17,19,21,23,25,27,29,31]);
    const rightIds = new Set([4,5,6,8,14,16,18,20,22,24,26,28,30,32]);
    if (leftIds.has(a) || leftIds.has(b))  return '#26a69a';
    if (rightIds.has(a) || rightIds.has(b)) return '#1565c0';
    return '#555';
  }

  ctx.lineWidth = 1.5;
  for (const [a, b] of CONN) {
    if (a >= skel.length || b >= skel.length) continue;
    ctx.strokeStyle = connColor(a, b);
    ctx.beginPath();
    const [ax, ay] = pt(skel[a]);
    const [bx, by] = pt(skel[b]);
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.stroke();
  }

  ctx.fillStyle = '#80cbc4';
  for (const lm of skel) {
    const [x, y] = pt(lm);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

const es = new EventSource('/events');
es.onmessage = function(ev) {
  const d = JSON.parse(ev.data);

  // header
  const badge = document.getElementById('badge');
  const tracking = d.mode === 'TRACKING';
  badge.textContent = tracking ? 'TRACKING' : 'NOT TRACKING';
  badge.className = 'badge ' + (tracking ? 'tracking' : 'waiting');

  document.getElementById('timer').textContent =
    d.phase_remaining.toFixed(1) + 's';
  document.getElementById('meta').textContent =
    'cycle ' + d.cycle + '  ·  ' + d.buffer_size + ' frames';

  // progress bar
  const bar = document.getElementById('bar');
  const pct = ((d.phase_duration - d.phase_remaining) / d.phase_duration * 100).toFixed(1);
  bar.style.width = pct + '%';
  bar.className = 'bar ' + (tracking ? 'tracking' : 'waiting');

  // result panel — only rebuild DOM when status changes
  if (d.status !== lastStatus) {
    lastStatus = d.status;
    canvas = null; ctx = null;
    const box = document.getElementById('result');
    if (d.status === 'loading') {
      box.innerHTML = '<div class="spinner">' +
        '<div class="dot"></div><div class="dot"></div><div class="dot"></div>' +
        '<span>scanning…</span></div>';
    } else if (d.status === 'no_human') {
      box.innerHTML = '<div class="no-human">no human detected, trying again…</div>';
    }
    // 'detected' case: drawSkeleton handles the DOM
  }

  if (d.status === 'detected' && d.skeleton) {
    drawSkeleton(d.skeleton);
    document.getElementById('hint').textContent =
      'avg of ' + d.buffer_size + ' frames  (cycle ' + d.cycle + ')';
  } else {
    document.getElementById('hint').textContent = '';
  }
};
es.onerror = function() {
  document.getElementById('badge').textContent = 'DISCONNECTED';
  document.getElementById('badge').className = 'badge waiting';
};
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/feed')
def video_feed():
    def generate():
        while True:
            frame = state.get_latest_bgr()
            if frame is not None:
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        buf.tobytes() +
                        b'\r\n'
                    )
            time.sleep(0.04)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/events')
def sse():
    def generate():
        while True:
            data = state.snapshot()
            yield 'data: ' + json.dumps(data) + '\n\n'
            time.sleep(0.5)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# ── ROS thread ────────────────────────────────────────────────────────────────

def ros_main():
    rospy.init_node('pose_web_server', anonymous=False)

    def landmarks_cb(msg):
        try:
            state.on_landmarks(json.loads(msg.data))
        except Exception as e:
            rospy.logwarn('[WebServer] bad landmark msg: %s' % e)

    def image_cb(ros_image):
        state.on_image(ros_image)

    rospy.Subscriber('/stable_gest/landmarks', String, landmarks_cb)
    rospy.Subscriber('/stable_gest/image_annotated', Image, image_cb)
    rospy.loginfo('[WebServer] subscribed to /stable_gest/ topics')
    rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import argparse
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=WEB_PORT)
    args = parser.parse_args()

    def run_flask():
        try:
            app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)
        except OSError as e:
            print('[WebServer] ERROR: %s' % e, flush=True)
            print('[WebServer] usage: python3 webserver.py [--port PORT]', flush=True)
            import os; os._exit(1)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print('[WebServer] http://0.0.0.0:%d' % args.port, flush=True)

    ros_main()  # owns the main thread and its signal handlers


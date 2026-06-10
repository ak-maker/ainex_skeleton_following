#!/usr/bin/env python3
"""
Pose Detection Web Server

Subscribes to:
  /head_tracker/skeleton   — live most-stable skeleton (updated each valid frame by control.py)
  /head_tracker/cycle_end  — end-of-cycle snapshot     (published by control.py each cycle)
  /stable_gest/image_annotated — MJPEG source

Serves a live web UI showing the camera feed, real-time skeleton, and the previous cycle's skeleton.
"""
import json
import time
import threading

import cv2
import rospy
import numpy as np
from flask import Flask, Response, render_template_string
from sensor_msgs.msg import Image
from std_msgs.msg import String

WEB_PORT = 8080


class StateManager:
    def __init__(self):
        self.lock           = threading.Lock()
        self.current_skel   = None   # latest from /head_tracker/skeleton
        self.prev_skel      = None   # snapshot at last cycle end
        self.prev_had_human = None   # True/False/None (None = no cycle completed yet)
        self.cycle          = 0      # number of completed cycles
        self.latest_bgr     = None

    def on_skeleton(self, skel):
        with self.lock:
            self.current_skel = skel

    def on_cycle_end(self, payload):
        with self.lock:
            self.prev_skel      = payload.get('skeleton')
            self.prev_had_human = payload.get('had_human', False)
            self.cycle          = payload.get('cycle', 0) + 1
            self.current_skel   = None  # reset live view for the new cycle

    def on_image(self, ros_image):
        frame = np.ndarray(
            (ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        ).copy()
        if getattr(ros_image, 'encoding', '') == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self.lock:
            self.latest_bgr = frame

    def snapshot(self):
        with self.lock:
            return {
                'current_skeleton': self.current_skel,
                'prev_skeleton':    self.prev_skel,
                'prev_had_human':   self.prev_had_human,
                'cycle':            self.cycle,
            }

    def get_latest_bgr(self):
        with self.lock:
            return self.latest_bgr


state = StateManager()
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
header h1{font-size:16px;color:#aaa;letter-spacing:.05em;margin-right:auto}
.cycle-badge{padding:3px 10px;border-radius:3px;font-size:12px;font-weight:bold;
             letter-spacing:.08em;background:#1a237e;color:#9fa8da}
.layout{display:flex;gap:10px;padding:10px}
.pane{background:#141414;border:1px solid #222;border-radius:5px;padding:10px}
.feed-pane{flex:1;min-width:0}
.skel-pane{width:265px;flex-shrink:0}
.pane-title{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
img#feed{width:100%;border-radius:3px;background:#000;display:block}
canvas.skel{width:100%;background:#0a0a0a;border-radius:3px;display:block}
.msg{display:flex;align-items:center;justify-content:center;height:130px;
     font-size:12px;color:#555;gap:5px;flex-wrap:wrap;justify-content:center}
.msg.no-human{color:#b71c1c}
.dot{width:6px;height:6px;border-radius:50%;background:#333;flex-shrink:0;
     animation:pulse 1.1s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.15s}.dot:nth-child(3){animation-delay:.3s}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}
@keyframes cycle-flash{0%{border-color:#388e3c;box-shadow:0 0 10px #388e3c}100%{border-color:#222;box-shadow:none}}
.cycle-flash{animation:cycle-flash 0.8s ease-out forwards}
</style>
</head>
<body>
<header>
  <h1>Pose Stability</h1>
  <span id="cycle-badge" class="cycle-badge">cycle 0</span>
</header>

<div class="layout">
  <div class="pane feed-pane">
    <div class="pane-title">Camera</div>
    <img id="feed" src="/feed" alt="camera feed">
  </div>
  <div class="pane skel-pane" id="live-pane">
    <div class="pane-title">Live Skeleton</div>
    <div id="live-box">
      <div class="msg"><span class="dot"></span><span class="dot"></span><span class="dot"></span><span>scanning…</span></div>
    </div>
  </div>
  <div class="pane skel-pane">
    <div class="pane-title">Previous Cycle</div>
    <div id="prev-box">
      <div class="msg"><span>waiting for first cycle…</span></div>
    </div>
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

const LEFT_IDS  = new Set([1,2,3,7,13,15,17,19,21,23,25,27,29,31]);
const RIGHT_IDS = new Set([4,5,6,8,14,16,18,20,22,24,26,28,30,32]);
function connColor(a, b) {
  if (LEFT_IDS.has(a)  || LEFT_IDS.has(b))  return '#26a69a';
  if (RIGHT_IDS.has(a) || RIGHT_IDS.has(b)) return '#1565c0';
  return '#555';
}

function drawSkel(boxId, skel) {
  const box = document.getElementById(boxId);
  let cv = box.querySelector('canvas');
  if (!cv) {
    box.innerHTML = '<canvas class="skel"></canvas>';
    cv = box.querySelector('canvas');
  }
  const W = cv.offsetWidth || 245, H = Math.round(W * 1.4);
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);
  const pad = 20;
  const pt = lm => [lm.x * (W - 2*pad) + pad, lm.y * (H - 2*pad) + pad];
  ctx.lineWidth = 1.5;
  for (const [a, b] of CONN) {
    if (a >= skel.length || b >= skel.length) continue;
    ctx.strokeStyle = connColor(a, b);
    ctx.beginPath();
    const [ax, ay] = pt(skel[a]), [bx, by] = pt(skel[b]);
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.fillStyle = '#80cbc4';
  for (const lm of skel) {
    const [x, y] = pt(lm);
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2*Math.PI); ctx.fill();
  }
}

function showMsg(boxId, text, cls='') {
  document.getElementById(boxId).innerHTML =
    '<div class="msg ' + cls + '">' + text + '</div>';
}

const SPINNER = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

let prevCycle = -1;

const es = new EventSource('/events');
es.onmessage = function(ev) {
  const d = JSON.parse(ev.data);

  document.getElementById('cycle-badge').textContent = 'cycle ' + d.cycle;

  // live skeleton — updates every SSE tick
  if (d.current_skeleton) {
    drawSkel('live-box', d.current_skeleton);
  } else {
    showMsg('live-box', SPINNER + '<span>scanning…</span>');
  }

  // previous cycle — only rebuild DOM when cycle number advances
  if (d.cycle !== prevCycle) {
    prevCycle = d.cycle;
    const pane = document.getElementById('live-pane');
    pane.classList.remove('cycle-flash');
    void pane.offsetWidth;  // force reflow so re-adding the class restarts the animation
    pane.classList.add('cycle-flash');
    pane.addEventListener('animationend', () => pane.classList.remove('cycle-flash'), {once: true});
    if (d.prev_skeleton) {
      drawSkel('prev-box', d.prev_skeleton);
    } else if (d.prev_had_human === false) {
      showMsg('prev-box', 'no human detected', 'no-human');
    } else {
      showMsg('prev-box', 'waiting for first cycle…');
    }
  }
};
es.onerror = function() {
  showMsg('live-box', 'DISCONNECTED');
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


def ros_main():
    rospy.init_node('pose_web_server', anonymous=False)

    def skeleton_cb(msg):
        try:
            state.on_skeleton(json.loads(msg.data))
        except Exception as e:
            rospy.logwarn('[WebServer] bad skeleton msg: %s' % e)

    def cycle_end_cb(msg):
        try:
            state.on_cycle_end(json.loads(msg.data))
        except Exception as e:
            rospy.logwarn('[WebServer] bad cycle_end msg: %s' % e)

    def image_cb(ros_image):
        state.on_image(ros_image)

    rospy.Subscriber('/head_tracker/skeleton',       String, skeleton_cb)
    rospy.Subscriber('/head_tracker/cycle_end',      String, cycle_end_cb)
    rospy.Subscriber('/stable_gest/image_annotated', Image,  image_cb)
    rospy.loginfo('[WebServer] subscribed to /head_tracker/ and /stable_gest/ topics')
    rospy.spin()


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

    ros_main()

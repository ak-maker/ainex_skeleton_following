#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo Control GUI v2 — with action sequences and robot visualization.
Run:  python3 servo_gui.py
Open: http://<robot_ip>:9090
"""

import json
import time
import sqlite3
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import rospy
from ainex_kinematics.motion_manager import MotionManager

# ============================================================
# Servo mapping
# ============================================================
SERVO_NAMES = {
    1: 'l_ank_roll',   2: 'r_ank_roll',
    3: 'l_ank_pitch',  4: 'r_ank_pitch',
    5: 'l_knee',       6: 'r_knee',
    7: 'l_hip_pitch',  8: 'r_hip_pitch',
    9: 'l_hip_roll',  10: 'r_hip_roll',
    11: 'l_hip_yaw',  12: 'r_hip_yaw',
    13: 'l_sho_pitch', 14: 'r_sho_pitch',
    15: 'l_sho_roll',  16: 'r_sho_roll',
    17: 'l_el_pitch',  18: 'r_el_pitch',
    19: 'l_el_yaw',    20: 'r_el_yaw',
    21: 'l_gripper',   22: 'r_gripper',
    23: 'head_pan',    24: 'head_tilt',
}

GROUPS = [
    ('Head', [23, 24]),
    ('Left Arm', [13, 15, 17, 19, 21]),
    ('Right Arm', [14, 16, 18, 20, 22]),
    ('Left Leg', [11, 9, 7, 5, 3, 1]),
    ('Right Leg', [12, 10, 8, 6, 4, 2]),
]

# ============================================================
# Load poses from .d6a files
# ============================================================
D6A_DIR = '/home/ubuntu/software/ainex_controller/ActionGroups'

def load_d6a_first(filename):
    """Load first frame from d6a. Returns {servo_id: pulse}."""
    path = '%s/%s.d6a' % (D6A_DIR, filename)
    conn = sqlite3.connect(path)
    row = conn.execute('SELECT * FROM ActionGroup ORDER BY [Index] LIMIT 1').fetchone()
    conn.close()
    pulses = {i: row[i + 1] for i in range(1, 23)}
    pulses[23] = 500
    pulses[24] = 500
    return pulses

def load_d6a_all(filename):
    """Load all frames from d6a. Returns list of {time, servos}."""
    path = '%s/%s.d6a' % (D6A_DIR, filename)
    conn = sqlite3.connect(path)
    rows = conn.execute('SELECT * FROM ActionGroup ORDER BY [Index]').fetchall()
    conn.close()
    frames = []
    for row in rows:
        servos = {i: row[i + 1] for i in range(1, 23)}
        servos[23] = 500
        servos[24] = 500
        frames.append({'time': row[1], 'servos': servos})
    return frames

# Single-frame presets
PRESETS = {}
for key, label, fname in [('stand', 'Stand', 'stand'),
                            ('t_pose', 'T-Pose', '0'),
                            ('walk_ready', 'Walk Ready', 'walk_ready')]:
    try:
        PRESETS[key] = {'label': label, 'pulses': load_d6a_first(fname)}
    except Exception as e:
        print('[WARN] Preset %s: %s' % (fname, e))

# Multi-frame actions
ACTIONS = {}
for key, label, fname in [('wave', 'Wave', 'wave'),
                            ('greet', 'Greet', 'greet'),
                            ('hurdles', 'Hurdles', 'hurdles')]:
    try:
        frames = load_d6a_all(fname)
        ACTIONS[key] = {'label': label, 'frames': frames, 'count': len(frames)}
        print('[LOAD] %s: %d frames' % (fname, len(frames)))
    except Exception as e:
        print('[WARN] Action %s: %s' % (fname, e))

# ============================================================
# State
# ============================================================
current_pulses = dict(PRESETS.get('walk_ready', PRESETS['stand'])['pulses'])
action_playing = False
action_stop = False

# ============================================================
# ROS
# ============================================================
rospy.init_node('servo_gui', anonymous=True)
motion_mgr = MotionManager()

def send_servo(servo_id, pulse, duration=300):
    motion_mgr.set_servos_position(duration, [[servo_id, pulse]])

def send_all(pulses, duration=1000):
    cmds = [[sid, p] for sid, p in sorted(pulses.items()) if sid <= 24]
    motion_mgr.set_servos_position(duration, cmds)

def play_action(name):
    """Play a multi-frame action sequence in background thread."""
    global action_playing, action_stop
    if name not in ACTIONS:
        return
    action_playing = True
    action_stop = False
    frames = ACTIONS[name]['frames']
    print('[ACTION] Playing %s (%d frames)' % (name, len(frames)), flush=True)
    for i, frame in enumerate(frames):
        if action_stop:
            print('[ACTION] Stopped', flush=True)
            break
        current_pulses.update(frame['servos'])
        send_all(current_pulses, frame['time'])
        time.sleep(frame['time'] / 1000.0)
    action_playing = False
    print('[ACTION] Done', flush=True)

# ============================================================
# HTML with visualization
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AiNex Servo Control</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 10px; }
h1 { text-align: center; color: #00d4ff; margin-bottom: 10px; font-size: 1.3em; }

.buttons { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-bottom: 12px; }
.buttons button {
    padding: 8px 14px; border: none; border-radius: 6px; cursor: pointer;
    font-size: 0.85em; font-weight: 600; color: #fff; transition: transform 0.1s;
}
.buttons button:active { transform: scale(0.95); }
.buttons button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-preset { background: #2ecc71; }
.btn-action { background: #9b59b6; }
.btn-stop { background: #e74c3c; }

.main { display: grid; grid-template-columns: 320px 1fr; gap: 12px; }
@media (max-width: 750px) { .main { grid-template-columns: 1fr; } }

.viz-panel { background: #16213e; border-radius: 10px; padding: 8px; }
.viz-panel canvas { width: 100%; border-radius: 6px; }
.viz-label { text-align: center; color: #666; font-size: 0.75em; margin-top: 4px; }

.controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 8px; }
.group { background: #16213e; border-radius: 8px; padding: 10px; }
.group h2 { font-size: 0.9em; color: #00d4ff; margin-bottom: 6px; border-bottom: 1px solid #333; padding-bottom: 3px; }
.servo-row {
    display: grid; grid-template-columns: 85px 1fr 58px; align-items: center;
    gap: 4px; margin-bottom: 4px;
}
.servo-label { font-size: 0.75em; color: #aaa; }
.servo-label span { color: #555; }
input[type=range] { width: 100%; accent-color: #00d4ff; cursor: pointer; height: 18px; }
input[type=number] {
    width: 54px; background: #0f3460; color: #fff; border: 1px solid #444;
    border-radius: 4px; padding: 3px; text-align: center; font-size: 0.8em;
}
input[type=number]:focus { border-color: #00d4ff; outline: none; }
#toast {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: #2ecc71; color: #fff; padding: 8px 20px; border-radius: 20px;
    font-size: 0.85em; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 99;
}
#toast.show { opacity: 1; }
#status { text-align: center; color: #e67e22; font-size: 0.85em; margin-bottom: 6px; min-height: 1.2em; }
</style>
</head>
<body>
<h1>AiNex Servo Control</h1>
<div id="status"></div>
<div class="buttons" id="buttons"></div>
<div class="main">
  <div class="viz-panel">
    <canvas id="cvFront" width="300" height="420"></canvas>
    <div class="viz-label">Front View</div>
    <canvas id="cvSide" width="300" height="420" style="margin-top:8px;"></canvas>
    <div class="viz-label">Side View</div>
  </div>
  <div class="controls" id="controls"></div>
</div>
<div id="toast"></div>

<script>
const SN = SERVO_NAMES_JSON;
const GROUPS = GROUPS_JSON;
const PRESETS = PRESETS_JSON;
const ACTIONS = ACTIONS_JSON;
let S = STATE_JSON;  // current state {servo_id: pulse}
let polling = false;

function toast(m) {
    const t = document.getElementById('toast');
    t.textContent = m; t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1200);
}
function setStatus(m) { document.getElementById('status').textContent = m; }

// ---- Buttons ----
const btnDiv = document.getElementById('buttons');
function addBtn(cls, label, fn) {
    const b = document.createElement('button');
    b.className = cls; b.textContent = label; b.onclick = fn;
    btnDiv.appendChild(b); return b;
}
for (const [key, info] of Object.entries(PRESETS)) {
    addBtn('btn-preset', info.label, async () => {
        const r = await fetch('/preset', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({name:key})});
        S = await r.json(); updateUI(); drawAll(); toast(info.label);
    });
}
for (const [key, info] of Object.entries(ACTIONS)) {
    addBtn('btn-action', info.label + ' (' + info.count + ')', async () => {
        setStatus('Playing ' + info.label + '...');
        fetch('/action', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({name:key})});
        startPolling();
    });
}
addBtn('btn-stop', 'STOP', async () => {
    await fetch('/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    setStatus(''); stopPolling();
});

// ---- Polling during action playback ----
let pollTimer = null;
function startPolling() {
    polling = true;
    pollTimer = setInterval(async () => {
        try {
            const r = await fetch('/state');
            const d = await r.json();
            S = d.pulses || d;
            if (d.playing === false) { setStatus(''); stopPolling(); }
            updateUI(); drawAll();
        } catch(e) {}
    }, 250);
}
function stopPolling() { polling = false; clearInterval(pollTimer); }

// ---- Servo controls ----
const ctrlDiv = document.getElementById('controls');
for (const [gName, sids] of GROUPS) {
    const g = document.createElement('div'); g.className = 'group';
    g.innerHTML = '<h2>' + gName + '</h2>';
    for (const sid of sids) {
        const nm = SN[sid] || ('s'+sid);
        const row = document.createElement('div'); row.className = 'servo-row';
        row.innerHTML =
            '<div class="servo-label">' + nm + ' <span>#'+sid+'</span></div>' +
            '<input type="range" min="0" max="1000" id="sl_'+sid+'">' +
            '<input type="number" min="0" max="1000" id="nm_'+sid+'">';
        g.appendChild(row);
        const sl = row.querySelector('input[type=range]');
        const nm2 = row.querySelector('input[type=number]');
        let tm = null;
        function sv(v) {
            v = Math.max(0,Math.min(1000,v)); S[sid]=v; sl.value=v; nm2.value=v;
            drawAll();
            clearTimeout(tm);
            tm = setTimeout(() => {
                fetch('/servo',{method:'POST',headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({id:sid,pulse:v})});
            }, 50);
        }
        sl.addEventListener('input', () => sv(parseInt(sl.value)));
        nm2.addEventListener('change', () => sv(parseInt(nm2.value)));
        nm2.addEventListener('keydown', e => {
            if(e.key==='ArrowUp'){e.preventDefault();sv((parseInt(nm2.value)||0)+10);}
            if(e.key==='ArrowDown'){e.preventDefault();sv((parseInt(nm2.value)||0)-10);}
        });
    }
    ctrlDiv.appendChild(g);
}
function updateUI() {
    for (const [sid,val] of Object.entries(S)) {
        const sl=document.getElementById('sl_'+sid), nm=document.getElementById('nm_'+sid);
        if(sl)sl.value=val; if(nm)nm.value=val;
    }
}
updateUI();

// ============================================================
// VISUALIZATION — 3D coordinates + cabinet projection
// ============================================================
const DEG = Math.PI / 180;
const PI = Math.PI;

// 3D coords: X=right(viewer), Y=down, Z=forward(into screen)
// Cabinet projection: depth(Z) shows as diagonal offset
const OBX = 0.30, OBY = -0.22;
function proj(x,y,z) { return [x + z*OBX, y + z*OBY]; }

// --- Servo → angle helpers ---
function lShoRoll()  { return Math.max(0, (830-(S[15]||830))*0.2727) * DEG; }
function rShoRoll()  { return Math.max(0, ((S[16]||170)-170)*0.2727) * DEG; }
function lShoPitch() { return (725-(S[13]||835)) * 0.08 * DEG; }
function rShoPitch() { return ((S[14]||165)-275) * 0.08 * DEG; }
function lElBend()   { return Math.max(0, Math.min(135, (500-(S[19]||500))*0.24)) * DEG; }
function rElBend()   { return Math.max(0, Math.min(135, ((S[20]||500)-500)*0.24)) * DEG; }
function lElRot()    { return (500-(S[17]||500)) / 750 * 180 * DEG; }
function rElRot()    { return (500-(S[18]||500)) / 750 * 180 * DEG; }
function lHipPitch() { return ((S[7]||350)-350) * 0.12 * DEG; }
function rHipPitch() { return (650-(S[8]||650)) * 0.12 * DEG; }
function lHipRoll()  { return (500-(S[9]||500)) * 0.12 * DEG; }
function rHipRoll()  { return ((S[10]||500)-500) * 0.12 * DEG; }
function lKneeBend() { return Math.max(0, ((S[5]||500)-500)*0.12) * DEG; }
function rKneeBend() { return Math.max(0, (500-(S[6]||500))*0.12) * DEG; }

const COL_L='#4a9eff', COL_R='#ff6b6b', COL_BODY='#888', COL_JOINT='#fff', COL_BG='#0d1b2a';

function drawLimb(ctx,x1,y1,x2,y2,col,w) {
    ctx.strokeStyle=col; ctx.lineWidth=w; ctx.lineCap='round';
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
}
function drawJoint(ctx,x,y,col,r) {
    r=r||4; ctx.fillStyle=col; ctx.beginPath(); ctx.arc(x,y,r,0,PI*2); ctx.fill();
    ctx.strokeStyle='#333'; ctx.lineWidth=1; ctx.stroke();
}
function drawHead(ctx,x,y,r) {
    ctx.fillStyle='#223'; ctx.strokeStyle=COL_BODY; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(x,y,r,0,PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle='#00d4ff';
    ctx.beginPath(); ctx.arc(x-4,y-2,2,0,PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(x+4,y-2,2,0,PI*2); ctx.fill();
}
function drawBg(ctx) {
    const W=ctx.canvas.width, H=ctx.canvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle=COL_BG; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='#1a2a3a'; ctx.lineWidth=0.5;
    for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
}
// (drawHand removed — el_pitch now moves the forearm directly)

// Normalize a 3D vector
function norm3(x,y,z){ const l=Math.sqrt(x*x+y*y+z*z)||1; return [x/l,y/l,z/l]; }

// ---- 3/4 VIEW (main) ----
// Uses cabinet projection so ALL joint angles are visible:
// - sho_roll → lateral arm raise (X) → fully visible
// - sho_pitch → forward arm swing (Z) → visible as diagonal offset
// - el_bend → depends on arm pose, always partially visible
function drawMain(ctx) {
    const W=ctx.canvas.width, H=ctx.canvas.height;
    drawBg(ctx);

    const cx=W/2, headY=45, headR=16;
    const shoY=headY+headR+20, shoSpan=42;
    const hipY=shoY+130, hipSpan=24;
    const uAL=85, fAL=78;  // upper arm, forearm length
    const uLL=95, lLL=88;  // upper leg, lower leg length

    drawHead(ctx, cx, headY, headR);
    // Neck + torso
    const [nsx,nsy] = proj(0,0,0);
    drawLimb(ctx, cx+nsx, headY+headR, cx+nsx, shoY, COL_BODY, 3);
    drawLimb(ctx, cx-shoSpan, shoY, cx+shoSpan, shoY, COL_BODY, 4);
    drawLimb(ctx, cx, shoY, cx, hipY, COL_BODY, 4);
    drawLimb(ctx, cx-hipSpan, hipY, cx+hipSpan, hipY, COL_BODY, 4);

    function drawArm(side, roll, pitch, bend, elRot, col) {
        const s = (side==='L') ? 1 : -1;
        const sx = cx + s*shoSpan, sy = shoY;

        // Upper arm 3D direction (unit vector)
        // After sho_pitch then sho_roll:
        // x = s*sin(roll)*cos(pitch), y = cos(roll)*cos(pitch), z = sin(pitch)
        const ua_x = s * Math.sin(roll) * Math.cos(pitch);
        const ua_y = Math.cos(roll) * Math.cos(pitch);
        const ua_z = Math.sin(pitch);

        // Elbow screen position
        const [edx, edy] = proj(ua_x*uAL, ua_y*uAL, ua_z*uAL);
        const ex = sx+edx, ey = sy+edy;

        // Forearm: elbow bend goes UP in T-pose, FORWARD when arm at side
        // Desired bend dir rotates with roll: (0, -sin(roll), cos(roll))
        // Then project perpendicular to upper arm and normalize
        const raw_y = -Math.sin(roll), raw_z = Math.cos(roll);
        let dt = ua_y*raw_y + ua_z*raw_z;
        let b0x = -ua_x*dt, b0y = raw_y - ua_y*dt, b0z = raw_z - ua_z*dt;
        let bL = Math.sqrt(b0x*b0x + b0y*b0y + b0z*b0z);
        if(bL < 0.01){ b0x=0; b0y=0; b0z=1; bL=1; }
        b0x/=bL; b0y/=bL; b0z/=bL;
        // el_pitch: rotate bend dir around upper arm (Rodrigues)
        const cp_x = ua_y*b0z - ua_z*b0y;
        const cp_y = ua_z*b0x - ua_x*b0z;
        const cp_z = ua_x*b0y - ua_y*b0x;
        const bd_x = b0x*Math.cos(elRot) + cp_x*Math.sin(elRot);
        const bd_y = b0y*Math.cos(elRot) + cp_y*Math.sin(elRot);
        const bd_z = b0z*Math.cos(elRot) + cp_z*Math.sin(elRot);
        // Forearm = upper_arm*cos(bend) + bend_dir*sin(bend)
        let fa_x = ua_x * Math.cos(bend) + bd_x * Math.sin(bend);
        let fa_y = ua_y * Math.cos(bend) + bd_y * Math.sin(bend);
        let fa_z = ua_z * Math.cos(bend) + bd_z * Math.sin(bend);
        const [nx,ny,nz] = norm3(fa_x, fa_y, fa_z);

        const [fdx, fdy] = proj(nx*fAL, ny*fAL, nz*fAL);
        const wx = ex+fdx, wy = ey+fdy;

        drawLimb(ctx, sx,sy, ex,ey, col, 5);
        drawLimb(ctx, ex,ey, wx,wy, col, 4);
        drawJoint(ctx, sx, sy, COL_JOINT, 5);
        drawJoint(ctx, ex, ey, COL_JOINT);
        drawJoint(ctx, wx, wy, col, 3);
    }

    drawArm('L', lShoRoll(), lShoPitch(), lElBend(), lElRot(), COL_L);
    drawArm('R', rShoRoll(), rShoPitch(), rElBend(), rElRot(), COL_R);

    function drawLeg(side, roll, pitch, knee, col) {
        const s = (side==='L') ? 1 : -1;
        const hx = cx + s*hipSpan, hy = hipY;

        // Upper leg 3D: roll=lateral tilt, pitch=forward swing
        const ul_x = s * Math.sin(roll);
        const ul_y = Math.cos(roll) * Math.cos(pitch);
        const ul_z = Math.cos(roll) * Math.sin(pitch);

        const [kdx,kdy] = proj(ul_x*uLL, ul_y*uLL, ul_z*uLL);
        const kx = hx+kdx, ky = hy+kdy;

        // Lower leg: knee bends BACKWARD (-Z) in the leg's sagittal plane
        // When roll=0: bend goes backward (-Z)
        // When roll>0: bend stays mostly backward
        let ll_x = ul_x * Math.cos(knee);
        let ll_y = ul_y * Math.cos(knee) + Math.sin(Math.abs(roll)) * Math.sin(knee);
        let ll_z = ul_z * Math.cos(knee) - Math.cos(Math.abs(roll)) * Math.sin(knee);
        const [nx,ny,nz] = norm3(ll_x, ll_y, ll_z);

        const [adx,ady] = proj(nx*lLL, ny*lLL, nz*lLL);
        const ax = kx+adx, ay = ky+ady;

        drawLimb(ctx, hx,hy, kx,ky, col, 5);
        drawLimb(ctx, kx,ky, ax,ay, col, 4);
        drawJoint(ctx, hx, hy, COL_JOINT, 5);
        drawJoint(ctx, kx, ky, COL_JOINT);
        drawJoint(ctx, ax, ay, col, 3);
    }

    drawLeg('L', lHipRoll(), lHipPitch(), lKneeBend(), COL_L);
    drawLeg('R', rHipRoll(), rHipPitch(), rKneeBend(), COL_R);

    ctx.fillStyle='#555'; ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText('L', cx+shoSpan+14, shoY);
    ctx.fillText('R', cx-shoSpan-14, shoY);
    ctx.fillStyle='#333'; ctx.fillText('3/4 View (depth shown as diagonal)', cx, H-6);
}

// ---- SIDE VIEW ----
function drawSide(ctx) {
    const W=ctx.canvas.width, H=ctx.canvas.height;
    drawBg(ctx);

    const cx=W/2, headY=45, headR=16;
    const shoY=headY+headR+20;
    const hipY=shoY+130;
    const uAL=85, fAL=78, uLL=95, lLL=88;

    drawHead(ctx, cx, headY, headR);
    drawLimb(ctx, cx, headY+headR, cx, shoY, COL_BODY, 3);
    drawLimb(ctx, cx, shoY, cx, hipY, COL_BODY, 4);

    // Side view: screen X = -Z (forward=left), screen Y = Y (down=down)
    // sho_pitch visible, sho_roll causes foreshortening
    function drawArmS(pitch, roll, bend, rot, col, dash) {
        const uVis = uAL * Math.max(0.15, Math.cos(roll));
        // endPt: angle 0=down, positive=clockwise(right on screen = backward)
        const ex = cx + Math.sin(-pitch)*uVis;
        const ey = shoY + Math.cos(-pitch)*uVis;
        // Elbow bend visible in side = bend * cos(roll). Bends FORWARD (left on screen)
        const sVis = bend * Math.cos(Math.abs(roll));
        const fVis = fAL * Math.max(0.15, Math.cos(bend * Math.sin(Math.abs(roll))));
        const fAng = -pitch - sVis;
        const wx = ex + Math.sin(fAng)*fVis;
        const wy = ey + Math.cos(fAng)*fVis;
        if(dash){ctx.setLineDash([6,4]);}else{ctx.setLineDash([]);}
        drawLimb(ctx, cx,shoY, ex,ey, col, 5);
        drawLimb(ctx, ex,ey, wx,wy, col, 4);
        ctx.setLineDash([]);
        drawJoint(ctx, cx,shoY, COL_JOINT, 5);
        drawJoint(ctx, ex,ey, COL_JOINT);
        drawJoint(ctx, wx,wy, col, 3);
    }
    drawArmS(lShoPitch(), lShoRoll(), lElBend(), lElRot(), COL_L, false);
    drawArmS(rShoPitch(), rShoRoll(), rElBend(), rElRot(), COL_R, true);

    function drawLegS(pitch, roll, knee, col, dash) {
        const uVis = uLL * Math.max(0.25, Math.cos(roll));
        const kx = cx + Math.sin(-pitch)*uVis;
        const ky = hipY + Math.cos(-pitch)*uVis;
        // Knee bends backward (right on screen)
        const ax = kx + Math.sin(-pitch+knee)*lLL;
        const ay = ky + Math.cos(-pitch+knee)*lLL;
        if(dash){ctx.setLineDash([6,4]);}else{ctx.setLineDash([]);}
        drawLimb(ctx, cx,hipY, kx,ky, col, 5);
        drawLimb(ctx, kx,ky, ax,ay, col, 4);
        ctx.setLineDash([]);
        drawJoint(ctx, cx,hipY, COL_JOINT, 5);
        drawJoint(ctx, kx,ky, COL_JOINT);
        drawJoint(ctx, ax,ay, col, 3);
    }
    drawLegS(lHipPitch(), lHipRoll(), lKneeBend(), COL_L, false);
    drawLegS(rHipPitch(), rHipRoll(), rKneeBend(), COL_R, true);

    ctx.font='11px sans-serif';
    ctx.fillStyle=COL_L; ctx.textAlign='left'; ctx.fillText('-- Left', 10, H-18);
    ctx.fillStyle=COL_R; ctx.fillText('--- Right', 10, H-6);
    ctx.fillStyle='#444'; ctx.textAlign='center';
    ctx.fillText('< Forward     Back >', cx, H-6);
}

function drawAll() {
    drawMain(document.getElementById('cvFront').getContext('2d'));
    drawSide(document.getElementById('cvSide').getContext('2d'));
}
drawAll();
</script>
</body>
</html>"""

def build_html():
    servo_names_js = json.dumps(SERVO_NAMES)
    groups_js = json.dumps(GROUPS)
    presets_js = json.dumps({k: {'label': v['label']} for k, v in PRESETS.items()})
    actions_js = json.dumps({k: {'label': v['label'], 'count': v['count']} for k, v in ACTIONS.items()})
    state_js = json.dumps(current_pulses)

    html = HTML_PAGE
    html = html.replace('SERVO_NAMES_JSON', servo_names_js)
    html = html.replace('GROUPS_JSON', groups_js)
    html = html.replace('PRESETS_JSON', presets_js)
    html = html.replace('ACTIONS_JSON', actions_js)
    html = html.replace('STATE_JSON', state_js)
    return html.encode('utf-8')

# ============================================================
# HTTP Handler
# ============================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            data = build_html()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/state':
            resp = {'pulses': current_pulses, 'playing': action_playing}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/servo':
            sid = int(body['id'])
            pulse = max(0, min(1000, int(body['pulse'])))
            current_pulses[sid] = pulse
            send_servo(sid, pulse)
            self._json_ok({'ok': True})

        elif self.path == '/preset':
            name = body.get('name', '')
            if name in PRESETS:
                current_pulses.update(PRESETS[name]['pulses'])
                send_all(current_pulses)
                print('[GUI] Preset: %s' % name, flush=True)
            self._json_ok(current_pulses)

        elif self.path == '/action':
            name = body.get('name', '')
            if name in ACTIONS and not action_playing:
                t = threading.Thread(target=play_action, args=(name,), daemon=True)
                t.start()
            self._json_ok({'started': name})

        elif self.path == '/stop':
            global action_stop
            action_stop = True
            self._json_ok({'stopped': True})
        else:
            self.send_error(404)

    def _json_ok(self, data):
        d = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(d)

# ============================================================
# Main
# ============================================================
def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '0.0.0.0'

if __name__ == '__main__':
    PORT = 9090
    ip = get_ip()

    print('[GUI] Sending initial pose (walk_ready)...', flush=True)
    send_all(current_pulses, 1500)

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print('=' * 50, flush=True)
    print('  AiNex Servo Control GUI v2', flush=True)
    print('  http://%s:%d' % (ip, PORT), flush=True)
    print('=' * 50, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[GUI] Shutting down...', flush=True)
        server.server_close()

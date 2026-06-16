

import sqlite3
import sys

# the id numbers that coorrespond to the names of each servo.
# roll refers to s
SERVO_ID = {
    'l_ank_roll': 1,   'r_ank_roll': 2,
    'l_ank_pitch': 3,  'r_ank_pitch': 4,
    'l_knee': 5,       'r_knee': 6,
    'l_hip_pitch': 7,  'r_hip_pitch': 8,
    'l_hip_roll': 9,   'r_hip_roll': 10,
    'l_hip_yaw': 11,   'r_hip_yaw': 12,
    'l_sho_pitch': 13, 'r_sho_pitch': 14,
    'l_sho_roll': 15,  'r_sho_roll': 16,
    'l_el_pitch': 17,  'r_el_pitch': 18,
    'l_el_yaw': 19,    'r_el_yaw': 20,
    'l_gripper': 21,   'r_gripper': 22,
    'head_pan': 23,    'head_tilt': 24,
}

# Servos that live on the robot but are not stored in the .d6a file
# (the action files only cover Servo1..Servo22, the body and arms).
# These are emitted with a value of None so the pose dict can be filled in
# by hand later.
EMPTY_SERVOS = {'head_pan', 'head_tilt'}


def pulse_to_servos(pulse):
    """Convert a {servo_name: pulse} dict into [[servo_id, pulse], ...].

    Used to turn the *_PULSE pose dicts in pulses.py into the format
    MotionManager.set_servos_position expects. Head servos (and any None
    values) are skipped — the head is driven separately by head.py.
    """
    servos = []
    for name, value in pulse.items():
        if name in EMPTY_SERVOS or value is None:
            continue
        servos.append([SERVO_ID[name], value])
    return servos


def d6a_to_pulse(path, frame_index=0):
    """Read one frame of a .d6a file into a {servo_name: pulse} dict.

    For each servo name in SERVO_ID the value is read from the matching
    ``Servo<id>`` column. Servos in EMPTY_SERVOS (the head) are left as None
    because the action files do not store them.
    """
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(ActionGroup)")
        columns = [info[1] for info in cur.fetchall()]
        cur.execute("SELECT * FROM ActionGroup")
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError("no frames found in %s" % path)
    frame = dict(zip(columns, rows[frame_index]))

    pulse = {}
    for name, servo_id in SERVO_ID.items():
        if name in EMPTY_SERVOS:
            pulse[name] = None
        else:
            pulse[name] = frame.get('Servo%d' % servo_id)
    return pulse


def format_pulse(pulse, var_name='POSE_PULSE'):
    """Render a pulse dict as l_/r_ paired source lines, like STAND_PULSE."""
    names = list(SERVO_ID)
    lines = ['%s = {' % var_name]
    for i in range(0, len(names), 2):
        pair = names[i:i + 2]
        cells = []
        for name in pair:
            value = pulse.get(name)
            value_str = '' if value is None else str(value)
            cells.append("'%s': %s," % (name, value_str))
        lines.append('    ' + '  '.join('%-22s' % c for c in cells).rstrip())
    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 d6a_converter.py path/to/action.d6a [VAR_NAME]',
              file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    var_name = sys.argv[2] if len(sys.argv) > 2 else 'POSE_PULSE'
    print(format_pulse(d6a_to_pulse(path), var_name))

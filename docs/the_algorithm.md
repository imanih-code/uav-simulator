# The Algorithm

## General Architecture

```
Operator ──(command)──> GnuRadioChannel ──(raw signal)──> UAV
                              │                            │
                              │                    (processes command)
                              │                            │
                              v                            v
                        CommChaosAdapter            TelemetryPacket
                         (correlation)           (includes command_log)
                              │                            │
                              └────────── HUD ─────────────┘
                                           │
                                      SENT vs RCVD
```

## Per-Motor Timeout Stabilizer

**File**: `uavsim/entities/uav.py` — `_integrate_physics()`

Each motor tracks `_last_motor_cmd_time[i]`. If `_MOTOR_TIMEOUT = 0.15s` has elapsed since the last command to that motor, it is considered "timed out" and receives P+D correction. Otherwise, it gets raw throttle directly.

```
for each motor i:
  if now - _last_motor_cmd_time[i] >= 0.15s:
    throttle[i] = clip(min_raw + corrections[i], 0, 1)
  else:
    throttle[i] = _raw_throttle[i]
```

**P+D Correction** (timed-out motors only):
```
roll_corr  = clamp(rpy[0] * 0.15 + w[0] * 0.05, -0.10, +0.10)
pitch_corr = clamp(rpy[1] * 0.15 + w[1] * 0.05, -0.10, +0.10)

M0 = +roll_corr + pitch_corr    (FR)
M1 = +roll_corr - pitch_corr    (BR)
M2 = -roll_corr - pitch_corr    (BL)
M3 = -roll_corr + pitch_corr    (FL)
```

**Hover controller**: `-max(vel_z, 0) * 0.4` — only brakes ascent, does not accelerate descent.

## GNU Radio in Isolated Subprocess

**File**: `uavsim/comms/gnuradio_link.py`

GNU Radio 3.10 only uses TPB (thread-per-block). Its internal threads conflict with Pygame's OpenGL context (GLXBadContextState). Solution: `multiprocessing.Process` with `spawn` context.

```
Main process:              GNU Radio subprocess:
  Pygame + OpenGL           import gnuradio
  HUD                       _run_burst() → top_block.run()
  UAV logic                 TPB threads isolated
```

- `_run_burst()` runs in the subprocess → **no GLX conflict**
- `_job_queue` and `_result_queue` are `multiprocessing.Queue`
- Raw samples travel serialized via `pickle`

## Sensors & Telemetry

**File**: `uavsim/comms/telemetry.py`

The UAV sends packets every 0.1s as a binary struct:

| Field | Type |
|---|---|
| timestamp | double |
| position (xyz) | 3× double |
| velocity (xyz) | 3× double |
| attitude_rpy | 3× double |
| angular_velocity | 3× double |
| motor_throttle | 4× double |
| battery_percent | double |
| mass | double |
| health_percent | double |
| command_log | 10× int |

`command_log` encodes (opcode, motor_id, valid) into one int32.

## SENT vs RCVD Command Log

**Files**: `operator.py`, `hud.py`, `renderer.py`

The operator stores every sent command in `sent_log` (deque maxlen=10). The UAV includes its `_command_log` in every telemetry packet.

```
SENT        RCVD
THR+1       THR+1     ← green (match)
THR+2       —         ← orange (sent, lost)
—           BAD       ← red (CRC corrupt)
```

- SENT: green if it matches RCVD, orange otherwise
- RCVD: green if valid, red if BAD

## CommChaosAdapter (Signal Correlation)

**File**: `uavsim/comms/comm_chaos_adapter.py`

Normalized cross-correlation against known patterns:

1. `learn(label, samples)` — stores normalized pattern
2. `match(samples)` — correlates, returns `(label, confidence)`
3. Confidence threshold: 0.3

Does not use FFT because bursts are too short (~32 samples) and noise is AWGN, where cross-correlation is already optimal.

## TTF Font with OpenGL Textures

**Files**: `uavsim/rendering/ttf_font.py`, `assets/fonts/7-segment.ttf`

- Each glyph is rendered to a Pygame surface, uploaded as `GL_RGBA` texture
- Drawn with textured quads (alpha for transparency)
- Color applied multiplicatively: white texture × `glColor3f`
- Size 14 for general HUD, 28 for pause overlay

## Current State (unpushed commits)

```
1ad58d2 feat: add CommChaosAdapter for raw signal pattern matching
2f2476c fix: correct QueueFull exception and increase queue buffer to 1024
b9234cd feat: double PAUSED font size to 28pt for better visibility
a4238fc fix: add semi-transparent background behind PAUSED overlay for readability
8f2b44c feat: add max-throttle line and blinking MAX indicator to throttle bars
aa00daa feat: wire TTFFont into HUD renderer, replacing stroke font
```

## Pending

- Wire CommChaosAdapter into the receive flow (HUD)
- Show match confidence in the signal panel
- Adjust TTF text positions if misaligned

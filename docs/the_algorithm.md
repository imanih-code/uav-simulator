# The Algorithm

## General Architecture

```
pilot (keyboard / gamepad)
   │ sticks + switches
   v
UAVOperator ──RC frame 50 Hz──> radio link ──> UAV
                                (GNU Radio      │ LinkMonitor (LQ, time since last frame)
                                 or numpy)      │ arming logic
                                                │ Failsafe (stages, HOLD/LAND/RTH/DROP)
                                                │ FlightController (cascade + mixer)
                                                │ Motors (spin-up lag, thrust ∝ speed²)
                                                v RigidBody (6DOF) + ground contact
HUD <── UAVOperator <──telemetry 10 Hz── radio link <── TelemetryPacket
```

## RC Uplink Protocol

**File**: `uavsim/control/rc.py`

Every frame carries the *absolute* stick positions (11 bytes + CRC32 on
the link): type, sequence number, throttle 0..1000, roll/pitch/yaw
-500..500, and a flags byte (arm switch, flight mode, failsafe action).
The operator sends one every 20 ms whether or not anything changed. A lost
frame costs nothing permanent: the next one carries the full state again.
The sequence numbers / arrival times give the receiver its link quality.

## Flight Controller

**File**: `uavsim/control/flight_controller.py`

```
position --P--> velocity --PI--> acceleration --> tilt + collective thrust
                                        attitude error (quaternion) --P--> rate setpoint
                                        rate error --PI--> angular accel --> torque = I·α + ω×Iω
                                        mixer: [T, τx, τy, τz] = A·f  ->  f = A⁻¹·[...]
                                        motor command = sqrt(f / f_max)
```

| Mode | Sticks mean | Loops running |
|---|---|---|
| ACRO | body rates | rate |
| ANGLE | tilt angle, raw thrust | attitude + rate |
| ALT_HOLD | tilt angle, climb rate | altitude + attitude + rate |
| POS_HOLD | horizontal velocity, climb rate | all |
| NAV (failsafe only) | 3D target / descent rate | all |

The mixer is built from the motor positions and spin directions. On
saturation it keeps roll/pitch first, then yaw, then shifts the
collective ("airmode"), so attitude control survives full and zero
throttle. Gains are in acceleration units and multiplied by the inertia,
so they don't need retuning when the airframe mass changes. Altitude and
position holds brake first and latch the setpoint once the vehicle is
nearly stopped; any change of mode or of who is flying clears the latch.

## Failsafe

**File**: `uavsim/control/failsafe.py`

| Time without a valid frame | Stage | Behaviour |
|---|---|---|
| > 0.1 s | HOLD_LAST | keep the last sticks |
| > 0.3 s | STAGE1 | sticks centred; ACRO forced to ANGLE |
| > 1.0 s | STAGE2 | configured action: RTH, HOLD, LAND or DROP |

HOLD_LAST/STAGE1 recover on the next valid frame. STAGE2 recovers only
after 0.5 s of continuous good link **and** a stick deflection > 0.3
from the pilot. RTH climbs to max(15 m, current altitude), flies home at
6 m/s, descends at 1 m/s and disarms on touchdown.

## Fixed-Step Simulation

**File**: `main.py`

Physics + control run at 250 Hz in simulation time with an accumulator;
the renderer runs at up to 60 FPS. All protocol timing (RC rate, LQ
windows, failsafe stages) uses simulation time, so pausing never trips a
failsafe and results don't depend on frame rate.

## RF Environment (link budgets)

**Files**: `uavsim/world/rf.py`, `uavsim/entities/jammer.py`

Every frame, for each direction:

```
PL(d, h)   = 40.2 dB + 10·n(h)·log10(d)      (2.44 GHz, 1 m reference)
n(h)       = 2.0 + 0.9·max(0, 1 - h_max/30 m)  (h_max = higher antenna of the pair)
S          = P_tx + G_tx + G_rx - PL(d_link)
I          = Σ_j (EIRP_j + G_rx - PL(d_j))    summed in mW, only jammers that are on
SINR       = S - 10·log10(N + I)              N = -108 dBm
noise_voltage = 10^(-SINR/20)                 -> GnuRadioChannel / LoopbackChannel
```

| Link | Transmitter | Receiver (where interference counts) |
|---|---|---|
| Uplink (RC) | ground station, 27 dBm | UAV |
| Downlink (telemetry) | UAV, 20 dBm | ground station |

Jammer timing: BARRAGE always on; PULSED on for 30% of 0.25 s; SWEPT on
our channel for 12% of 0.6 s (72 ms). Bursts shorter than the 0.1 s
RX-loss timeout lower LQ without starting a failsafe.

Calibration against the link model (11-byte RC frame): ~90% delivered at
11 dB SINR, ~50% at 9 dB, ~10% at 7 dB. The 174-byte telemetry packet
needs ~1.5 dB more for the same delivery rate.

Not modelled yet: fading/shadowing, terrain blockage, narrowband or
reactive jammers (they need the channel itself to change, not just its
noise level), antenna patterns.

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

The UAV sends a 174-byte packet every 0.1 s (`"!H20d4h4B"`):

| Field | Type |
|---|---|
| seq | uint16 |
| timestamp (sim time) | double |
| position, velocity, attitude_rpy, angular_velocity | 4 × 3 doubles |
| motor commands | 4 doubles |
| battery, mass, health | 3 doubles |
| applied sticks (what the FC is acting on) | 4 int16 |
| uplink LQ %, flight mode, failsafe stage/action, flags (armed, landed) | 4 bytes |

The doubles make it ~16× bigger than an RC frame, which is why telemetry
is the first thing to die inside a jammer. Compacting it is planned.

## ChaoticLayer — Lorenz Attractor (Secret Bridge)

**File**: `uavsim/comms/comm_chaos_adapter.py` — class `ChaoticLayer`

Both Operator and UAV share identical Lorenz parameters and initial
conditions.  Integrating forward produces identical chaotic sequences,
forming a "secret language" that only they share.

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

Default parameters (classic chaos): sigma=10, rho=28, beta=8/3, x₀=y₀=z₀=1

The attractor state persists across calls so the trajectory is continuous.
The `x` component is used as the chaotic carrier.

```
Operator                          UAV
  │                                │
  ├─ ChaoticLayer(x₀,y₀,z₀)       ├─ ChaoticLayer(x₀,y₀,z₀)
  │   step() → {x,y,z}            │   step() → {x,y,z}  ← identical
  │                                │
  ├─ modulate(command, carrier)    ├─ demodulate(signal, carrier)
  │   → raw_samples                │   → command
  │                                │
  └────── GnuRadioChannel ────────┘
```

### Planned (next steps)

- **Modulate**: embed command bits into the chaotic carrier (e.g. x[n])
- **Demodulate**: recover bits using the synchronised carrier from the
  receiver's own attractor
- **Initial conditions as key**: different (x₀,y₀,z₀) tuples produce
  completely different trajectories → natural cryptographic separation
- **Resynchronisation**: periodically send a known sync pattern so the
  receiver can correct drift

## CommChaosAdapter (Signal Correlation)

**File**: `uavsim/comms/comm_chaos_adapter.py` — class `CommChaosAdapter`

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

- Implement `modulate()` / `demodulate()` in ChaoticLayer
- Wire ChaoticLayer into command send/receive flow
- Wire CommChaosAdapter into the receive flow (HUD)
- Show match confidence in the signal panel
- Periodic Lorenz resync between Operator and UAV

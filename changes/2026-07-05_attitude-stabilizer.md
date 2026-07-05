# Attitude Stabilizer — Uncommitted Changes (Jul 5, 2026)

These changes were applied to `uavsim/entities/uav.py` but never committed.
They were reverted to keep the codebase stable while a proper fix is designed.

## Summary

Refactor throttle control to separate user-commanded throttle (`_raw_throttle`)
from automatic corrections, and add a P+D attitude stabilizer that auto-levels
roll/pitch without fighting deliberate user input.

## Changes

### 1. `_raw_throttle` and `_last_pair_cmd_time` (new fields)

Added in `__init__` and `reset()`:

```python
self._raw_throttle: List[float] = [0.0, 0.0, 0.0, 0.0]
self._last_pair_cmd_time: List[float] = [0.0, 0.0]
```

- `_raw_throttle` stores the throttle level set by user commands separately
  from the actual motor throttle, so attitude corrections never cause integral
  windup.
- `_last_pair_cmd_time` stores timestamps per diagonal motor pair (0,2) and
  (1,3) to know which pairs the user is actively controlling.

### 2. `_mark_pair_active(motor_id)`

New method called whenever a `THROTTLE_UP` or `THROTTLE_DOWN` command is
applied. Records `time.time()` for the affected diagonal pair.

### 3. Attitude stabilizer constants

```python
_HOVER_GAIN = 0.4
_ATTITUDE_P_GAIN = 0.4
_ATTITUDE_D_GAIN = 0.12
_ACTIVE_TIMEOUT = 0.15
```

- `_HOVER_GAIN`: opposes upward velocity so the drone settles at hover throttle
  (was already committed, now reorganized).
- `_ATTITUDE_P_GAIN` / `_ATTITUDE_D_GAIN`: P+D gains for roll/pitch correction.
- `_ACTIVE_TIMEOUT`: seconds a diagonal pair stays "active" after the last user
  command (150ms). During this window the stabilizer does not touch that pair.

### 4. `_attitude_corrections() -> np.ndarray`

Computes per-motor throttle deltas (4-element array) to level the airframe:

```python
roll  = clip(rpy[0] * P_GAIN + wx * D_GAIN, -1, 1)
pitch = clip(rpy[1] * P_GAIN + wy * D_GAIN, -1, 1)

return [
    +roll + pitch,   # motor 0 (FR)
    +roll - pitch,   # motor 1 (BR)
    -roll - pitch,   # motor 2 (BL)
    -roll + pitch,   # motor 3 (FL)
]
```

### 5. `_apply_command` refactor

- Now reads `step_up` / `step_down` from `self.motors[0]` and increments
  `_raw_throttle` in steps (clamped to [0,1]).
- Calls `_mark_pair_active` on single-motor commands so the stabilizer knows
  the user is busy.
- `THROTTLE_UP_ALL` / `THROTTLE_DOWN_ALL` iterate over `_raw_throttle`.
- `EMERGENCY_CUT` zeros `_raw_throttle` instead of `motor.throttle`.

### 6. `_integrate_physics` — attitude correction pass

When armed, before hover correction, the stabilizer runs:

```python
if self._armed:
    att_corr = self._attitude_corrections()
    now = time.time()
    for i in range(4):
        pair = 0 if i in (0, 2) else 1
        if now - self._last_pair_cmd_time[pair] >= self._ACTIVE_TIMEOUT:
            target = clip(raw[i] + att_corr[i], 0, 1)
            self.motors[i].throttle = target
        else:
            self.motors[i].throttle = self._raw_throttle[i]
```

- Inactive pairs: throttle = `_raw_throttle + attitude correction` (clamped).
- Active pairs: throttle = `_raw_throttle` (no correction, full user control).

### 7. `reset()` updated

Also resets `_raw_throttle` and `_last_pair_cmd_time` to zero.

---

## Files affected

Only `uavsim/entities/uav.py`.

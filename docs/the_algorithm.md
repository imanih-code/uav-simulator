# The Algorithm

## Arquitectura General

```
Operator ──(Comando)──> GnuRadioChannel ──(señal raw)──> UAV
                              │                              │
                              │                         (procesa comando)
                              │                              │
                              v                              v
                        CommChaosAdapter               TelemetryPacket
                         (correlación)                (incluye command_log)
                              │                              │
                              └────────── HUD ───────────────┘
                                       │
                                   SENT vs RCVD
```

## Per-Motor Timeout Stabilizer

**Archivo**: `uavsim/entities/uav.py` — `_integrate_physics()`

Cada motor tiene un `_last_motor_cmd_time[i]`. Si han pasado `_MOTOR_TIMEOUT = 0.15s` desde el último comando a ese motor, se considera "timeout" y recibe corrección P+D. Si no, recibe el throttle raw directo.

```
para cada motor i:
  si now - _last_motor_cmd_time[i] >= 0.15s:
    throttle[i] = clip(min_raw + corrections[i], 0, 1)
  sino:
    throttle[i] = _raw_throttle[i]
```

**Corrección P+D** (solo en motores timeout):
```
roll_corr  = clamp(rpy[0] * 0.15 + w[0] * 0.05, -0.10, +0.10)
pitch_corr = clamp(rpy[1] * 0.15 + w[1] * 0.05, -0.10, +0.10)

M0 = +roll_corr + pitch_corr    (FR)
M1 = +roll_corr - pitch_corr    (BR)
M2 = -roll_corr - pitch_corr    (BL)
M3 = -roll_corr + pitch_corr    (FL)
```

**Hover controller**: `-max(vel_z, 0) * 0.4` — solo frena ascenso, no acelera caída.

## GNU Radio en Subproceso Aislado

**Archivo**: `uavsim/comms/gnuradio_link.py`

GNU Radio 3.10 solo usa TPB (thread-per-block). Sus threads internos chocan con el contexto OpenGL de Pygame (GLXBadContextState). Solución: `multiprocessing.Process` con contexto `spawn`.

```
Proceso principal:          Subproceso GNU Radio:
  Pygame + OpenGL            import gnuradio
  HUD                        _run_burst() → top_block.run()
  lógica del UAV             threads TPB aislados
```

- `_run_burst()` corre en el subproceso → **sin conflicto GLX**
- `_job_queue` y `_result_queue` son `multiprocessing.Queue`
- Los raw_samples viajan serializados con `pickle`

## Sensores y Telemetría

**Archivo**: `uavsim/comms/telemetry.py`

El UAV envía paquetes cada 0.1s con struct binario:

| Campo | Tipo |
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

El `command_log` codifica (opcode, motor_id, valid) en un int32.

## SENT vs RCVD Command Log

**Archivos**: `operator.py`, `hud.py`, `renderer.py`

El operador guarda cada comando enviado en `sent_log` (deque maxlen=10). El UAV incluye su `_command_log` en cada paquete de telemetría.

```
SENT        RCVD
THR+1       THR+1     ← verde (coinciden)
THR+2       —         ← naranja (enviado, perdido)
—           BAD       ← rojo (CRC corrupto)
```

- SENT: verde si coincide con RCVD, naranja si no
- RCVD: verde si válido, rojo si BAD

## CommChaosAdapter (Correlación de Señales)

**Archivo**: `uavsim/comms/comm_chaos_adapter.py`

Correlación cruzada normalizada contra patrones conocidos:

1. `learn(label, samples)` — guarda patrón normalizado
2. `match(samples)` — correlaciona, devuelve `(label, confidence)`
3. Umbral de confianza: 0.3

No usa FFT porque las ráfagas son muy cortas (~32 muestras) y el ruido es AWGN, donde la correlación cruzada ya es óptima.

## Fuente TTF con Texturas OpenGL

**Archivos**: `uavsim/rendering/ttf_font.py`, `assets/fonts/7-segment.ttf`

- Cada glyph se renderiza a surface (Pygame), se sube como textura `GL_RGBA`
- Se dibuja con quads texturizados (alpha para transparencia)
- Color se aplica multiplicando: textura blanca × `glColor3f`
- Tamaño 14 para HUD general, 28 para overlay de pausa

## Estado Actual (commits sin push)

```
1ad58d2 feat: add CommChaosAdapter for raw signal pattern matching
2f2476c fix: correct QueueFull exception and increase queue buffer to 1024
b9234cd feat: double PAUSED font size to 28pt for better visibility
a4238fc fix: add semi-transparent background behind PAUSED overlay for readability
8f2b44c feat: add max-throttle line and blinking MAX indicator to throttle bars
aa00daa feat: wire TTFFont into HUD renderer, replacing stroke font
```

## Pendientes

- Conectar CommChaosAdapter al flujo de recepción (HUD)
- Mostrar confianza del match en el panel de señales
- Ajustar posiciones de texto TTF si quedan desalineadas

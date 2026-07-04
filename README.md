# UAVSIM

Simulador de dron (UAV) en Python. Física real de 4 motores en configuración
X, comunicación por comandos/telemetría codificados en bits, cámara de
mundo libre/seguimiento, y un HUD con indicadores reales + gráficos de
señal, todo renderizado con OpenGL usando solo puntos y líneas (sin
texturas, sin fuentes externas).

Nada del movimiento del UAV está "programado" explícitamente: solo se
calcula empuje por motor, se integra fuerza/torque sobre un cuerpo rígido,
y el desplazamiento (traslación, pitch, roll, yaw) es un resultado emergente
de esa física.

## Instalación

**1. GNU Radio** (no es un paquete de pip -- instalalo con el gestor de paquetes de tu sistema):

```bash
# Ubuntu/Debian
sudo apt-get install gnuradio

# Fedora
sudo dnf install gnuradio gnuradio-devel

# Arch
sudo pacman -S gnuradio

# macOS (Homebrew)
brew install gnuradio

# O, multiplataforma, via conda:
conda install -c conda-forge gnuradio
```

Verificá que quedó instalado y es importable desde el Python que vas a usar:

```bash
python3 -c "from gnuradio import gr, blocks, digital, channels; print(gr.version())"
```

⚠️ **Si te tira un error de NumPy tipo "compiled using NumPy 1.x cannot be run in NumPy 2.x"**:
el GNU Radio de tu distro está compilado contra NumPy 1.x. Bajá la versión de numpy de
ese entorno (`pip install "numpy<2" --force-reinstall`) -- es lo que dice el
`requirements.txt`. Esto es una limitación del paquete de GNU Radio de tu sistema, no
de este proyecto.

**2. El resto de las dependencias, con pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Si usás un venv, asegurate de que GNU Radio sea visible desde adentro (los paquetes de
sistema instalados via apt/dnf/pacman van a `/usr/lib/python3/dist-packages`, que los
venv no siempre heredan -- si el import falla solo dentro del venv, corré `main.py` con
el Python del sistema en vez de un venv, o creá el venv con `--system-site-packages`).

## Ejecutar

```bash
python main.py
```

## Controles

**Motores** — el dron es una X vista desde arriba, con 4 motores. Cada
tecla numérica acelera un motor; la letra justo arriba de ella (en QWERTY)
lo desacelera gradualmente hasta 0.

| Motor         | Acelerar | Desacelerar |
|---------------|----------|-------------|
| Front-right   | `1`      | `Q`         |
| Back-right    | `2`      | `W`         |
| Back-left     | `3`      | `E`         |
| Front-left    | `4`      | `R`         |

El UAV **arranca posado en el suelo, quieto**. Si no le das suficiente
empuje total (los 4 motores combinados superando su peso), simplemente se
queda ahí — no cae, no se hunde. Recién despega cuando tú, como operador,
le das el empuje necesario.

**Cámara de mundo** (nunca primera persona — es un observador externo,
nunca la vista del propio dron):

| Tecla         | Acción |
|---------------|--------|
| Flechas       | Modo FREE: mover la cámara. Modo FOLLOW: orbitar / zoom |
| Mouse         | Rotar la vista (orbitar en FOLLOW, mirar alrededor en FREE) |
| `V`           | Alternar entre FOLLOW (persigue al UAV, tipo GTA) y FREE (cámara de espectador, totalmente libre) |
| `ESC`         | Salir |

Al presionar `V` la cámara no "salta": conserva el punto de vista actual y
solo cambia cómo se controla desde ahí en adelante.

## HUD

Todo lo que muestra el HUD sale de la señal que el UAV transmite
periódicamente (o de las estadísticas propias de los canales), nunca de
leer el UAV directamente — el `HUD` solo conversa con el `UAVOperator`.

- **Batería** (`BAT`, %) — se descarga más rápido cuanto más empuje total
  usan los motores.
- **Altitud** (`ALT`, metros).
- **Ángulo del dron** (`ROL`/`PIT`/`YAW`, grados) — roll, pitch, yaw.
- **Posición horizontal** (`POSX`/`POSY`, metros).
- **Peso** (`MASS`, kg) — masa total = cuerpo + los 4 motores, tratada
  como una sola masa puntual para la física (no se modela la inercia de
  cada motor por separado).
- **Ancho de banda** — dos paneles tipo osciloscopio, `TX` (canal de subida:
  Operator → UAV, comandos) y `RX` (canal de bajada: UAV → Operator,
  telemetría). Cada panel muestra **un solo canal** (nada de multicanal):
  se ve el ancho de banda en bytes/seg y, dibujada como una onda cuadrada
  real (NRZ), la señal que el `CommGatewayOutput` efectivamente codificó y
  transmitió bit a bit — no es una animación decorativa, son los bits
  reales de cada mensaje.
- Barras de throttle por motor (abajo), como ya había antes.

Todo el texto del HUD (letras, números, `%`, `:`, etc.) se dibuja con una
fuente vectorial propia (`rendering/vector_font.py`) hecha 100% de
segmentos de línea — nada de texturas ni archivos de fuente.

## Comunicación: GNU Radio real, no una cola en memoria

El Operator y el UAV son, cada uno, emisor y receptor del otro (el Operator
transmite comandos, el UAV transmite telemetría) y **la señal entre ambos
viaja de verdad por GNU Radio**: modulación GMSK real, un canal simulado
con ruido AWGN (`channels.channel_model`), demodulación real, y
sincronización por access code (`digital.correlate_access_code_tag_bb`).
No es una cola de Python haciéndose pasar por radio.

Cada dirección (`command_link`, `telemetry_link`) es un `GnuRadioChannel`
independiente con su propio hilo de fondo:

```
send(bytes) -> [preámbulo + access code + payload + CRC32 + padding de cola]
             -> digital.gmsk_mod (modulación real)
             -> channels.channel_model (ruido AWGN simulado)
             -> digital.gmsk_demod (demodulación real)
             -> digital.correlate_access_code_tag_bb (sincronización real)
             -> verificación CRC32
             -> bytes recuperados (o el paquete se pierde, como en una radio real)
```

Consecuencias reales de esto, no cosméticas:

- **Los paquetes se pueden perder o corromper**: con ruido bajo (default
  `noise_voltage=0.05`) casi todo llega bien, pero es una radio real, no una
  garantía. Si el access code no sincroniza o el CRC32 falla, el paquete se
  descarta — nunca se le entrega basura a la app (`UAV`/`Operator` ignoran
  silenciosamente los paquetes corruptos, no crashean).
- **La entrega es asíncrona y con jitter real**: cada ráfaga (armar +
  modular + pasar por el canal + demodular) cuesta ~10ms de CPU real, así
  que corre en un hilo de fondo por canal, no bloquea el loop principal. Por
  eso el `UAVOperator` ya no manda un comando por frame: retransmite cada
  combinación (motor, dirección) a un máximo de 20 Hz (`COMMAND_SEND_INTERVAL`),
  como lo haría un transmisor RC real, y muy por debajo del techo medido de
  ~100 ráfagas/seg por hilo.
- **El throttle ya no depende de `dt`**: como los comandos llegan async y
  con jitter (no un tick perfecto por frame como antes), cada `Motor` ahora
  sube/baja un paso fijo (`throttle_step_up`/`throttle_step_down`) por
  comando recibido, en vez de una tasa multiplicada por `dt`.
- **Aparece un poco de arrastre aerodinámico** (`RigidBody.linear_drag_coefficient`
  / `angular_drag_coefficient`): con la entrega perfectamente sincronizada de
  antes, los 4 motores siempre subían exactamente igual y nunca había par de
  giro por asimetría. Con radio real, el jitter entre motores es real (uno
  puede llegar unos milisegundos antes que otro), y sin ningún arrastre eso
  se acumulaba en un giro sin límite. El arrastre es física pasiva (resiste
  cualquier velocidad/giro existente, nunca empuja hacia una actitud
  objetivo) — no es una autoestabilización ni un piloto automático.

## Arquitectura

```
uavsim/
├── comms/
│   ├── command.py         Command + CommandOpcode: protocolo de 1 byte
│   ├── telemetry.py       TelemetryPacket: struct binario (batería + masa)
│   ├── gnuradio_link.py   GnuRadioChannel: el enlace de radio real (GMSK +
│   │                      canal con ruido + demod + CRC), con su propio
│   │                      hilo de fondo
│   └── gateway.py          CommGatewayInput/Output: envuelven un
│                            GnuRadioChannel: Output ve tráfico TX
│                            (transmitido), Input ve tráfico RX (recibido
│                            de verdad, con las pérdidas ya aplicadas)
├── physics/
│   ├── motor.py          Motor: throttle -> empuje + torque de reacción
│   │                      (step fijo por comando, ya no por dt)
│   ├── battery.py        Battery: se descarga según el uso de los motores
│   └── rigid_body.py     RigidBody: 6DOF genérico (numpy + scipy Rotation)
│                          + arrastre aerodinámico lineal/angular
├── world/
│   └── environment.py    World / FlatGroundPlane + apply_ground_contact()
│                          (el UAV no atraviesa el piso; despega solo si
│                          tiene empuje suficiente)
├── entities/
│   ├── uav.py             UAV: 4 motores en X, masa total, batería, aplica
│   │                      comandos, corre física + contacto de suelo, emite
│   │                      telemetría periódica
│   └── operator.py        UAVOperator: teclas -> Command, con rate-limit
│                          de 20Hz por (motor, dirección) hacia el
│                          CommGatewayOutput
├── hud/
│   └── hud.py              HUD: telemetría + ancho de banda + señal cruda,
│                            todo leído desde el Operator
└── rendering/
    ├── window.py            Ventana + input unificado (motores, flechas,
    │                        mouse, toggle de cámara) vía pygame
    ├── camera.py             Camera: modo FOLLOW (orbit) y FREE (espectador)
    ├── vector_font.py        Fuente vectorial (solo GL_LINES)
    └── renderer.py           Dibuja terreno + UAV + HUD completo
```

### Por qué el bug de la cámara pasaba

Antes, el UAV no tenía colisión con el suelo: caía en caída libre desde el
frame 1 (gravedad sin nada que la detenga), y la cámara estaba pegada con
un offset fijo a la posición del UAV. En un par de segundos el dron caía
tan lejos que salía del *far clipping plane* de la perspectiva → la
pantalla se quedaba en el color de fondo (casi negro). Eso también hacía
que el HUD pareciera "no existir": la escena se rompía casi al instante.
Ahora `apply_ground_contact()` frena al UAV en el piso hasta que el propio
empuje lo levanta, y la cámara nunca depende de la orientación del UAV (solo
de su posición), así que aunque el dron dé vueltas por un empuje asimétrico,
la cámara no "gira loca" con él.

### Otras decisiones

- **Protocolo en bits real**: `Command.encode()`/`decode()` empaquetan cada
  orden en 1 byte real, y la UAV decodifica en su `CommGatewayInput`.
- **Telemetría con `struct`**: pack/unpack binario con la librería estándar.
- **Física con `numpy` + `scipy.spatial.transform.Rotation`**: cuaterniones
  de scipy en vez de matrices de rotación hechas a mano.
- **Separación estricta de capas**: `RigidBody` no sabe qué es un motor ni
  qué es el suelo; `Motor` no sabe qué es un UAV; `UAV` no sabe qué es un
  teclado ni una cámara; `HUD` nunca habla con el UAV, solo con lo que el
  `Operator` ya recibió; `Camera` no sabe nada de física ni de comms.

## Próximos pasos (no incluidos todavía)

- Jammers: con el enlace ya siendo GNU Radio real, esto ahora es
  literalmente subirle `noise_voltage` (o `frequency_offset`) a un
  `GnuRadioChannel` según la distancia a un emisor de interferencia — la
  infraestructura ya está.
- Terreno no plano.
- Contacto de suelo más realista (normal force real en vez de clamp).

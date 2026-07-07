Propuesta: DSSS Caótico para Resiliencia Anti-Jamming

## Problema

Hoy, los jammers elevan `noise_voltage` en `channels.channel_model()`. El AWGN corrompe la señal GMSK y los paquetes se pierden (fallo de sync o CRC). El UAV pierde comunicación completamente dentro del radio de un jammer.

No hay nada que el UAV pueda hacer electrónicamente — la SNR es demasiado baja para que el demodulador GMSK recupere los bits.

## Idea central

Si ambas partes (operador y UAV) comparten una secuencia pseudoaleatoria, el transmisor puede **ensanchar** cada bit en N chips. El receptor **correlaciona** la señal recibida contra la misma secuencia. El ruido AWGN, al ser incorrelado con la secuencia, se promedia a cero. La señal, al estar correlada, colapsa en un pico.

La ganancia de procesamiento es:

    G_p = 10 · log10(N)  dB

Cada vez que duplicas N, ganas ~3 dB de SNR. Con N=1024, G_p ≈ 30 dB. Un jammer necesita 1000× más potencia para lograr el mismo efecto.

## Arquitectura propuesta

No se reemplaza GMSK — se envuelve:

    bits → spreading (1 → N chips) → GMSK mod → AWGN → GMSK demod → despreading (correlación) → bits

         ↑ secuencia caótica compartida (Lorenz)          ↑ misma secuencia

### Transmisor

Por cada bit b ∈ {+1, -1}:

1. Generar N chips de la secuencia caótica c[n] desde el Lorenz (ambos lados comparten σ, ρ, β, CI).
2. Multiplicar: x[n] = b · c[n], n = 0 … N-1. Esto ensancha el bit.

Si el bit es +1, se envía la secuencia tal cual. Si es -1, se envía invertida. La señal transmitida es NRZ binaria (±1) a tasa N veces mayor que la tasa de bits.

### Receptor

1. GMSK demod entrega chips ruidosos: r[n] = b · c[n] + w[n].
2. Correlacionar contra c[n]:

       ρ = Σ r[n] · c[n] = b · Σ c[n]² + Σ w[n] · c[n]
         = b · E_c + ξ

   donde E_c es la energía de la secuencia y ξ es AWGN filtrado por c[n].
3. Decisión: sign(ρ).

### Análisis de SNR

A la entrada del despreader:

    SNR_in = E_chip / N₀

A la salida del despreader (para un bit completo de N chips):

    SNR_out = N · SNR_in

Factor N es la ganancia de procesamiento.

Para un jammer que eleva noise_voltage a 0.8, si antes la SNR era insuficiente (p. ej. SNR_in = 1 dB → paquetes perdidos), con N = 256 la SNR_out efectiva es ~25 dB — recuperación perfecta.

## Integración en el simulador existente

### Cambios en gnuradio_link.py

Se añaden dos operaciones alrededor del pipeline GMSK:

```python
def _spread(payload_bits: np.ndarray, sequence: np.ndarray, N: int) -> np.ndarray:
    """Ensancha cada bit replicándolo N veces y multiplicando por ±sequence."""
    chips = np.repeat(payload_bits, N).astype(np.float64)  # 0/1 → repetir N veces
    chips = 2.0 * chips - 1.0                               # mapear a ±1
    chips = chips * np.tile(sequence, len(payload_bits))    # multiplicar por secuencia caótica
    return ((chips + 1.0) / 2.0).astype(np.uint8)           # volver a 0/1 para GMSK
```

```python
def _despread(chips: np.ndarray, sequence: np.ndarray, N: int) -> np.ndarray:
    """Correlaciona bloques de N chips contra la secuencia y decide el bit."""
    chips_f = 2.0 * chips.astype(np.float64) - 1.0          # mapear a ±1
    bits = np.zeros(len(chips) // N, dtype=np.uint8)
    for i in range(len(bits)):
        block = chips_f[i*N : (i+1)*N]
        corr = np.dot(block, sequence)                       # correlación
        bits[i] = 1 if corr > 0 else 0
    return bits
```

### Cambios en el flujo de _run_burst()

1. El payload (bytes) se expande a bits con `np.unpackbits`.
2. Se genera la secuencia Lorenz de longitud N con `ChaoticLayer.generate(N)`.
3. Se llama `_spread(bits, sequence, N)` antes de `vector_source_b`.
4. A la salida del correlator, antes de `np.packbits`, se llama `_despread(chips, sequence, N)`.

### ChaoticLayer como fuente de secuencias

El `ChaoticLayer` existente (Lorenz) ya genera secuencias continuas. Se añade un método:

```python
def get_spreading_sequence(self, N: int) -> np.ndarray:
    """Genera N chips normalizados a ±1 a partir del estado actual del Lorenz."""
    raw = self.generate(N)
    # Binarizar: mediana como umbral para asegurar ≈50% de ±1
    threshold = np.median(raw)
    seq = np.where(raw > threshold, 1.0, -1.0)
    return seq
```

Ambos lados comparten `ChaoticLayer(seed, sigma, rho, beta)` — mismo estado → mismas secuencias.

### Parámetros clave

| Parámetro | Valor propuesto | Efecto |
|---|---|---|
| N (chips/bit) | 64 – 1024 | Mayor N → más ganancia, más latencia |
| Símbolos por chip | 2 (vs 4 por bit antes) | Tasa de símbolos = (bits/s) · N · spc |

Nótese que el ancho de banda aumenta en factor N (conspicuo en los paneles TX/RX del HUD).

## Lo que NO cambia

- No se añaden canales extra. Sigue siendo un enlace SISO.
- No hay realimentación: el receptor no necesita pedir retransmisiones.
- No hay heurísticas: es un receptor por correlación (matched filter óptimo para AWGN).
- Los jammers siguen funcionando exactamente igual — solo que su efecto se reduce en factor N.
- El cifrado (`CommChaosAdapter`) puede aplicarse antes o después del spreading sin conflicto: son capas ortogonales (confidencialidad vs. resiliencia).

## Simulación del efecto

Con los valores actuales del simulador:

- Base: noise_voltage = 0.005 → SNR alta → 0% pérdida
- Jammer centro: noise_voltage = 0.8 → SNR muy baja → ~100% pérdida
- Con DSSS N=256: SNR_out = N · SNR_in = 256 · SNR_in → G_p ≈ 24 dB

Con 24 dB de ganancia, el jammer en el centro se siente como noise_voltage efectivo de 0.8 / 16 = 0.05 — casi como estar fuera del jammer.

## Limitaciones

1. **Latencia**: Cada bit requiere N muestras. A misma tasa de símbolos, la tasa de bits efectiva cae en factor N. Compensación: usar N más pequeño o reducir spc.
2. **Captura de secuencia**: El receptor necesita conocer el inicio de la secuencia. Se puede anteponer un preámbulo conocido (el access code actual sirve).
3. **Sincronización**: Ambos Lorenz deben estar sincronizados. Si se desvían, la correlación cae. Solución: reiniciar el Lorenz periódicamente con una semilla compartida acordada fuera de banda.

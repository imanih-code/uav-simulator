"""Thin, direction-aware wrappers around a `GnuRadioChannel`.

`CommGatewayOutput` is the transmitting endpoint: `send()` hands a payload
to the real GNU Radio link, and its bandwidth/history reflect everything
actually transmitted (airtime used), whether or not it ever arrived.

`CommGatewayInput` is the receiving endpoint: `receive()`/`receive_all()`
pull whatever the channel's background worker has successfully
demodulated and CRC-verified, and its bandwidth/history reflect only what
was actually, successfully received.

Every entity that transmits gets an Output; anything that receives gets an
Input. The UAVOperator holds an Output (commands out) and an Input
(telemetry in). The UAV holds the mirror image.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from uavsim.comms.gnuradio_link import GnuRadioChannel


class CommGatewayOutput:
    """Transmitting endpoint. Entities call `send` to push a raw payload."""

    def __init__(self, channel: GnuRadioChannel) -> None:
        self._channel = channel

    def send(self, payload: bytes) -> None:
        self._channel._transmit(payload)

    def bandwidth_bps(self, window: float = 1.0) -> float:
        return self._channel.tx_bandwidth_bps(window)

    def recent_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._channel.tx_transmissions(window)


class CommGatewayInput:
    """Receiving endpoint. Entities call `receive`/`receive_all` to read
    whatever the channel actually managed to deliver."""

    def __init__(self, channel: GnuRadioChannel) -> None:
        self._channel = channel

    def receive(self) -> Optional[bytes]:
        return self._channel._receive()

    def receive_all(self) -> List[bytes]:
        """Drain every payload currently available, oldest first."""
        packets: List[bytes] = []
        while (packet := self._channel._receive()) is not None:
            packets.append(packet)
        return packets

    def bandwidth_bps(self, window: float = 1.0) -> float:
        return self._channel.rx_bandwidth_bps(window)

    def recent_transmissions(self, window: Optional[float] = None) -> List[Tuple[float, bytes]]:
        return self._channel.rx_transmissions(window)

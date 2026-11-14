"""Small PID controller that works on scalars or numpy vectors."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


class PID:
    """PID with integrator clamping and derivative-on-measurement.

    ``update(error, dt, rate=...)``: when the caller already knows the
    derivative of the *measured* quantity (a gyro rate, a vertical speed),
    passing it as ``rate`` avoids differentiating the error, which would
    kick hard every time the setpoint jumps.
    """

    def __init__(
        self,
        kp: ArrayLike,
        ki: ArrayLike = 0.0,
        kd: ArrayLike = 0.0,
        i_limit: ArrayLike = np.inf,
        size: int = 1,
    ) -> None:
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (size,)).copy()
        self.ki = np.broadcast_to(np.asarray(ki, dtype=float), (size,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (size,)).copy()
        self.i_limit = np.broadcast_to(np.asarray(i_limit, dtype=float), (size,)).copy()
        self._size = size
        self.integral = np.zeros(size)
        self._prev_error: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.integral = np.zeros(self._size)
        self._prev_error = None

    def update(
        self,
        error: ArrayLike,
        dt: float,
        rate: Optional[ArrayLike] = None,
        integrate: bool = True,
    ) -> np.ndarray:
        error = np.atleast_1d(np.asarray(error, dtype=float))
        if integrate and dt > 0.0:
            self.integral = np.clip(
                self.integral + self.ki * error * dt, -self.i_limit, self.i_limit
            )
        if rate is not None:
            derivative = -np.atleast_1d(np.asarray(rate, dtype=float))
        elif self._prev_error is not None and dt > 0.0:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = np.zeros(self._size)
        self._prev_error = error
        return self.kp * error + self.integral + self.kd * derivative

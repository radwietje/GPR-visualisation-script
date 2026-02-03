from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from models import Candidate


def depth_ok(c: Candidate, min_trace: int, max_trace: int) -> bool:
    # Depth is measured along the trace (X) axis
    return min_trace <= c.center_x <= max_trace


def amplitude_ok(c: "Candidate", T1: float) -> bool:
    return c.max_amplitude > T1


def hyperbola_fit_sideways(c: "Candidate") -> Tuple[float, Optional[np.ndarray]]:
    """Fit a right-opening hyperbola in trace (x) vs sample (y)."""
    if len(c.x_indices) < 20:
        return np.inf, None

    x = c.x_indices.astype(float)  # Trace (Depth)
    y = c.y_indices.astype(float)  # Sample (Position)

    x0_guess = np.min(x)
    y0_guess = np.mean(y)

    def sideways_hyperbola(y, x0, y0, a, b):
        return x0 + a * (np.sqrt(1.0 + ((y - y0) / (b + 0.1)) ** 2) - 1.0)

    try:
        bounds = (
            [np.min(x) - 50, np.min(y) - 20, 0.1, 0.1],  # lower bounds
            [np.min(x) + 50, np.max(y) + 20, 2000.0, 500.0],  # upper bounds
        )

        popt, _ = curve_fit(
            sideways_hyperbola,
            y,
            x,
            p0=[x0_guess, y0_guess, 100.0, 20.0],
            bounds=bounds,
            maxfev=5000,
        )

        x_fit = sideways_hyperbola(y, *popt)
        error = np.mean(np.abs(x - x_fit))  # MAE

        return error, popt

    except Exception:
        return np.inf, None

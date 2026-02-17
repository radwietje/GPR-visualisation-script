import numpy as np

class Candidate:
    def __init__(self, pixels: np.ndarray, full_data: np.ndarray, dt: float, dx: float):
        self.pixels = pixels

        # Pixel indices: rows = samples (Y), cols = traces (X)
        self.y_indices = np.array([p[0] for p in pixels])
        self.x_indices = np.array([p[1] for p in pixels])

        self.center_y = np.mean(self.y_indices)
        self.center_x = np.mean(self.x_indices)

        # Approximate size in trace and sample directions
        self.width_trace = np.percentile(self.x_indices, 95) - np.percentile(self.x_indices, 5)
        self.height_sample = np.max(self.y_indices) - np.min(self.y_indices)

        intensities = np.abs(full_data[self.y_indices, self.x_indices])
        self.max_amplitude = np.max(intensities) if len(intensities) > 0 else 0.0

        self.fit_error = None
        self.hyperbola_params = None
        self.apex_x = None
        self.apex_y = None

    def __repr__(self):
        return (
            f"Cand(X~{int(self.center_x)}, Y~{int(self.center_y)}, "
            f"W={self.width_trace:.1f}, Amp={self.max_amplitude:.3g})"
        )
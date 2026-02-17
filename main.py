'''
GPR Visualisation Script Usage
==============================

Single File Processing:
----------------------
1. Set the desired .out file path in SETTINGS["filename"] (e.g., "gprdata/yourfile.out").
2. Ensure process_all = False (default).
3. Run the script. Only the specified file will be processed.

Multiple File Processing:
------------------------
1. Set process_all = True.
2. The script will process all .out files in the gprdata folder, one by one.
3. SETTINGS["filename"] will be overwritten for each file in the folder.

Note: If SETTINGS["filename"] is set to "_", you will be prompted to specify a file for single-file mode.
'''
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, gaussian_filter1d, label
from scipy.io import loadmat

from helpers import amplitude_ok, depth_ok, hyperbola_fit_sideways
from models import Candidate


def estimate_ground_column(
    amplitude_img: np.ndarray,
    max_search_fraction: float = 0.25,
    smooth_sigma: float = 3,
    safety_margin: int = 5,
) -> Tuple[int, int, np.ndarray]:
    """Estimates the ground reflection position in 'sideways' GPR data.

    This function analyzes the mean amplitude along the trace axis (columns)
    to find the first major peak, which is assumed to be the ground reflection.
    A cutoff is established shortly after this peak to ignore ground-level
    and direct-wave signals in subsequent processing.

    Args:
        amplitude_img: 2D numpy array of GPR signal amplitudes (samples x traces).
        max_search_fraction: The fraction of initial traces to search for the
            ground reflection (e.g., 0.25 means the first 25%).
        smooth_sigma: The sigma value for the 1D Gaussian filter applied to the
            column amplitude profile to reduce noise before peak detection.
        safety_margin: The number of traces to add to the detected ground peak
            to define the final cutoff index.

    Returns:
        A tuple containing:
        - ground_col (int): The estimated trace index of the ground reflection.
        - cutoff_col (int): The trace index used as the cutoff for detection.
        - col_profile_smooth (np.ndarray): The smoothed column amplitude profile.
    """
    _, n_traces = amplitude_img.shape

    # Mean amplitude per trace (column profile)
    col_profile = amplitude_img.mean(axis=0)
    col_profile_smooth = gaussian_filter1d(col_profile, sigma=smooth_sigma)

    search_limit = max(10, int(max_search_fraction * n_traces))
    ground_col = int(np.argmax(col_profile_smooth[:search_limit]))
    cutoff_col = min(n_traces - 1, ground_col + safety_margin)

    print(
        f"Estimated ground around trace {ground_col}, "
        f"ignoring traces [0..{cutoff_col}] for detection."
    )

    return ground_col, cutoff_col, col_profile_smooth


def extract_candidates(
    filename: str,
    sigma: float = 2.0,
    ground_search_fraction: float = 0.25,
    ground_margin: int = 5,
) -> Tuple[
    List["Candidate"],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[int],
    Optional[np.ndarray],
]:
    """Loads and preprocesses GPR data to extract potential reflection candidates.

    This function performs the initial stages of the detection pipeline:
    1. Reads a GPR data file (.out HDF5 or .mat).
    2. Applies background removal (mean trace subtraction).
    3. Applies a depth-dependent gain to amplify deeper signals.
    4. Estimates the ground position and defines a cutoff.
    5. Creates a binary mask by thresholding the data.
    6. Uses morphological dilation and connected component labeling to find blobs.
    7. Filters blobs by size and position (must be below ground) to create
       a list of candidates.

    Args:
        filename: Path to the input file (.out HDF5 or .mat).
        sigma: The standard deviation multiplier used to set the detection
            threshold above the background mean.
        ground_search_fraction: Fraction of traces to search for the ground
            reflection. Passed to `estimate_ground_column`.
        ground_margin: Safety margin in traces to add to the ground position for
            the cutoff. Passed to `estimate_ground_column`.

    Returns:
        A tuple containing:
        - candidates (List[Candidate]): A list of potential reflection objects.
        - data_gained (Optional[np.ndarray]): The processed GPR data with gain.
        - binary_mask (Optional[np.ndarray]): The binary image used for detection.
        - cutoff_col (Optional[int]): The calculated ground cutoff trace index.
        - col_profile_smooth (Optional[np.ndarray]): The smoothed column profile.
        Returns None for array values if the file cannot be read.
    """
    print(f"\nReading {filename}...")
    try:
        if filename.endswith('.mat'):
            # Load .mat file
            mat_data = loadmat(filename)
            # Find the GPR data variable (commonly named 'gpr_data', 'data', etc.)
            # Try common names first
            gpr_key = None
            for key in ['gpr_data', 'data', 'GPR_data', 'Data']:
                if key in mat_data:
                    gpr_key = key
                    break
            
            if gpr_key is None:
                # If not found, use the first non-metadata key
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        gpr_key = key
                        break
            
            if gpr_key is None:
                print(f"Error: No data array found in .mat file")
                return [], None, None, None, None
            
            raw_data_3d = np.array(mat_data[gpr_key])
            print(f"Loaded data from key '{gpr_key}' with shape: {raw_data_3d.shape}")
            
            # Data should be [t, X, N] where t=time/depth, X=coordinate, N=line number
            # Process the first line (N=0) as [t, X]
            if raw_data_3d.ndim == 3:
                raw_data = raw_data_3d[:, :, 0].T  # Transpose to (trace, sample)
                print(f"Extracted first line: {raw_data.shape}")
            elif raw_data_3d.ndim == 2:
                raw_data = raw_data_3d.T  # Transpose to (trace, sample)
            else:
                print(f"Error: Unexpected data dimensions {raw_data_3d.ndim}")
                return [], None, None, None, None
            
            dt = 1.0  # Default dt value for .mat files (adjust if needed)
        else:
            # Load .out HDF5 file (original code)
            with h5py.File(filename, "r") as f:
                # Data stored as (sample, trace)
                raw_data = np.array(f["rxs"]["rx1"]["Ey"]).T
                dt = f.attrs["dt"]
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], None, None, None, None

    # Remove average trace (background)
    mean_trace = np.mean(raw_data, axis=1, keepdims=True)
    data_clean = raw_data - mean_trace

    n_samples, n_traces = data_clean.shape

    # Apply gain along trace (depth) axis
    trace_indices = np.arange(n_traces)
    gain_curve = (trace_indices / 100.0) ** 1.5
    data_gained = data_clean * gain_curve[np.newaxis, :]

    amplitude_img = np.abs(data_gained)

    # Ground detection
    _, cutoff_col, col_profile_smooth = estimate_ground_column(
        amplitude_img,
        max_search_fraction=ground_search_fraction,
        smooth_sigma=3,
        safety_margin=ground_margin,
    )

    # Zero out columns up to cutoff_col (ground and shallow noise)
    amplitude_for_detection = amplitude_img.copy()
    amplitude_for_detection[:, : cutoff_col + 1] = 0.0

    # Compute threshold on modified image
    bg_mean = np.mean(amplitude_for_detection)
    bg_std = np.std(amplitude_for_detection)
    threshold = bg_mean + (sigma * bg_std)
    print(f"Threshold: {threshold:.5f} (sigma={sigma})")

    binary_mask = amplitude_for_detection > threshold

    # Dilate to connect faint hyperbola segments (wider in X)
    dil_struct = np.ones((5, 10))
    binary_mask = binary_dilation(binary_mask, structure=dil_struct)

    labeled_array, num_features = label(binary_mask)

    candidates = []
    for label_id in range(1, num_features + 1):
        pixel_coords = np.argwhere(labeled_array == label_id)
        if len(pixel_coords) > 50:  # skip very small blobs
            cand = Candidate(pixel_coords, data_gained, dt, 0)
            # Keep only candidates below the ground cutoff
            if cand.center_x > cutoff_col:
                candidates.append(cand)

    print(f"Found {len(candidates)} raw candidates (below ground cutoff)")

    return candidates, data_gained, binary_mask, cutoff_col, col_profile_smooth


def detect_reflections(
    filename: str, settings: Dict[str, Any]
) -> Tuple[
    List["Candidate"],
    List["Candidate"],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[int],
    Optional[np.ndarray],
]:
    """Runs the full GPR reflection detection pipeline on a single file.

    This function orchestrates the process of loading data, finding candidates,
    and then filtering them based on a set of criteria to identify valid
    hyperbolic reflections.

    Args:
        filename: The path to the GPR data file.
        settings: A dictionary containing all parameters for the detection
            process, including sigma, depth range, width, and fit error.

    Returns:
        A tuple containing:
        - reflections (List[Candidate]): A list of validated reflection objects,
          sorted by amplitude.
        - candidates (List[Candidate]): The raw list of all candidates found
          before the final filtering stage.
        - data_gained (Optional[np.ndarray]): The processed GPR data.
        - binary_mask (Optional[np.ndarray]): The binary detection mask.
        - ground_cutoff (Optional[int]): The detected ground cutoff trace index.
        - col_profile_smooth (Optional[np.ndarray]): The smoothed column profile.
    """
    print("\n" + "=" * 70)
    print("GPR SIDEWAYS HYPERBOLA DETECTOR (WITH GROUND REMOVAL)")
    print("=" * 70)

    (candidates, data_gained, binary_mask, ground_cutoff, col_profile_smooth) = (
        extract_candidates(
            filename,
            sigma=settings["sigma"],
            ground_search_fraction=settings.get("ground_search_fraction", 0.25),
            ground_margin=settings.get("ground_margin", 5),
        )
    )

    if len(candidates) == 0:
        return [], [], data_gained, binary_mask, ground_cutoff, col_profile_smooth

    # Intensity threshold: manual or auto from candidates
    T1_manual = settings.get("intensity_T1", None)
    if T1_manual is None:
        all_amps = np.array([c.max_amplitude for c in candidates])
        T1 = np.percentile(all_amps, 70)  # keep stronger 30%
        print(f"Auto intensity threshold (70th percentile): {T1:.5g}")
    else:
        T1 = T1_manual
        print(f"Manual intensity threshold: {T1:.5g}")

    print(
        f"Depth (trace) range filter: [{settings['min_trace']}, {settings['max_trace']}]"
    )
    print(f"Min width (trace): {settings['min_trace_width']}")
    print(f"Max fit error: {settings['max_fit_error']}")
    print(f"Ground cutoff trace: {ground_cutoff}")
    print("=" * 70)

    reflections = []
    for _, cand in enumerate(candidates):
        print(f"\nChecking {cand}...")

        # 1. Width filter (remove narrow vertical blobs)
        if cand.width_trace < settings["min_trace_width"]:
            print(
                f"  REJECTED: Too narrow ({cand.width_trace:.1f} < {settings['min_trace_width']})"
            )
            continue

        # 2. Depth filter (trace index range)
        if not depth_ok(cand, settings["min_trace"], settings["max_trace"]):
            print("  REJECTED: Center trace out of bounds")
            continue

        # 3. Amplitude filter
        if not amplitude_ok(cand, T1):
            print(f"  REJECTED: Amplitude {cand.max_amplitude:.3g} < T1={T1:.3g}")
            continue

        # 4. Hyperbola shape filter
        err, params = hyperbola_fit_sideways(cand)
        cand.fit_error = err
        cand.hyperbola_params = params
        if params is not None:
            x0, y0, a, b = params
            cand.apex_x = x0
            cand.apex_y = y0
            print(f"  Apex at (trace={x0:.1f}, sample={y0:.1f})")

        print(f"  Fit Error: {err:.2f}")
        if err < settings["max_fit_error"]:
            print("  VALIDATED")
            reflections.append(cand)
        else:
            print("  REJECTED: Poor shape fit")

    # Sort validated reflections by amplitude (strongest first)
    reflections.sort(key=lambda c: c.max_amplitude, reverse=True)

    print("\n" + "=" * 70)
    print(f"RESULT: {len(reflections)} validated reflections (sorted by amplitude)")
    print("=" * 70)

    return (
        reflections,
        candidates,
        data_gained,
        binary_mask,
        ground_cutoff,
        col_profile_smooth,
    )


def visualize_results(
    reflections: List[Candidate],
    all_candidates: List[Candidate],
    data_gained: np.ndarray,
    binary_mask: np.ndarray,
    ground_cutoff: int,
) -> None:
    """Generates and displays a 4-panel plot summarizing the detection results.

    The four panels show:
    1. The processed GPR data with the ground cutoff line.
    2. The binary mask used for blob detection.
    3. All candidate blobs found below the ground line.
    4. The final validated reflections with fitted hyperbolas and apex markers.

    The resulting plot is displayed and saved to 'gpr_sideways_ground_removed.png'.

    Args:
        reflections: A list of the final, validated Candidate objects.
        all_candidates: A list of all Candidate objects found before filtering.
        data_gained: The processed GPR data (2D numpy array).
        binary_mask: The binary mask (2D numpy array) from thresholding.
        ground_cutoff: The trace index separating the ground from the subsurface.
        col_profile_smooth: The smoothed column amplitude profile (currently unused
            in the plot but passed for potential future use).
    """
    _, axes = plt.subplots(2, 2, figsize=(16, 8))

    # 1. Gained data with ground cutoff line - with contrast stretching
    amplitude_data = np.abs(data_gained)
    # Stretch contrast using percentile-based clipping
    vmin = np.percentile(amplitude_data, 0.01)
    vmax = np.percentile(amplitude_data, 99.99)  
    axes[0, 0].imshow(amplitude_data, cmap="rainbow", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].axvline(
        ground_cutoff, color="cyan", linestyle="--", linewidth=2, label="Ground cutoff"
    )
    axes[0, 0].set_title("GPR Data (Sideways View)")
    axes[0, 0].set_xlabel("Trace # (Depth)")
    axes[0, 0].set_ylabel("Sample # (Position)")
    axes[0, 0].legend(loc="upper right")

    # 2. Binary mask used for detection
    axes[0, 1].imshow(binary_mask, cmap="rainbow", aspect="auto")
    axes[0, 1].axvline(ground_cutoff, color="cyan", linestyle="--", linewidth=2)
    axes[0, 1].set_title("Binary mask")
    axes[0, 1].set_xlabel("Trace # (Depth)")
    axes[0, 1].set_ylabel("Sample # (Position)")

    # 3. All raw candidates (below ground)
    axes[1, 0].imshow(np.abs(data_gained), cmap="rainbow", aspect="auto", alpha=0.5)
    for cand in all_candidates:
        axes[1, 0].scatter(cand.x_indices, cand.y_indices, s=1, c="blue", alpha=0.5)
    axes[1, 0].axvline(ground_cutoff, color="cyan", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Candidates Below Ground")
    axes[1, 0].set_xlabel("Trace # (Depth)")
    axes[1, 0].set_ylabel("Sample # (Position)")

    # 4. Validated strongest reflections
    axes[1, 1].imshow(np.abs(data_gained), cmap="rainbow", aspect="auto", alpha=0.5)

    for idx, refl in enumerate(reflections):
        # Scatter candidate points
        axes[1, 1].scatter(refl.x_indices, refl.y_indices, s=2, c="red", alpha=0.7)

        # Fitted hyperbola curve
        if refl.hyperbola_params is not None:
            x0, y0, a, b = refl.hyperbola_params
            y_min, y_max = np.min(refl.y_indices), np.max(refl.y_indices)
            y_plot = np.linspace(y_min, y_max, 200)
            x_plot = x0 + a * (np.sqrt(1.0 + ((y_plot - y0) / (b + 0.1)) ** 2) - 1.0)
            axes[1, 1].plot(x_plot, y_plot, "g-", linewidth=2.0)

            # Mark apex
            axes[1, 1].scatter(x0, y0, c="cyan", s=35, marker="x", zorder=4)

            # Label each reflection
            label_x = float(x0)
            label_y = float(y0)
            axes[1, 1].text(
                label_x,
                label_y,
                f"R{idx}",
                color="yellow",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.2"),
            )

    axes[1, 1].axvline(ground_cutoff, color="cyan", linestyle="--", linewidth=2)
    axes[1, 1].set_title(f"Validated Reflections Below Ground ({len(reflections)})")
    axes[1, 1].set_xlabel("Trace # (Depth)")
    axes[1, 1].set_ylabel("Sample # (Position)")

    plt.tight_layout()
    plt.savefig("visual_gpr_data.png", dpi=150)
    print("\nPlot saved to visual_gpr_data.png")
    plt.show()


if __name__ == "__main__":
    SETTINGS = {
        "filename": "gprdata/Chelton_sand_targets.mat",  # Path to the GPR data file to process
        # Noise sensitivity multiplier: controls detection threshold. Higher values = less sensitive (fewer detections).
        "sigma": 5,
        # Depth range for reflection detection (in trace indices). Ignores reflections above min_trace and below max_trace.
        "min_trace": 50,
        "max_trace": 400,
        # Intensity threshold for filtering by amplitude. None = auto-calculate as 70th percentile of candidate amplitudes.
        "intensity_T1": None,
        # Minimum horizontal width (in traces) for a candidate to be valid. Removes thin vertical noise artifacts.
        "min_trace_width": 5,
        # Maximum allowed error when fitting a hyperbola shape to the candidate. Lower values = stricter shape matching.
        "max_fit_error": 1000,
        # Fraction of traces to search when detecting ground reflection (e.g., 0.25 = search first 25% of traces).
        "ground_search_fraction": 0.25,
        # Number of traces to exclude after the detected ground position to avoid ground-surface artifacts.
        "ground_margin": 5,
    }

    # Set this to True to process all files in gprdata, or False to process only the specified file
    process_all = False

    if process_all:
        gpr_data_folder = Path(__file__).parent / "gprdata"
        for input_file in gpr_data_folder.glob("**/*.mat"):  # Changed to .mat files
            print(f"Processing file {input_file.as_posix()}")
            SETTINGS["filename"] = input_file.as_posix()
            (
                reflections,
                all_candidates,
                data_gained,
                binary_mask,
                ground_cutoff,
                col_profile_smooth,
            ) = detect_reflections(SETTINGS["filename"], SETTINGS)

            if data_gained is None:
                print(f"No data loaded for {input_file}. Skipping.")
                continue
            visualize_results(
                reflections,
                all_candidates,
                data_gained,
                binary_mask,
                ground_cutoff,
            )
    else:
        if SETTINGS["filename"] == "_":
            print("Please specify the input filename in SETTINGS['filename'] (e.g., 'gprdata/yourfile.mat')")
        else:
            print(f"Processing file {SETTINGS['filename']}")
            (
                reflections,
                all_candidates,
                data_gained,
                binary_mask,
                ground_cutoff,
                col_profile_smooth,
            ) = detect_reflections(SETTINGS["filename"], SETTINGS)

            if data_gained is None:
                print("No data loaded. Please check the filename and try again.")
            else:
                visualize_results(
                    reflections,
                    all_candidates,
                    data_gained,
                    binary_mask,
                    ground_cutoff,
                )



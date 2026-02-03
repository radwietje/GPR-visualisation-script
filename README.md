# GPR Hyperbola Detector

## Overview
This project contains a Python script (`main.py`) designed to analyze Ground Penetrating Radar (GPR) data. It detects hyperbolic reflections, indicative of subsurface objects, within HDF5 data files. The tool is specialized for a "sideways" data orientation where depth is measured along the trace axis.

## Functionality

### 1. Data Loading & Preprocessing
- **Input**: Reads HDF5 files containing electromagnetic field data (specifically `['rxs']['rx1']['Ey']`).
- **Background Removal**: Subtracts the mean trace to remove static background noise.
- **Gain**: Applies a depth-dependent gain curve to enhance deeper signals.
- **Ground Detection**: Automatically estimates the ground surface position based on amplitude profiles to ignore shallow surface reflections.

### 2. Candidate Extraction
- **Thresholding**: Calculates a dynamic threshold based on image statistics (`sigma`) to identify regions of interest. 
- **Segmentation**: Uses morphological dilation and connected component labeling to group pixels into candidate "blobs".
- **Filtering**: Discards candidates that are too small or located above the estimated ground level.

### 3. Reflection Analysis
- **Hyperbola Fitting**: Fits a hyperbolic curve to each candidate. The model assumes a "sideways" orientation (Trace vs. Sample), where the trace index represents depth.
- **Validation**: Filters candidates based on:
  - **Width**: Minimum trace width.
  - **Depth**: Specific trace index ranges.
  - **Amplitude**: Intensity thresholds.
  - **Shape**: Goodness of fit (Mean Absolute Error) to the hyperbola model.

### 4. Visualization
The script generates a summary image (`gpr_sideways_ground_removed.png`) containing four panels:
1.  **GPR Data**: Processed data with the ground cutoff line.
2.  **Binary Mask**: The thresholded detection mask.
3.  **Candidates**: All detected blobs below the ground line.
4.  **Reflections**: Final validated reflections with fitted hyperbola curves and apex markers.

## Dependencies
- `numpy`
- `h5py`
- `scipy`
- `matplotlib`

See also `pyproject.toml`.

## Usage
1.  Place your `.out` or `.h5` GPR data files in the `gpr data` folder (included in the repository).
2.  Open `main.py`.
3.  The script will automatically process all `*.out` files found in the `gpr data` folder and its subdirectories.
4.  Adjust detection parameters in the `SETTINGS` dictionary as needed:
    - `sigma`: Controls detection sensitivity (lower = more candidates, higher = stricter).
    - `min_trace` / `max_trace`: Sets the depth range of interest (in trace indices).
    - `max_fit_error`: Determines how strictly candidates must resemble a hyperbola.
    - `ground_search_fraction`: Fraction of traces to search for ground reflection (default: 0.25).
    - `ground_margin`: Extra traces to ignore after ground detection (default: 5).
5.  Run the script:
```bash
    python main.py
```
6.  Check the console for detection logs and open the generated PNG file to view results.

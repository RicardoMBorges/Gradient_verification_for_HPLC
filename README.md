# Gradient_verification_for_HPLC
We will explore the process of calculating and visualizing the usage of solvents in a preparative High-Performance Liquid Chromatography (HPLC) system. Our focus will be on the dynamic changes in solvent B's percentage over the course of the HPLC run, understanding the gradient profile, and calculating the total volumes of solvents A and B used.


---

## Repository Contents: *Gradient_verification_for_HPLC*

From your uploads and GitHub link, the repository is organized around **processing HPLC gradient data** and verifying chromatograms against solvent gradients.

### 1. `requirements.txt`

Contains minimal dependencies:

* `numpy`
* `pandas`
* `plotly`
* `matplotlib`

---

### 2. `gradient_processing.py`

This is the **core processing module**, with functions for:

* **Data extraction and combination**

  * `extract_data(file_path)` → pulls retention time + intensity data from `.txt` files.
  * `combine_and_trim_data(input_folder, output_folder, retention_time_start, retention_time_end)` → merges chromatograms, trims by retention time, and outputs `combined_data.csv`.

* **Preprocessing**

  * `remove_unwanted_regions(df, start_value, end_value)` → zeroes out specific retention time regions.

* **Gradient calculations**

  * `volumns_HPLC(gradient_segments, flow_rate_mL_min)` → calculates solvent A/B volumes over gradient segments.

* **Visualization**

  * `gradient_plot(gradient)` → interactive Plotly plot of solvent B % vs. time.
  * `plot_gradient_and_chromatogram(gradient, directory_path, start_column, end_column, retention_time_start, retention_time_end)` → overlays gradient profile with chromatogram(s) for comparison.

---

### 3. `run.bat`

Windows batch script to set up environment and run the pipeline (likely calls the notebook or Python script).

---

### 4. `Gradient_verification_for_HPLC_v2.ipynb`

A **Jupyter Notebook** that acts as the main workflow:

* Loads chromatogram data.
* Loads gradient table (start/end times, %B).
* Calls `gradient_processing.py` functions for:

  * Trimming chromatograms,
  * Overlaying gradient profile,
  * Calculating solvent volumes,
  * Producing interactive plots.

---

## How It All Connects

1. Place `.txt` chromatogram files in an input folder.
2. Define retention time window (e.g., 4–30 min).
3. Prepare a **gradient DataFrame** (`start_time`, `end_time`, `start_B%`, `end_B%`).
4. Run:

   * `combine_and_trim_data()` → to merge chromatograms.
   * `gradient_plot()` → to check solvent B program.
   * `plot_gradient_and_chromatogram()` → to overlay chromatograms with gradient.
   * `volumns_HPLC()` → to check actual solvent usage.
5. Outputs:

   * `combined_data.csv` (merged chromatogram)
   * Interactive plots in notebook/Plotly.

---

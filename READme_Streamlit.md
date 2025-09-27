# HPLC Gradient Verification

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/<your-username>/<your-repo>/<branch>/app.py)

This Streamlit app helps you **verify HPLC gradients** against chromatogram data.  
Upload your chromatogram `.txt` files, define or upload a solvent gradient, and the app will:

- Combine chromatograms into a single aligned table  
- Mask retention time regions (set intensity = 0)  
- Overlay chromatograms with the gradient (%B vs time)  
- Calculate solvent A and B consumption per gradient segment and in total  
- Export combined chromatograms as CSV  

---

## How to Use

### Online (recommended)
Simply open the app in your browser:  
👉 [Launch on Streamlit Cloud](https://share.streamlit.io/<your-username>/<your-repo>/<branch>/app.py)

### Local Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
````

Run the app:

```bash
streamlit run app.py
```

---

## 📂 Input Formats

### Chromatogram `.txt` files

* Must contain a header line starting with:

  ```
  R.Time (min)
  ```
* Followed by two columns: **RT (min)** and **Intensity**
* Example:

  ```
  R.Time (min) Intensity
  0.00 123
  0.05 140
  ...
  ```

### Gradient table

* CSV with four columns:

  ```
  start_time,end_time,start_B%,end_B%
  0,5,5,5
  5,20,5,95
  20,25,95,95
  25,30,95,5
  ```
* Or edit the default gradient directly in the app.

---

## 🖼️ Features

* **Interactive plots** (Plotly)

  * Chromatogram intensity vs retention time
  * Gradient profile (%B) interpolated smoothly
  * Overlay of chromatograms and gradient
* **Data export**

  * Download merged chromatograms as `combined_data.csv`
* **Solvent usage**

  * Table with solvent A and B volumes for each segment
  * Totals at the bottom

---

## ⚙️ Requirements

* `streamlit`
* `pandas`
* `numpy`
* `plotly`

Optional:

* `pyarrow` (for Streamlit’s interactive data editor, not required)

---

## 📝 Notes

* Large `.txt` files may take longer to render.
* For deployment, keep logos/images in a `static/` folder inside the repo.
* If `pyarrow` is not installed, the app falls back to a simple CSV editor and HTML table rendering.

---

## 📌 Example

Upload two `.txt` chromatograms + the example gradient:

![Example screenshot](static/example_screenshot.png)

---

## 👥 Authors

Developed by [Ricardo M. Borges](https://github.com/RicardoMBorges) and collaborators.
 LAABio – IPPN – UFRJ.

```

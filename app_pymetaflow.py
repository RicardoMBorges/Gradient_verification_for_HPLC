
import io
import os
import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_processing_HPLC as dp

# Alignment helpers (PAFFT, RAFFT, Icoshift)
from alignment_utils import alignment_controls, align_df

# Optional deps for alignment & modeling
try:
    from pyicoshift import icoshift  # noqa: F401
except Exception:
    icoshift = None

# Guarded scikit-learn imports
SKLEARN_AVAILABLE = True
try:
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
except Exception as _e:
    SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e

# SciPy availability (needed for advanced STOCSY modes in dp)
SCIPY_AVAILABLE = True
try:
    from scipy import stats  # noqa: F401
    from scipy.optimize import curve_fit  # noqa: F401
    from scipy.spatial.distance import cosine  # noqa: F401
    from scipy.fft import fft  # noqa: F401
    from scipy.stats import skewnorm  # noqa: F401
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="pyMETAflow HPLC", layout="wide")
st.title("pyMETAflow HPLC")
st.caption("Parse LabSolutions ASCII, map via metadata, preprocess, align (Icoshift/PAFFT/RAFFT), visualize, STOCSY, and (optionally) model.")

PYARROW_AVAILABLE = True
try:
    import pyarrow 
except Exception:
    PYARROW_AVAILABLE = False

STATIC_DIR = Path(__file__).parent / "static"
LOGO_pyMETAflow_HPLC_PATH = STATIC_DIR / "pyMETAflow_HPLC.png"
try:
    logo = Image.open(LOGO_pyMETAflow_HPLC_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/pyMETAflow_HPLC.png")

# ------------------ Sidebar: Uploads & Preferences ------------------
LOGO_PATH = STATIC_DIR / "LAABio.png"

try:
    logo = Image.open(LOGO_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")

st.markdown(
    """
    Upload chromatogram **.txt** files and a gradient table to compare chromatograms
    against the solvent program, mask unwanted RT regions, and compute solvent A/B volumes.  

    Developed by **Ricardo M Borges** and **LAABio-IPPN-UFRJ**  
    contact: ricardo_mborges@yahoo.com.br  

    🔗 Details: [GitHub repository](https://github.com/RicardoMBorges/Gradient_verification_for_HPLC)

    Check also: [DAFdiscovery](https://dafdiscovery.streamlit.app/)
    
    Check also: [TLC2Chrom](https://tlc2chrom.streamlit.app/)
    """
)

# PayPal donate button
st.sidebar.markdown("""
<hr>
<center>
<p>To support the app development:</p>
<a href="https://www.paypal.com/donate/?business=2FYTFNDV4F2D4&no_recurring=0&item_name=Support+with+%245+→+Send+receipt+to+tlc2chrom.app@gmail.com+with+your+login+email+→+Access+within+24h!&currency_code=USD" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal button" border="0">
</a>
</center>
""", unsafe_allow_html=True)

st.sidebar.markdown("""---""")

TUTORIAL_URL = "https://github.com/RicardoMBorges/Gradient_verification_for_HPLC/blob/main/READme_Streamlit.md"
try:
    st.sidebar.link_button("📘 Tutorial", TUTORIAL_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{TUTORIAL_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        '</a>',
        unsafe_allow_html=True,
    )


MockData_URL = "https://github.com/RicardoMBorges/Gradient_verification_for_HPLC/blob/main/READme_Streamlit.md"
try:
    st.sidebar.link_button("Mock Data", MockData_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{MockData_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">Mock Data</button>'
        '</a>',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("""---""")
# ================== INPUTS & PARSING (ORDERED) ==================

# ---- 1) Upload Chromatograms (.txt) ----
st.sidebar.header("1) Upload chromatograms (.txt)")
mode = st.sidebar.radio(
    "Input mode",
    ["2D LabSolutions ASCII (uploads)", "3D PDA (folder of .txt)", "3D PDA (uploaded .txt)"],
    index=0,
)

# ------------------ Helpers ------------------
HEADER_RE = re.compile(r"^R\.Time \(min\)\s+Intensity\s*$", flags=re.MULTILINE)

def parse_labsolutions_ascii(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    """Return DF with columns: ['RT(min)', <sample_name>] parsed from LabSolutions ASCII .txt (2D)."""
    try:
        text = raw_bytes.decode("latin1", errors="ignore")
    except Exception:
        text = raw_bytes.decode("utf-8", errors="ignore")
    m = HEADER_RE.search(text)
    if not m:
        raise ValueError("Header 'R.Time (min)\\tIntensity' not found in file.")
    table = text[m.start():]
    df = pd.read_csv(io.StringIO(table), sep="\t", decimal=",", engine="python")
    df = df.iloc[:, :2].copy()
    df.columns = ["RT(min)", "Intensity"]
    df = df[pd.to_numeric(df["RT(min)"], errors="coerce").notna()]
    df["RT(min)"] = df["RT(min)"].astype(float)
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    base = os.path.splitext(os.path.basename(file_name))[0]
    return df.rename(columns={"Intensity": base}).reset_index(drop=True)

def outer_join_rt(dfs: dict) -> pd.DataFrame:
    combined = None
    for _, df in dfs.items():
        combined = df if combined is None else combined.merge(df, on="RT(min)", how="outer")
    if combined is not None:
        combined = combined.sort_values("RT(min)").reset_index(drop=True)
    return combined

# Display helper that reads preference from session_state so we can call it before step 4 renders
def show_df(df: pd.DataFrame):
    """Safely display a DataFrame without requiring PyArrow."""
    use_static = st.session_state.get("use_static_table", True)
    if (not use_static) and PYARROW_AVAILABLE:
        st.dataframe(df, use_container_width=True)
    else:
        try:
            html = df.to_html(index=False)
            st.markdown(html, unsafe_allow_html=True)
        except Exception:
            st.text(df.to_string(index=False))

# ------------------ Preprocessing helpers (unchanged) ------------------
def resample_to_grid(df: pd.DataFrame, step: float, rt_min=None, rt_max=None):
    if df is None:
        return None, None
    x = df["RT(min)"].values
    grid_min = float(rt_min) if rt_min is not None else float(np.nanmin(x))
    grid_max = float(rt_max) if rt_max is not None else float(np.nanmax(x))
    if grid_max <= grid_min:
        grid_max = grid_min + 0.01
    grid = np.arange(grid_min, grid_max + step/2, step, dtype=float)
    out = {"RT(min)": grid}
    for c in df.columns:
        if c == "RT(min)":
            continue
        y = df[c].astype(float).values
        y = pd.Series(y).interpolate(limit_direction="both").values
        out[c] = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return pd.DataFrame(out), grid

def moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return arr
    return pd.Series(arr).rolling(window=win, min_periods=1, center=True).mean().values

def baseline_subtract(arr: np.ndarray, method: str, param: float) -> np.ndarray:
    if method == "none":
        return arr
    if method == "median":
        return arr - np.nanmedian(arr)
    if method == "rolling_min":
        w = max(3, int(param))
        s = pd.Series(arr).rolling(window=w, min_periods=1, center=True).min().values
        return arr - s
    return arr

def normalize_trace(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return arr
    a = arr.copy()
    if mode == "max=1":
        m = np.nanmax(np.abs(a))
        return a / m if m else a
    if mode == "area=1":
        s = np.nansum(np.abs(a))
        return a / s if s else a
    if mode == "zscore":
        mu = np.nanmean(a); sd = np.nanstd(a)
        return (a - mu) / sd if sd else a
    return a

def preprocess_matrix(df: pd.DataFrame, smooth_win: int, baseline_method: str, baseline_param: float, norm_mode: str) -> pd.DataFrame:
    if df is None:
        return None
    out = df.copy()
    for c in out.columns:
        if c == "RT(min)":
            continue
        y = out[c].astype(float).values
        y = moving_average(y, smooth_win)
        y = baseline_subtract(y, baseline_method, baseline_param)
        y = normalize_trace(y, norm_mode)
        out[c] = y
    return out

def matrix_to_XY(df_grid: pd.DataFrame, sample_order: list) -> tuple:
    X = []
    for s in sample_order:
        X.append(df_grid[s].values.astype(float))
    X = np.vstack(X)  # samples x points
    feats = [f"{rt:.4f}" for rt in df_grid["RT(min)"].values]
    return X, feats

# ------------------ Build 'combined' by selected mode ------------------
# If we already imported PDA once, keep using it unless explicitly cleared
combined = st.session_state.get("pda_df", None)
pda_meta = st.session_state.get("pda_meta", None)

#combined = None

@st.cache_data(show_spinner="Reading PDA files…")
def _import_3d_cached(folder: str, wl: float, outname: str):
    return dp.import_3D_data(folder, target_wavelength=wl, output_filename=outname)

if mode == "2D LabSolutions ASCII (uploads)":
    uploads_2d = st.sidebar.file_uploader(
        "ASCII export(s) with 'R.Time (min)\\tIntensity' header",
        type=["txt"], accept_multiple_files=True
    )
    parsed = {}
    report_rows = []
    if uploads_2d:
        for f in uploads_2d:
            try:
                raw = f.getvalue()  # robust even if read elsewhere
                df = parse_labsolutions_ascii(f.name, raw)
                parsed[f.name] = df
                report_rows.append({"file": f.name, "status": "parsed", "rows": len(df)})
            except Exception as e:
                report_rows.append({"file": f.name, "status": f"error: {e}", "rows": 0})
    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        st.subheader("Parsing report")
        show_df(report_df)
    combined = outer_join_rt(parsed) if parsed else None

elif mode == "3D PDA (folder of .txt)":
    st.sidebar.markdown("Pick wavelength and a local folder of PDA ASCII files.")
    wl_pick = st.sidebar.number_input("Target wavelength (nm)", value=320.0, step=1.0, key="pda_wl_folder")
    input_folder = st.sidebar.text_input("Local folder path with .txt files", key="pda_folder_path")

    colA, colB = st.sidebar.columns(2)
    do_extract = colA.button("Extract traces at wavelength", key="pda_folder_extract")
    do_clear   = colB.button("Clear current PDA matrix", key="pda_clear")

    if do_clear:
        st.session_state.pop("pda_df", None)
        st.session_state.pop("pda_meta", None)
        st.success("Cleared PDA matrix from memory.")

    if do_extract:
        if input_folder and os.path.isdir(input_folder):
            combined_3d = _import_3d_cached(
                input_folder, float(wl_pick), f"combined_{int(wl_pick)}nm.csv"
            )

            if combined_3d is None or combined_3d.empty:
                st.error("No data combined from the folder.")
            else:
                st.session_state["pda_df"] = combined_3d
                st.session_state["pda_meta"] = {"src":"folder","path":input_folder,"wl":float(wl_pick)}
                combined = combined_3d  # available this run too
                st.success(f"Built & cached matrix from {combined_3d.shape[1]-1} files at ≈{wl_pick} nm")
        else:
            st.error("Please enter a valid existing folder path.")

    # Show current matrix if we have one
    if st.session_state.get("pda_df") is not None:
        meta = st.session_state.get("pda_meta", {})
        st.caption(f"Current PDA matrix (source={meta.get('src','?')}, wl≈{meta.get('wl','?')} nm)")
        show_df(st.session_state["pda_df"].head(10))
        combined = st.session_state["pda_df"]


elif mode == "3D PDA (uploaded .txt)":
    uploads_pda = st.sidebar.file_uploader(
        "PDA ASCII .txt exports (multi-wavelength 3D tables)",
        type=["txt"], accept_multiple_files=True, key="pda_uploads"
    )
    wl_pick_up = st.sidebar.number_input("Target wavelength (nm) for uploads",
                                         value=320.0, step=1.0, key="pda_wl_uploads")
    colA, colB = st.sidebar.columns(2)
    do_extract = colA.button("Extract wavelength from uploads", key="pda_upload_extract")
    do_clear   = colB.button("Clear current PDA matrix", key="pda_upload_clear")

    if do_clear:
        st.session_state.pop("pda_df", None)
        st.session_state.pop("pda_meta", None)
        st.success("Cleared PDA matrix from memory.")

    if uploads_pda and do_extract:
        import tempfile, shutil, os
        tmpdir = tempfile.mkdtemp(prefix="pymetaflow_pda_")
        try:
            for f in uploads_pda:
                data = f.getvalue()
                with open(os.path.join(tmpdir, os.path.basename(f.name)), "wb") as out:
                    out.write(data)
            combined_3d = _import_3d_cached(
                tmpdir, float(wl_pick_up), f"combined_{int(wl_pick_up)}nm.csv")
            if combined_3d is None or combined_3d.empty:
                st.warning("No valid 3D tables found in the uploaded files.")
            else:
                st.session_state["pda_df"] = combined_3d
                st.session_state["pda_meta"] = {
                    "src":"uploads",
                    "wl":float(wl_pick_up),
                    "files":[f.name for f in uploads_pda]
                }
                combined = combined_3d
                st.success(f"Built & cached matrix from {combined_3d.shape[1]-1} uploads at ≈{wl_pick_up} nm")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if st.session_state.get("pda_df") is not None:
        meta = st.session_state.get("pda_meta", {})
        st.caption(f"Current PDA matrix (source={meta.get('src','?')}, wl≈{meta.get('wl','?')} nm)")
        show_df(st.session_state["pda_df"].head(10))
        combined = st.session_state["pda_df"]


# ---- 2) Optional: Upload metadata (.csv) ----
st.sidebar.header("2) Optional: Upload metadata (.csv)")
meta_file = st.sidebar.file_uploader("Metadata CSV (semicolon ';' as delimiter)", type=["csv", "txt"])

# ---- 3) Optional: Upload bioactivity (.csv) ----
st.sidebar.header("3) Optional: Upload bioactivity (.csv)")
bio_file = st.sidebar.file_uploader("Bioactivity CSV", type=["csv"])

# ---- 4) Display preferences ----
st.sidebar.header("4) Display preferences")
use_static_table = st.sidebar.checkbox("Use static table (avoid PyArrow)", value=True, key="use_static_table")

# ------------------ Metadata / Bioactivity tables ------------------
meta_df = None
if meta_file is not None:
    try:
        meta_df = pd.read_csv(meta_file, sep=";")
    except Exception as e:
        st.error(f"Failed to read metadata: {e}")
if meta_df is not None:
    st.subheader("Metadata (head)")
    show_df(meta_df.head(3))

bio_df = None
if bio_file is not None:
    try:
        bio_df = pd.read_csv(bio_file)
        st.subheader("Bioactivity (head)")
        show_df(bio_df.head(3))
    except Exception as e:
        st.error(f"Failed to read bioactivity: {e}")

# ------------------ Column mapping & rename ------------------
if meta_df is not None:
    cols = meta_df.columns.tolist()
    col_sample = st.selectbox(
        "Sample ID column (matches chromatogram columns)",
        options=cols,
        index=cols.index("Samples") if "Samples" in cols else 0
    )
    default_hplc = "HPLC_filename" if "HPLC_filename" in cols else ("MS_filename" if "MS_filename" in cols else cols[0])
    col_hplc = st.selectbox(
        "Metadata column containing HPLC file stems",
        options=cols,
        index=cols.index(default_hplc)
    )
    if "BioAct_filename" in cols:
        col_bio = st.selectbox(
            "Metadata column with BioActivity file stems (optional)",
            options=["<none>"] + cols,
            index=(cols.index("BioAct_filename") + 1)
        )
        if col_bio == "<none>":
            col_bio = None
    else:
        st.caption("Tip: Add a 'BioAct_filename' column in metadata to enable BioActivity fusion.")
        col_bio = None
else:
    col_sample, col_hplc, col_bio = None, None, None

# Rename combined columns according to metadata mapping (if we have a matrix)
if combined is not None:
    file_stems = [c for c in combined.columns if c != "RT(min)"]
    if meta_df is not None and col_sample and col_hplc:
        name_map = {}
        for _, row in meta_df.iterrows():
            stem = str(row[col_hplc]).replace(".txt", "")
            name_map[stem] = str(row[col_sample])
        ren = {stem: name_map[stem] for stem in file_stems if stem in name_map}
        if ren:
            combined = combined.rename(columns=ren)

# ------------------ Preprocessing controls ------------------
st.markdown("---")
st.subheader("Preprocessing")
c1, c2, c3, c4 = st.columns(4)
with c1:
    grid_step = st.number_input("Uniform grid step (min): Defines the time resolution after resampling chromatograms", value=0.02, min_value=0.001, step=0.001, format="%.3f")
with c2:
    smooth_win = st.number_input("Smoothing window (pts): Number of points for moving average smoothing.", value=1, min_value=1, step=1)
with c3:
    baseline_method = st.selectbox("Baseline: Method for subtracting background signal.", options=["none", "median", "rolling_min"], index=1)
with c4:
    baseline_param = st.number_input("Baseline param (for rolling_min): Window size (number of points) for computing the rolling minimum.", value=101, min_value=3, step=2)

c5, c6 = st.columns(2)
with c5:
    norm_mode = st.selectbox("Normalization", options=["none", "max=1", "area=1", "zscore"], index=2)
with c6:
    rt_range = st.text_input("RT range (min,max) or blank", value="")
    rt_min = rt_max = None
    if rt_range.strip():
        try:
            parts = [float(x) for x in rt_range.split(",")]
            if len(parts) == 2:
                rt_min, rt_max = parts
        except Exception:
            st.warning("RT range not parsed. Use format like: 0.5,45")

df_grid = None
if combined is not None:
    df_grid, _grid = resample_to_grid(combined, step=float(grid_step), rt_min=rt_min, rt_max=rt_max)
    df_grid = preprocess_matrix(df_grid, int(smooth_win), baseline_method, float(baseline_param), norm_mode)

# ---------------- RT CLIPPING (before alignment & referencing) ----------------
st.markdown("---")
st.subheader("Clip RT Range")

# Choose which matrix to operate on: prefer the resampled grid; else the combined raw
clip_target_name = "df_grid" if ('df_grid' in locals() and df_grid is not None) else "combined"
base_df = df_grid if clip_target_name == "df_grid" else combined

if base_df is None:
    st.caption("Load data to enable RT clipping.")
else:
    axis_col = base_df.columns[0]  # usually "RT(min)"

    # Build a light signature of the current base_df so we can detect changes
    def _df_signature(df: pd.DataFrame, axis_col: str):
        cols = tuple(df.columns)
        n = len(df)
        lo = float(df[axis_col].iloc[0]) if n else np.nan
        hi = float(df[axis_col].iloc[-1]) if n else np.nan
        return (cols, n, lo, hi)

    full_key = f"{clip_target_name}_FULL"
    sig_key  = f"{clip_target_name}_FULL_SIG"
    bounds_key = f"rt_bounds_{clip_target_name}"

    current_sig = _df_signature(base_df, axis_col)

    # (Re)cache FULL when missing or signature changed
    if (full_key not in st.session_state) or (sig_key not in st.session_state) or (st.session_state[sig_key] != current_sig):
        st.session_state[full_key] = base_df.copy()
        st.session_state[sig_key]  = current_sig
        # reset bounds to full whenever upstream matrix changes
        _df = st.session_state[full_key]
        st.session_state[bounds_key] = (float(_df[axis_col].min()), float(_df[axis_col].max()))

    full_df = st.session_state[full_key]
    rt_min_full = float(full_df[axis_col].min())
    rt_max_full = float(full_df[axis_col].max())

    # Restore saved bounds if present; otherwise default to full range
    if bounds_key not in st.session_state:
        st.session_state[bounds_key] = (rt_min_full, rt_max_full)
    saved_lo, saved_hi = st.session_state[bounds_key]

    # AUTO-APPLY the saved bounds to produce the ACTIVE (clipped) df every rerun
    active_df = dp.filter_rt_range(full_df, float(saved_lo), float(saved_hi), axis_column=axis_col)
    if clip_target_name == "df_grid":
        df_grid = active_df
    else:
        combined = active_df

    # UI controls reflect the current (saved) bounds
    col_a, col_b = st.columns(2)
    with col_a:
        start_rt = st.number_input(
            "Start RT (min)",
            value=float(saved_lo), min_value=rt_min_full, max_value=rt_max_full,
            step=0.1, format="%.3f"
        )
    with col_b:
        end_rt = st.number_input(
            "End RT (min)",
            value=float(saved_hi), min_value=rt_min_full, max_value=rt_max_full,
            step=0.1, format="%.3f"
        )

    # Clamp in case widgets return tiny float outside
    start_rt = float(min(max(start_rt, rt_min_full), rt_max_full))
    end_rt   = float(min(max(end_rt,   rt_min_full), rt_max_full))

    # Buttons
    c1, c2, _ = st.columns([1, 1, 2])
    do_apply = c1.button(f"Apply to {clip_target_name}", key=f"apply_clip_{clip_target_name}")
    do_reset = c2.button("Reset RT to full range", key=f"reset_clip_{clip_target_name}")

    if do_reset:
        st.session_state[bounds_key] = (rt_min_full, rt_max_full)
        active_df = dp.filter_rt_range(full_df, rt_min_full, rt_max_full, axis_column=axis_col)
        if clip_target_name == "df_grid":
            df_grid = active_df
        else:
            combined = active_df
        st.success(f"Reset {clip_target_name} RT range to [{rt_min_full:.3f}, {rt_max_full:.3f}]")

    if do_apply:
        lo, hi = (start_rt, end_rt) if start_rt <= end_rt else (end_rt, start_rt)
        filtered = dp.filter_rt_range(full_df, lo, hi, axis_column=axis_col)
        if filtered is None or filtered.empty:
            st.warning("RT filter produced an empty DataFrame. No changes applied.")
        else:
            st.session_state[bounds_key] = (lo, hi)
            if clip_target_name == "df_grid":
                df_grid = filtered
            else:
                combined = filtered
            st.success(f"Clipped {clip_target_name} to RT ∈ [{lo:.3f}, {hi:.3f}] — {len(filtered)} points.")

    # Preview current (post-clip) shape
    cur_lo, cur_hi = st.session_state[bounds_key]
    st.caption(f"Current {clip_target_name} RT window: [{cur_lo:.3f}, {cur_hi:.3f}]")
    with st.expander(f"Preview {clip_target_name} (head)"):
        preview_df = df_grid if clip_target_name == "df_grid" else combined
        show_df(preview_df.head(10))


# ---------------- RT REFERENCING (anchor & shift samples) ----------------
st.markdown("---")
st.subheader("RT Referencing (anchor samples to a common RT)")

# Choose input matrix: prefer df_grid (resampled) if available, else combined
ref_in_name = "df_grid" if ('df_grid' in locals() and df_grid is not None) else "combined"
ref_in_df = df_grid if ref_in_name == "df_grid" else combined

if ref_in_df is None:
    st.caption("Load/process data to enable RT referencing.")
    st.session_state.pop("referenced_df", None)
else:
    # Defaults from current data
    axis_col = ref_in_df.columns[0]  # "RT(min)"
    lo_axis = float(ref_in_df[axis_col].min())
    hi_axis = float(ref_in_df[axis_col].max())
    mid = (lo_axis + hi_axis) / 2.0

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        thresh = st.number_input("Peak threshold (fraction of max)", min_value=0.0, max_value=1.0,
                                 value=0.10, step=0.01, format="%.2f")
    with c2:
        target_position = st.number_input("Target RT (min)", value=mid, step=0.01, format="%.3f")
    with c3:
        offsetppm = st.text_input("Anchor RT (optional)", value="")  # leave blank for auto-peak

    # Search window
    d1, d2 = st.columns(2)
    with d1:
        x_lo = st.number_input("Search RT min", value=max(lo_axis, target_position - 0.5),
                               step=0.01, format="%.3f")
    with d2:
        x_hi = st.number_input("Search RT max", value=min(hi_axis, target_position + 0.5),
                               step=0.01, format="%.3f")

    # Buttons
    b1, b2 = st.columns([1, 1])
    do_reference = b1.button(f"Reference {ref_in_name}")
    do_clear = b2.button("Clear referenced copy")

    if do_clear:
        st.session_state.pop("referenced_df", None)
        st.info("Cleared referenced data.")

    if do_reference:
        off = None
        if offsetppm.strip():
            try:
                off = float(offsetppm)
            except ValueError:
                st.warning("Anchor RT must be numeric (e.g., 11.15). Falling back to auto-peak.")
                off = None

        try:
            referenced_df, offsets = dp.ref_spectra_to_df(
                ref_in_df,
                thresh=float(thresh),
                target_position=float(target_position),
                offsetppm=off,
                xlim=(float(min(x_lo, x_hi)), float(max(x_lo, x_hi)))
            )
            st.session_state["referenced_df"] = referenced_df
            st.success(f"Referenced {ref_in_name} → anchor at RT={target_position:.3f} min")
            with st.expander("Offsets used per sample (RT shift anchors)"):
                show_df(pd.DataFrame.from_dict(offsets, orient="index", columns=["offset_rt(min)"]))
        except Exception as e:
            st.error(f"Referencing failed: {e}")
            st.session_state.pop("referenced_df", None)

    # Always fetch from session for downstream use
    referenced_df = st.session_state.get("referenced_df", None)

    if referenced_df is not None:
        st.caption(f"Preview: referenced {ref_in_name}")
        show_df(referenced_df.head(10))

        # Plot: helper if available; else Plotly overlay
        try:
            dp.create_chromatogram_plot(
                referenced_df,
                x_axis_col='RT(min)',
                start_column=1,
                end_column=len(referenced_df.columns) - 1,  # last sample column index
                title='Referenced Chromatogram Overlapping',
                output_file='Referenced_chromatogram_overlapping.html',
                show_fig=False
            )
            st.success("Saved: Referenced_chromatogram_overlapping.html")
        except Exception:
            fig = go.Figure()
            for col in referenced_df.columns[1:]:
                fig.add_trace(go.Scatter(x=referenced_df['RT(min)'], y=referenced_df[col],
                                         name=col, mode='lines'))
            fig.update_layout(title="Referenced Chromatogram Overlapping",
                              xaxis_title="RT(min)", yaxis_title="Intensity")
            st.plotly_chart(fig, use_container_width=True)

        # Optional: promote referenced to working matrix
        use_as_working = st.checkbox("Use referenced data as new working matrix", value=False)
        if use_as_working:
            if ref_in_name == "df_grid":
                df_grid = referenced_df
            else:
                combined = referenced_df
            st.success(f"Replaced {ref_in_name} with referenced data.")

# ------------------ Alignment (Icoshift / PAFFT / RAFFT) ------------------
st.markdown("---")
df_aligned = None

# Prefer referenced data; else df_grid; else combined
align_source = st.session_state.get("referenced_df", None)
if align_source is None:
    align_source = df_grid if ('df_grid' in locals() and df_grid is not None) else combined

if align_source is not None:
    st.subheader("Alignment")
    st.markdown("**Icoshift:** Interval correlation optimized shifting (commonly used for chromatogram/NMR alignment).")
    st.markdown("**PAFFT / RAFFT:** FFT-based alignment approaches (phase or recursive). These correct RT drifts between chromatograms.")
    sample_names = list(align_source.columns[1:])
    method, params = alignment_controls(align_source, sample_names=sample_names)
    df_aligned = align_df(align_source, method, **params)
else:
    st.info("Upload and preprocess data (and/or reference RT) to enable alignment.")



# ---------- NORMALIZATION & SCALING ----------
st.markdown("---")
st.subheader("Normalization & Scaling")

# Choose the matrix we will operate on:
# Prefer the resampled+preprocessed grid; fall back to the raw combined matrix
working_df = df_grid if df_grid is not None else combined
if working_df is None:
    st.info("Load and preprocess data to enable normalization/scaling.")
else:
    # Columns to exclude from math ops
    exclude_default = ["RT(min)"]
    exclude_cols = st.multiselect(
        "Columns to EXCLUDE from transforms",
        options=list(working_df.columns),
        default=[c for c in exclude_default if c in working_df.columns],
        help="These columns are passed through untouched."
    )

    # ---------- OPTIONAL TRANSFORMS (stabilize variance) ----------
    st.markdown("**Transforms (optional, before normalization)**")
    tform = st.radio(
        "Variance-stabilizing transform",
        options=["none", "log10(+c)", "sqrt", "cbrt"],
        help="Applied before normalization. log10 uses a small constant to avoid log(0).",
        horizontal=True,
    )
    log_const = None
    if tform == "log10(+c)":
        log_const = st.number_input("Constant c for log10(x+c)", min_value=0.0, value=1.0, step=0.1)

    # ---------- NORMALIZATION ----------
    norm_choice = st.selectbox(
        "Normalization method",
        options=[
            "none",
            "z-score",
            "median",
            "std-dev",
            "min-max (0–1)",
            "quantile",
            "PQN (Probabilistic Quotient)",
            "normalize by control"
        ],
        index=0,
        help="Choose how to put samples on a comparable scale."
    )

    # Per-method parameters
    norm_params = {}
    if norm_choice == "normalize by control":
        candidates = [c for c in working_df.columns if c not in exclude_cols and c != "RT(min)"]
        if not candidates:
            st.warning("No eligible columns for control normalization.")
        norm_params["control_column"] = st.selectbox(
            "Control column (divides all others)",
            options=candidates if candidates else ["<none>"],
            help="All non-excluded columns are divided by this one."
        )
    elif norm_choice == "PQN (Probabilistic Quotient)":
        st.caption("PQN uses the sample-wise median spectrum as reference by default.")
    elif norm_choice == "median":
        norm_params["target_median"] = st.number_input(
            "Target median value",
            min_value=0.0, value=1.0, step=0.1,
            help="Each column is scaled to reach this median."
        )

    # ---------- SCALING ----------
    scale_choice = st.selectbox(
        "Scaling method",
        options=[
            "none",
            "standard (mean=0, sd=1)",
            "robust (median/IQR)",
            "pareto (mean-centered / sqrt(sd))",
            "range (mean-centered / range)",
            "min-max (custom range)"
        ],
        index=0,
        help="Scaling reshapes feature distributions for multivariate models."
    )

    scale_params = {}
    if scale_choice == "min-max (custom range)":
        col1, col2 = st.columns(2)
        with col1: scale_params["new_min"] = st.number_input("New min", value=0.0, step=0.1)
        with col2: scale_params["new_max"] = st.number_input("New max", value=1.0, step=0.1)

    def apply_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()

        # 1) TRANSFORM
        if tform == "log10(+c)":
            df_out = dp.log_transform(df_out, constant=log_const)
        elif tform == "sqrt":
            df_out = dp.sqrt_transform(df_out)
        elif tform == "cbrt":
            df_out = dp.cbrt_transform(df_out)
        # "none" → do nothing

        # 2) NORMALIZATION
        if norm_choice == "z-score":
            df_out = dp.z_score_normalize(df_out, exclude_columns=exclude_cols)
        elif norm_choice == "median":
            df_out = dp.median_normalize(df_out, target_median=norm_params.get("target_median", 1.0),
                                         exclude_columns=exclude_cols)
        elif norm_choice == "std-dev":
            df_out = dp.std_dev_normalize(df_out, exclude_columns=exclude_cols)
        elif norm_choice == "min-max (0–1)":
            df_out = dp.min_max_normalize(df_out)
        elif norm_choice == "quantile":
            df_out = dp.quantile_normalize(df_out, exclude_columns=exclude_cols)
        elif norm_choice == "PQN (Probabilistic Quotient)":
            df_out = dp.pqn_normalize(df_out, reference=None, exclude_columns=exclude_cols)
        elif norm_choice == "normalize by control":
            ctrl = norm_params.get("control_column")
            if ctrl and ctrl in df_out.columns:
                df_out = dp.normalize_by_control(df_out, control_column=ctrl, exclude_columns=exclude_cols)
            else:
                st.warning("Valid control column not selected. Skipping control normalization.")

        # 3) SCALING
        if scale_choice == "standard (mean=0, sd=1)":
            df_out = dp.standard_scale(df_out, exclude_columns=exclude_cols)
        elif scale_choice == "robust (median/IQR)":
            df_out = dp.robust_scale(df_out, exclude_columns=exclude_cols)
        elif scale_choice == "pareto (mean-centered / sqrt(sd))":
            df_out = dp.pareto_scale(df_out, exclude_columns=exclude_cols)
        elif scale_choice == "range (mean-centered / range)":
            df_out = dp.range_scale(df_out, exclude_columns=exclude_cols)
        elif scale_choice == "min-max (custom range)":
            df_out = dp.min_max_scale(df_out,
                                      new_min=scale_params.get("new_min", 0.0),
                                      new_max=scale_params.get("new_max", 1.0),
                                      exclude_columns=exclude_cols)
        # "none" → do nothing

        return df_out

    processed_df = apply_pipeline(working_df)
    st.success("Applied transforms.")
    show_df(processed_df.head(3))


# ------------------ STOCSY (Structured correlation) ------------------
def stocsy_linear(target_rt: float, X: pd.DataFrame, rt_values: pd.Series):
    """
    Minimal STOCSY (linear): correlate each row in X with the target row chosen by RT.
    X: rows=RT points, columns=samples (intensities)
    rt_values: Series aligned with X rows (RT(min))
    Returns corr (np.array), covar (np.array)
    """
    idx = (rt_values - float(target_rt)).abs().idxmin()
    target = X.loc[idx].astype(float).values
    A = target.astype(float)
    Xv = X.values.astype(float)
    corr = np.zeros(len(X), dtype=float)
    covar = np.zeros(len(X), dtype=float)
    for i in range(len(X)):
        y = Xv[i, :]
        m = np.isfinite(A) & np.isfinite(y)
        if m.sum() < 2:
            continue
        a = A[m]; b = y[m]
        cov = np.cov(a, b)[0, 1] if m.sum() > 1 else 0.0
        denom = a.std(ddof=1) * b.std(ddof=1)
        r = cov / denom if denom > 0 else 0.0
        corr[i] = float(r)
        covar[i] = float(cov)
    return corr, covar

if df_aligned is not None:
    st.markdown("---")
    st.subheader("STOCSY")

    # Controls
    cols = st.columns([2, 1, 1])
    with cols[0]:
        target_rt = st.number_input("Target RT (min)", value=11.25, step=0.05, format="%.2f")
    with cols[1]:
        mode = st.selectbox(
            "Model",
            ["linear", "exponential", "sinusoidal", "sigmoid", "gaussian", "fft", "polynomial", "piecewise", "skewed_gauss"],
            index=0
        )
    with cols[2]:
        run_btn = st.button("Run STOCSY", type="primary")

    # Optional: BioActivity fusion as the driver
    use_bio_driver = st.checkbox("Use BioActivity fusion and drive STOCSY with BioAct", value=False)
    MergeDF = None
    new_axis = None
    if use_bio_driver:
        if bio_df is None:
            st.error("BioActivity CSV not loaded. Upload it in the sidebar.")
        elif meta_df is None:
            st.error("Metadata not configured. Load metadata and select mapping columns above.")
        else:
            # Ensure expected mapping columns exist
            if 'col_sample' in locals() and 'col_hplc' in locals() and col_sample and col_hplc:
                try:
                    normalized_df = df_aligned.copy()
                    LC = normalized_df.drop(columns="RT(min)")
                    RT = normalized_df["RT(min)"]

                    ordered_samples = meta_df[col_sample].astype(str).tolist()
                    ordered_hplc = meta_df[col_hplc].astype(str).str.replace(".txt", "", regex=False).tolist()
                    if 'col_bio' in locals() and col_bio:
                        ordered_bio = meta_df[col_bio].astype(str).str.replace(".txt", "", regex=False).tolist()
                    else:
                        ordered_bio = ordered_hplc  # fall back

                    # LC columns already renamed to sample IDs when mapping applied
                    present_samples = [s for s in ordered_samples if s in LC.columns.astype(str).tolist()]

                    # Prepare BioAct data
                    bio_df_tmp = bio_df.copy()
                    def get_bio_cols(df):
                        return df.columns.astype(str).str.replace(".txt", "", regex=False).tolist()
                    bio_cols = get_bio_cols(bio_df_tmp)
                    if not all(b in bio_cols for b in ordered_bio):
                        if bio_df_tmp.shape[1] > 1:
                            bio_df_tmp = bio_df_tmp.iloc[:, 1:]
                            bio_cols = get_bio_cols(bio_df_tmp)

                    samples_ok = []
                    bio_cols_needed = []
                    for s, bb in zip(ordered_samples, ordered_bio):
                        if s in present_samples and (bb in bio_cols):
                            samples_ok.append(s)
                            bio_cols_needed.append(bb)

                    if len(samples_ok) == 0:
                        st.error("No overlapping samples between LC and BioAct using current metadata mapping.")
                    else:
                        LC_ord = LC[samples_ok].copy()

                        # Choose a numeric row from bio_df_tmp across required cols
                        picked = None
                        for r in range(bio_df_tmp.shape[0]):
                            vals = pd.to_numeric(bio_df_tmp[bio_cols_needed].iloc[r], errors="coerce")
                            if vals.notna().mean() > 0.8:
                                picked = vals.values.astype(float)
                                break
                        if picked is None:
                            picked = pd.to_numeric(bio_df_tmp[bio_cols_needed].iloc[0], errors="coerce").values.astype(float)

                        BioActdata = pd.DataFrame([picked], columns=bio_cols_needed)
                        BioActdata.rename(columns={i: j for i, j in zip(bio_cols_needed, samples_ok)}, inplace=True)
                        BioActdata = BioActdata[samples_ok]

                        MergeDF = pd.concat([LC_ord, BioActdata], ignore_index=True)

                        gap = float(RT.values[-1] - RT.values[-2]) if len(RT) >= 2 else 0.01
                        new_point = float(RT.values[-1]) + (gap if gap > 0 else 0.01)
                        new_axis = pd.concat([RT, pd.Series([new_point])], ignore_index=True)

                        st.success(f"Merged LC ({LC_ord.shape[0]}) + BioAct (1) for {len(samples_ok)} sample(s).")
                except Exception as e:
                    st.error(f"BioActivity fusion failed: {e}")
            else:
                st.error("Select valid 'Sample ID' and 'HPLC file stems' columns in metadata.")

    if run_btn:
        # Choose data & axis
        if use_bio_driver and (MergeDF is not None) and (new_axis is not None):
            Xmat = MergeDF
            rt_vals = new_axis
            target_for_run = float(new_axis.values[-1])
        else:
            Xmat = df_aligned.drop(columns="RT(min)")
            rt_vals = df_aligned["RT(min)"]
            target_for_run = float(target_rt)

        corr = covar = None
        # Prefer user's dp implementation if available
        if dp is not None and hasattr(dp, "STOCSY_LC_mode"):
            try:
                corr, covar = dp.STOCSY_LC_mode(target_for_run, Xmat, rt_vals, mode=mode)
            except Exception as e:
                st.warning(f"dp.STOCSY_LC_mode failed ({e}). Falling back to linear implementation.")

        if corr is None or covar is None:
            corr, covar = stocsy_linear(target_for_run, Xmat, rt_vals)

        res = pd.DataFrame({"RT(min)": rt_vals.values, "Correlation": corr, "Covariance": covar})
        topn = st.slider("Show top-N by |Correlation|", min_value=5, max_value=200, value=5, step=5)
        top = res.reindex(res["Correlation"].abs().sort_values(ascending=False).index).head(topn)
        st.markdown("**Top associations (by |Correlation|):**")
        show_df(top)

        figc = px.scatter(
            res, x="RT(min)", y="Covariance",
            color="Correlation", color_continuous_scale="Jet",
            render_mode="webgl",
            title=f"STOCSY from {target_for_run:.2f} min — Covariance colored by Correlation"
        )
        figc.update_traces(marker=dict(size=5))
        figc.add_trace(go.Scatter(x=res["RT(min)"], y=res["Covariance"], mode="lines", line=dict(width=1), name="Covariance"))
        st.plotly_chart(figc, use_container_width=True)

        figr = px.line(res, x="RT(min)", y="Correlation", title="Correlation vs RT(min)")
        st.plotly_chart(figr, use_container_width=True)

        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download STOCSY table (CSV)",
            data=csv_bytes,
            file_name=f"stocsy_{target_for_run:.2f}min_{mode}.csv",
            mime="text/csv"
        )

# ------------------ Visualizations ------------------
st.markdown("---")
if df_aligned is not None:
    st.subheader("Visualizations")
    plot_df = df_aligned.melt(id_vars="RT(min)", var_name="Sample", value_name="Intensity").dropna(subset=["Intensity"])

    t1, t2, t3 = st.tabs(["Overlay", "Stacked", "Heatmap"])
    with t1:
        fig = px.line(plot_df, x="RT(min)", y="Intensity", color="Sample", title="Overlay chromatograms")
        fig.update_layout(xaxis_title="RT (min)", yaxis_title="Intensity", legend_title="Sample")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        samples_sorted = sorted([c for c in df_aligned.columns if c != "RT(min)"])
        step = st.number_input("Stack offset", value=2.0, step=0.5)
        offset_map = {s: i * step for i, s in enumerate(samples_sorted)}
        plot_df["Intensity_offset"] = plot_df.apply(lambda r: r["Intensity"] + offset_map[r["Sample"]], axis=1)
        fig2 = px.line(plot_df, x="RT(min)", y="Intensity_offset", color="Sample", title="Stacked chromatograms")
        fig2.update_layout(xaxis_title="RT (min)", yaxis_title=f"Intensity + offset (step={step})")
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        mat = df_aligned[[c for c in df_aligned.columns if c != "RT(min)"]].T.values
        fig3 = go.Figure(data=go.Heatmap(
            z=mat,
            x=df_aligned["RT(min)"].values,
            y=[c for c in df_aligned.columns if c != "RT(min)"],
            coloraxis="coloraxis"
        ))
        fig3.update_layout(title="Intensity heatmap", xaxis_title="RT (min)", yaxis_title="Sample", coloraxis_colorscale="Viridis")
        st.plotly_chart(fig3, use_container_width=True)

def get_working_matrix():
    # strict precedence: the furthest processed wins
    for key in ["normalized_scaled_df", "aligned_df", "referenced_df", "preprocessed_df", "clipped_df", "combined"]:
        df = st.session_state.get(key, None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            return df, key
    return None, None


# ------------------ Modeling (PCA / PLS) ------------------
# prefer aligned, else preprocessed grid, else raw combined
model_df, source_key = get_working_matrix()
if model_df is None:
    st.subheader("Modeling")
    st.info("No matrix available for modeling yet. Load data, preprocess, and/or align first.")
else:
    st.caption(f"Using data source: **{source_key}**")
    samples = [c for c in model_df.columns if c != "RT(min)"]
    X, feats = matrix_to_XY(model_df, samples)
    # ... PCA / PLS sections unchanged (but use model_df for RT axis) ...

if model_df is None:
    st.subheader("Modeling")
    st.info("No matrix available for modeling yet. Load data, preprocess, and/or align first.")
else:
    from typing import List

    def _norm_name(s):
        return str(s).replace(".txt", "").strip()

    def _suggest_key_col(meta_df: pd.DataFrame, sample_cols: List[str]) -> str | None:
        norm_samples = {_norm_name(s) for s in sample_cols}
        best_col, best_hits = None, -1
        for col in meta_df.columns:
            col_norm_vals = {_norm_name(v) for v in meta_df[col].astype(str).tolist()}
            hits = len(norm_samples & col_norm_vals)
            if hits > best_hits:
                best_col, best_hits = col, hits
        return best_col

    st.subheader("Modeling")
    if not SKLEARN_AVAILABLE:
        st.warning(
            "scikit-learn is not installed. Install it to enable PCA/PLS sections.\n"
            "Conda: conda install -c conda-forge scikit-learn\n"
            "Pip:   pip install scikit-learn"
        )
    else:
        # Use model_df consistently
        samples = [c for c in model_df.columns if c != "RT(min)"]
        X, feats = matrix_to_XY(model_df, samples)

        # pick metadata column that matches sample names
        sample_name_key_col = None
        if meta_df is not None and not meta_df.empty:
            guess = _suggest_key_col(meta_df, samples)
            sample_name_key_col = st.selectbox(
                "Metadata column that matches chromatogram sample names",
                options=meta_df.columns.tolist(),
                index=(meta_df.columns.tolist().index(guess) if guess in meta_df.columns else 0)
            )
            norm_samples = {_norm_name(s) for s in samples}
            norm_meta_vals = {_norm_name(v) for v in meta_df[sample_name_key_col].astype(str).tolist()}
            overlap = len(norm_samples & norm_meta_vals)
            st.caption(f"**Name match check:** {overlap}/{len(samples)} chromatogram columns found in metadata '{sample_name_key_col}'.")

        # labels (may be None)
        labels = None
        y_col = "<none>"
        if meta_df is not None and not meta_df.empty:
            y_col = st.selectbox("Response/label column", options=["<none>"] + meta_df.columns.tolist(), index=0)
            if y_col != "<none>" and sample_name_key_col:
                meta_key_norm = meta_df[sample_name_key_col].astype(str).map(_norm_name)
                sample_to_label = dict(zip(meta_key_norm, meta_df[y_col]))
                labels = [sample_to_label.get(_norm_name(s), None) for s in samples]
                if all(l is None for l in labels):
                    st.warning(
                        "No labels matched your sample names. "
                        f"Check that '{sample_name_key_col}' contains the chromatogram names "
                        "(e.g., ASCII/HPLC stems) or pick a different metadata column."
                    )
                    labels = None

		# ---------------- PCA ----------------
            with st.expander("PCA", expanded=True):
                n_comp = st.slider("Components", min_value=2, max_value=10, value=2, step=1)
                scale_pca = st.checkbox("Standardize features before PCA", value=False)

                Xp = X.copy()
                if scale_pca:
                    Xp = StandardScaler(with_mean=True, with_std=True).fit_transform(Xp)

                pca = PCA(n_components=n_comp, random_state=0)
                scores = pca.fit_transform(Xp)
                expvar = pca.explained_variance_ratio_ * 100

                # Scores plot (PC1 vs PC2)
                df_scores = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Sample": samples})

                if labels is not None:
                    df_scores["Label"] = pd.Series(labels, dtype="object").fillna("NA").astype(str).values
                    figp = px.scatter(
                        df_scores, x="PC1", y="PC2", color="Label", hover_name="Sample",
                        title="PCA Scores"
                    )
                else:
                    figp = px.scatter(
                        df_scores, x="PC1", y="PC2", hover_name="Sample",
                        title="PCA Scores"
                    )

                # ⬇️ Add % contribution on the axes
                figp.update_layout(
                    xaxis_title=f"PC1 ({expvar[0]:.1f}%)",
                    yaxis_title=f"PC2 ({expvar[1]:.1f}%)"
                )

                st.plotly_chart(figp, use_container_width=True)

                # -------- NEW: Loadings plot (select any PC) --------
                st.markdown("**PCA X-Loadings**")
                max_pc = int(pca.n_components_)
                pc_idx = st.number_input("Component to show", min_value=1, max_value=max_pc, value=1, step=1)
                # pca.components_: shape (n_components, n_features); features follow RT order
                load_vec = pca.components_[pc_idx - 1, :]

                # Optional helpers
                show_abs = st.checkbox("Show absolute loadings", value=False)
                smooth_pts = st.number_input("Smooth (moving average, pts)", min_value=1, value=1, step=1)

                y_load = np.abs(load_vec) if show_abs else load_vec
                if smooth_pts > 1:
                    y_load = pd.Series(y_load).rolling(window=int(smooth_pts), min_periods=1, center=True).mean().values

                # RT axis from the working matrix used for PCA
                rt_axis = np.asarray(model_df["RT(min)"].values, dtype=float)

                df_load = pd.DataFrame({
                    "RT(min)": rt_axis,
                    "Loading": y_load
                })

                fig_load = px.line(
                    df_load, x="RT(min)", y="Loading",
                    title=f"PCA Loading — PC{pc_idx} ({'abs ' if show_abs else ''}loadings)"
                )
                fig_load.update_layout(xaxis_title="RT (min)", yaxis_title="Loading weight")
                st.plotly_chart(fig_load, use_container_width=True)




        # ---------------- PLS (PLS-DA or PLS Regression) ----------------
        with st.expander("PLS (PLS-DA / Regression)", expanded=False):
            if labels is None:
                st.info("Select a label/response column in Metadata to enable PLS.")
            else:
                y_raw = np.array(labels, dtype=object)

                # components limited by features AND samples-1
                max_lv = int(max(1, min(15, X.shape[1], X.shape[0] - 1)))
                n_comp_pls = st.slider("PLS components", min_value=2, max_value=max_lv, value=min(5, max_lv), step=1)
                scale_X = st.checkbox("Standardize X", value=True)

                y_num_try = pd.to_numeric(y_raw, errors="coerce")
                is_numeric = np.isfinite(y_num_try).all()

                X_pls = X.copy()
                if scale_X:
                    X_pls = StandardScaler(with_mean=True, with_std=True).fit_transform(X_pls)

                # ---------- helpers ----------
                def _r2x_from_scores_loadings(X_, T_, P_):
                    X_hat = T_ @ P_.T
                    sse = np.sum((X_ - X_hat) ** 2)
                    sst = np.sum((X_ - np.mean(X_, axis=0)) ** 2)
                    return float(1.0 - (sse / sst)) if sst > 0 else np.nan

                def _per_component_metrics_regression(X_, y_, max_a):
                    r2x_list, r2y_list, q2_list = [], [], []
                    n_splits = max(2, min(5, len(y_)))
                    for a in range(1, max_a + 1):
                        pls = PLSRegression(n_components=a).fit(X_, y_)
                        T, P = pls.x_scores_[:, :a], pls.x_loadings_[:, :a]
                        y_hat = pls.predict(X_)
                        r2y = r2_score(y_, y_hat)
                        r2x = _r2x_from_scores_loadings(X_, T, P)
                        # fresh CV each a
                        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                        press = sstot = 0.0
                        for tr, te in cv.split(X_):
                            m = PLSRegression(n_components=a).fit(X_[tr], y_[tr])
                            y_pred_te = m.predict(X_[te])
                            press += np.sum((y_[te] - y_pred_te) ** 2)
                            y_mean_tr = np.mean(y_[tr], axis=0, keepdims=True)
                            sstot += np.sum((y_[te] - y_mean_tr) ** 2)
                        q2 = float(1.0 - (press / sstot)) if sstot > 0 else np.nan
                        r2x_list += [r2x]; r2y_list += [r2y]; q2_list += [q2]
                    return r2x_list, r2y_list, q2_list

                def _per_component_metrics_plsda(X_, Y_, y_enc_, max_a):
                    r2x_list, r2y_list, q2_list = [], [], []
                    counts = np.bincount(y_enc_)
                    stratified = (len(counts) > 1 and counts.min() >= 2)
                    n_splits = max(2, min(5, int(counts.min() if stratified else len(y_enc_))))

                    for a in range(1, max_a + 1):
                        pls = PLSRegression(n_components=a).fit(X_, Y_)
                        T, P = pls.x_scores_[:, :a], pls.x_loadings_[:, :a]
                        Y_hat = pls.predict(X_)
                        if Y_.shape[1] == 1:
                            r2y = r2_score(Y_, Y_hat)
                        else:
                            r2y = float(np.mean([r2_score(Y_[:, j], Y_hat[:, j]) for j in range(Y_.shape[1])]))
                        r2x = _r2x_from_scores_loadings(X_, T, P)

                        # fresh splitter each a
                        if stratified:
                            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                            splits = cv.split(X_, y_enc_)
                        else:
                            cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
                            splits = cv.split(X_)

                        press = sstot = 0.0
                        for tr, te in splits:
                            m = PLSRegression(n_components=a).fit(X_[tr], Y_[tr])
                            Y_pred_te = m.predict(X_[te])
                            press += np.sum((Y_[te] - Y_pred_te) ** 2)
                            Y_mean_tr = np.mean(Y_[tr], axis=0, keepdims=True)
                            sstot += np.sum((Y_[te] - Y_mean_tr) ** 2)

                        q2 = float(1.0 - (press / sstot)) if sstot > 0 else np.nan
                        r2x_list += [r2x]; r2y_list += [r2y]; q2_list += [q2]
                    return r2x_list, r2y_list, q2_list

                def _plot_perf_bars(r2x_list, r2y_list, q2_list, title):
                    df_perf = pd.DataFrame({
                        "Component": np.arange(1, len(r2x_list) + 1),
                        "R²X": r2x_list, "R²Y": r2y_list, "Q²": q2_list,
                    })
                    df_long = df_perf.melt(id_vars="Component", value_vars=["R²X", "R²Y", "Q²"],
                                           var_name="Metric", value_name="Value")
                    fig_perf = px.bar(df_long, x="Component", y="Value", color="Metric",
                                      barmode="group", range_y=[0, 1], title=title, text="Value")
                    fig_perf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    fig_perf.update_layout(yaxis_title="Value (0–1)", xaxis_title="LV (component)")
                    st.plotly_chart(fig_perf, use_container_width=True)

                # ---------- Run ----------
                if is_numeric:
                    # PLS Regression
                    y = np.asarray(y_num_try, dtype=float).reshape(-1, 1)
                    r2x_list, r2y_list, q2_list = _per_component_metrics_regression(X_pls, y, n_comp_pls)

                    pls = PLSRegression(n_components=n_comp_pls).fit(X_pls, y)
                    T, P = pls.x_scores_, pls.x_loadings_

                    df_scores = pd.DataFrame({
                        "t1": T[:, 0],
                        "t2": T[:, 1] if T.shape[1] > 1 else np.zeros(T.shape[0]),
                        "Sample": [str(s) for s in samples],
                        "Label": pd.Series(y_raw, dtype="object").fillna("NA").astype(str).values,
                    })
                    fig_scores = px.scatter(df_scores, x="t1", y="t2", color="Label", hover_name="Sample",
                                            title="PLS Regression Scores")
                    fig_scores.update_layout(xaxis_title="t[1]", yaxis_title="t[2]")
                    st.plotly_chart(fig_scores, use_container_width=True)

                    rt_axis = np.array([float(r) for r in model_df["RT(min)"].values])
                    df_load = pd.DataFrame({"RT(min)": rt_axis, "p1": P[:, 0],
                                            "p2": P[:, 1] if P.shape[1] > 1 else np.zeros(P.shape[0])})
                    fig_load = go.Figure()
                    fig_load.add_trace(go.Scatter(x=df_load["RT(min)"], y=df_load["p1"], mode="lines", name="p[1]"))
                    if P.shape[1] > 1:
                        fig_load.add_trace(go.Scatter(x=df_load["RT(min)"], y=df_load["p2"], mode="lines", name="p[2]"))
                    fig_load.update_layout(title="X-Loadings (p[1], p[2])", xaxis_title="RT (min)", yaxis_title="Loading")
                    st.plotly_chart(fig_load, use_container_width=True)

                    _plot_perf_bars(r2x_list, r2y_list, q2_list, "PLS Regression Performance per Component")

                else:
                    # PLS-DA
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y_raw.astype(str))
                    classes = le.classes_
                    n_classes = len(classes)

                    if n_classes == 2:
                        Y = y_enc.reshape(-1, 1).astype(float)
                    else:
                        Y = np.zeros((len(y_enc), n_classes), dtype=float)
                        Y[np.arange(len(y_enc)), y_enc] = 1.0

                    r2x_list, r2y_list, q2_list = _per_component_metrics_plsda(X_pls, Y, y_enc, n_comp_pls)

                    pls = PLSRegression(n_components=n_comp_pls).fit(X_pls, Y)
                    T, P = pls.x_scores_, pls.x_loadings_

                    df_scores = pd.DataFrame({
                        "t1": T[:, 0],
                        "t2": T[:, 1] if T.shape[1] > 1 else np.zeros(T.shape[0]),
                        "Sample": [str(s) for s in samples],
                        "Class": [classes[i] for i in y_enc],
                    })
                    fig_scores = px.scatter(df_scores, x="t1", y="t2", color="Class", hover_name="Sample",
                                            title="PLS-DA Scores")
                    fig_scores.update_layout(xaxis_title="t[1]", yaxis_title="t[2]")
                    st.plotly_chart(fig_scores, use_container_width=True)

                    rt_axis = np.array([float(r) for r in model_df["RT(min)"].values])
                    df_load = pd.DataFrame({"RT(min)": rt_axis, "p1": P[:, 0],
                                            "p2": P[:, 1] if P.shape[1] > 1 else np.zeros(P.shape[0])})
                    fig_load = go.Figure()
                    fig_load.add_trace(go.Scatter(x=df_load["RT(min)"], y=df_load["p1"], mode="lines", name="p[1]"))
                    if P.shape[1] > 1:
                        fig_load.add_trace(go.Scatter(x=df_load["RT(min)"], y=df_load["p2"], mode="lines", name="p[2]"))
                    fig_load.update_layout(title="PLS-DA X-Loadings (p[1], p[2])", xaxis_title="RT (min)", yaxis_title="Loading")
                    st.plotly_chart(fig_load, use_container_width=True)

                    _plot_perf_bars(r2x_list, r2y_list, q2_list, "PLS-DA Performance per Component")

# ------------------ More ------------------


# ------------------ Export ------------------
if df_aligned is not None:
    st.subheader("Export")
    want_wide = st.checkbox("Include wide matrix (RT in rows, samples in columns)", value=True)
    want_long = st.checkbox("Include long format (RT, Sample, Intensity)", value=False)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("processed/aligned_matrix_wide.csv", df_aligned.to_csv(index=False))
        if want_long:
            long_df = df_aligned.melt(id_vars="RT(min)", var_name="Sample", value_name="Intensity")
            zf.writestr("processed/aligned_matrix_long.csv", long_df.to_csv(index=False))
        if meta_df is not None:
            zf.writestr("processed/metadata.csv", meta_df.to_csv(index=False))
        if bio_df is not None:
            zf.writestr("processed/bioactivity.csv", bio_df.to_csv(index=False))

    st.download_button(
        "⬇️ Download processed data (.zip)",
        data=zbuf.getvalue(),
        file_name="pyMETAflow_HPLC_processed.zip",
        mime="application/zip"
    )


# ------------------ Export (MetaboAnalyst) ---------------------------------------
st.markdown("---")
st.subheader("Export to MetaboAnalyst")

import io, re, time
from pathlib import Path

# 0) Pick which matrix to export (prefer fully processed)
try:
    export_df, export_source_key = get_working_matrix()   # <- your helper returning (df, key)
except Exception:
    # Fallback if you don't have get_working_matrix():
    # export_df = normalized_df or df_aligned or referenced_df or processed0 or combined
    export_df = None
    export_source_key = "unknown"

if not isinstance(export_df, pd.DataFrame) or export_df.empty:
    st.info("No matrix available to export yet. Load/preprocess data first.")
elif meta_df is None or meta_df.empty:
    st.warning("Load metadata (.csv) to include class labels for MetaboAnalyst.")
else:
    axis_col = "RT(min)"
    if axis_col not in export_df.columns:
        st.error(f"Axis column '{axis_col}' not found in export matrix.")
    else:
        sample_cols = [c for c in export_df.columns if c != axis_col]

        st.caption(f"Using data source: **{export_source_key}**  |  Samples detected: **{len(sample_cols)}**")

        # --- Sanitizer: alnum + underscore only (matches typical MetaboAnalyst expectations)
        def _san(x: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]", "", str(x))

        # 1) Guess the metadata column that matches the current sample names (after sanitization)
        samples_sanit = {_san(c) for c in sample_cols}
        guess_id, best_hits = None, -1
        for col in meta_df.columns:
            vals = {_san(v) for v in meta_df[col].astype(str).tolist()}
            hits = len(samples_sanit & vals)
            if hits > best_hits:
                best_hits, guess_id = hits, col

        # 2) Guess a class/phenotype column
        guess_class = None
        for key in ("class", "group", "condition", "phenotype"):
            cands = [c for c in meta_df.columns if key in c.lower()]
            if cands:
                guess_class = cands[0]
                break
        if guess_class is None and "ATTRIBUTE_classification" in meta_df.columns:
            guess_class = "ATTRIBUTE_classification"
        if guess_class is None:
            guess_class = meta_df.columns[-1]  # last column as fallback

        c1, c2 = st.columns(2)
        with c1:
            sample_id_col = st.selectbox(
                "Metadata column that matches sample columns",
                options=meta_df.columns.tolist(),
                index=(meta_df.columns.tolist().index(guess_id) if guess_id in meta_df.columns else 0),
                help="This column’s values should match your chromatogram column names (after sanitization)."
            )
        with c2:
            class_col = st.selectbox(
                "Metadata class / phenotype column",
                options=meta_df.columns.tolist(),
                index=(meta_df.columns.tolist().index(guess_class) if guess_class in meta_df.columns else 0)
            )

        # Overlap preview
        meta_ids_sanit = meta_df[sample_id_col].astype(str).map(_san)
        overlap = len(samples_sanit & set(meta_ids_sanit))
        st.caption(f"Matched **{overlap}/{len(sample_cols)}** samples after sanitization.")

        # Optional: show missing ones
        missing = sorted(list(samples_sanit - set(meta_ids_sanit)))
        if missing:
            with st.expander("Samples missing metadata (will be excluded)"):
                show_df(pd.DataFrame({"missing_sample_id_after_sanitize": missing}))

        # Output filename
        default_name = f"metaboanalyst_input_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        out_name = st.text_input("Output filename", value=default_name)

        # 3) Build & download
        build_btn = st.button("Build MetaboAnalyst CSV")
        if build_btn:
            try:
                csv_bytes = None

                # Prefer your dp.export_metaboanalyst if available
                if hasattr(dp, "export_metaboanalyst"):
                    out_path = Path(out_name).name  # simple local filename
                    _ = dp.export_metaboanalyst(
                        aligned_df=export_df,          # any feature×samples matrix with first col RT(min)
                        df_metadata=meta_df,
                        sample_id_col=sample_id_col,
                        class_col=class_col,
                        output_file=out_path
                    )
                    with open(out_path, "rb") as fh:
                        csv_bytes = fh.read()
                else:
                    # Fallback: in-memory writer matching MetaboAnalyst format
                    # Row 1: header (axis + samples)
                    # Row 2: class labels (blank in first cell)
                    # Remaining: data matrix
                    import csv
                    df_al = export_df.copy()

                    # Sanitize sample columns
                    orig_cols = list(df_al.columns)
                    new_cols = [orig_cols[0]] + [_san(c) for c in orig_cols[1:]]
                    df_al.columns = new_cols
                    sample_cols_san = new_cols[1:]

                    # Align metadata
                    meta2 = meta_df.copy()
                    meta2[sample_id_col] = meta2[sample_id_col].astype(str).map(_san)
                    meta2[class_col]     = meta2[class_col].astype(str).map(_san)
                    meta_idx = meta2.set_index(sample_id_col)
                    cls_series = meta_idx.reindex(sample_cols_san)[class_col]
                    valid = cls_series.dropna().index.tolist()

                    # Build reduced matrix with valid samples only
                    new_df = df_al[[axis_col] + valid].copy()

                    buf = io.StringIO()
                    w = csv.writer(buf)
                    # header row
                    w.writerow(new_df.columns.tolist())
                    # class row (first cell blank)
                    w.writerow([""] + cls_series.loc[valid].tolist())
                    # data rows
                    for i in range(len(new_df)):
                        w.writerow(new_df.iloc[i].values.tolist())
                    csv_bytes = buf.getvalue().encode("utf-8")

                if not csv_bytes:
                    raise RuntimeError("Failed to build CSV bytes.")

                st.success(f"MetaboAnalyst CSV built ({overlap} samples with class labels).")
                st.download_button(
                    "⬇️ Download MetaboAnalyst CSV",
                    data=csv_bytes,
                    file_name=out_name,
                    mime="text/csv"
                )

                # Tiny preview (first rows)
                try:
                    prev = pd.read_csv(io.BytesIO(csv_bytes), header=None, nrows=6)
                    st.caption("Preview (first rows):")
                    st.dataframe(prev, use_container_width=True)
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Export failed: {e}")

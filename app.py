
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="HPLC Gradient Verifier", layout="wide")

# --------------------------
# Helpers
# --------------------------
def show_df_html(df: pd.DataFrame, *, max_rows: int | None = None):
    """Render a DataFrame as plain HTML to avoid any pyarrow usage."""
    if df is None or df.empty:
        st.info("No data.")
        return
    if max_rows is not None:
        df = df.head(max_rows)
    html = df.to_html(index=False)
    st.markdown(html, unsafe_allow_html=True)

def parse_chrom_txt(uploaded_file) -> pd.DataFrame:
    data = []
    start_extraction = False
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    for line in content.splitlines():
        if line.strip().startswith("R.Time (min)"):
            start_extraction = True
            continue
        if start_extraction:
            cols = line.strip().split()
            if len(cols) == 2:
                cols = [c.replace(",", ".") for c in cols]
                data.append(cols)

    if not data:
        return None

    df = pd.DataFrame(data, columns=["RT(min)", uploaded_file.name])
    df["RT(min)"] = pd.to_numeric(df["RT(min)"], errors="coerce")
    df[uploaded_file.name] = pd.to_numeric(df[uploaded_file.name], errors="coerce")
    df = df.dropna()
    return df

def combine_chromatograms(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    combined = dfs[0].copy()
    for df in dfs[1:]:
        combined = pd.merge(combined, df, on="RT(min)", how="outer")
    combined = combined.sort_values("RT(min)").reset_index(drop=True)
    for col in combined.columns:
        if col != "RT(min)":
            combined[col] = combined[col].astype(float)
            combined[col] = combined[col].interpolate(method="linear")
    combined = combined.dropna(subset=["RT(min)"])
    return combined

def mask_region(df: pd.DataFrame, start_rt: float, end_rt: float) -> pd.DataFrame:
    masked = df.copy()
    if "RT(min)" not in masked.columns:
        return masked
    rows = masked["RT(min)"].between(min(start_rt, end_rt), max(start_rt, end_rt))
    sample_cols = [c for c in masked.columns if c != "RT(min)"]
    masked.loc[rows, sample_cols] = 0.0
    return masked

def plot_gradient_only(gradient_df: pd.DataFrame):
    times_list, perc_list = [], []
    for i, row in gradient_df.iterrows():
        if i == 0 or (i > 0 and float(gradient_df.loc[i-1, "end_time"]) != float(row["start_time"])):
            times_list.append(float(row["start_time"]))
            perc_list.append(float(row["start_B%"]))
        times_list.append(float(row["end_time"]))
        perc_list.append(float(row["end_B%"]))
    t = np.array(times_list)
    p = np.array(perc_list)
    if len(t) < 2:
        return go.Figure()
    time_range = np.arange(t.min(), t.max(), 0.1)
    p_interp = np.interp(time_range, t, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_range, y=p_interp, mode="lines", name="Solvent B (%)"))
    fig.update_layout(title="Gradient Profile (B%)", xaxis_title="Time (min)", yaxis_title="B (%)", template="plotly_white")
    return fig

def plot_overlay(chrom_df: pd.DataFrame, gradient_df: pd.DataFrame):
    times_list, perc_list = [], []
    for i, row in gradient_df.iterrows():
        if i == 0 or (i > 0 and float(gradient_df.loc[i-1, "end_time"]) != float(row["start_time"])):
            times_list.append(float(row["start_time"]))
            perc_list.append(float(row["start_B%"]))
        times_list.append(float(row["end_time"]))
        perc_list.append(float(row["end_B%"]))
    t = np.array(times_list)
    p = np.array(perc_list)
    if len(t) >= 2:
        time_range = np.arange(t.min(), t.max(), 0.1)
        p_interp = np.interp(time_range, t, p)
    else:
        time_range = np.array([])
        p_interp = np.array([])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in chrom_df.columns:
        if col == "RT(min)":
            continue
        fig.add_trace(
            go.Scatter(x=chrom_df["RT(min)"], y=chrom_df[col], mode="lines", name=col),
            secondary_y=False
        )
    if time_range.size > 0:
        fig.add_trace(
            go.Scatter(x=time_range, y=p_interp, mode="lines", name="Solvent B (%)", line=dict(dash="dot")),
            secondary_y=True
        )
    fig.update_layout(
        title="Chromatograms + Gradient (B%)",
        xaxis_title="RT (min)",
        legend_title="Series",
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_yaxes(title_text="Intensity (a.u.)", secondary_y=False)
    fig.update_yaxes(title_text="B (%)", secondary_y=True)
    return fig

def compute_volumes(gradient_df: pd.DataFrame, flow_mL_min: float) -> pd.DataFrame:
    rows = []
    total_A = 0.0
    total_B = 0.0
    for i, r in gradient_df.iterrows():
        start = float(r["start_time"])
        end = float(r["end_time"])
        b0 = float(r["start_B%"])
        b1 = float(r["end_B%"])
        duration = max(end - start, 0.0)
        seg_vol = flow_mL_min * duration
        avg_B = 0.5 * (b0 + b1) / 100.0
        vol_B = seg_vol * avg_B
        vol_A = seg_vol - vol_B
        total_A += vol_A
        total_B += vol_B
        rows.append({
            "segment": i + 1,
            "start_time": start,
            "end_time": end,
            "duration_min": duration,
            "avg_B_%": 100.0 * avg_B,
            "segment_total_mL": seg_vol,
            "segment_A_mL": vol_A,
            "segment_B_mL": vol_B,
        })
    df = pd.DataFrame(rows)
    totals = pd.DataFrame([{
        "segment": "TOTAL",
        "start_time": None,
        "end_time": None,
        "duration_min": df["duration_min"].sum() if not df.empty else 0.0,
        "avg_B_%": None,
        "segment_total_mL": df["segment_total_mL"].sum() if not df.empty else 0.0,
        "segment_A_mL": total_A,
        "segment_B_mL": total_B,
    }])
    return pd.concat([df, totals], ignore_index=True)

# --------------------------
# UI
# --------------------------
st.title("🔬 HPLC Gradient Verification")
st.markdown(
    "Upload chromatogram **.txt** files and a gradient table to compare chromatograms "
    "against the solvent program, mask unwanted RT regions, and compute solvent A/B volumes."
)

with st.sidebar:
    st.header("1) Upload chromatograms (.txt)")
    uploads = st.file_uploader("Select one or more files", type=["txt"], accept_multiple_files=True)
    st.caption("Files must include a line that starts with `R.Time (min)` followed by two numeric columns.")

    st.header("2) Retention time window")
    rt_min = st.number_input("Start RT (min)", value=0.0, step=0.5, format="%.2f")
    rt_max = st.number_input("End RT (min)", value=30.0, step=0.5, format="%.2f")

    st.header("3) Mask region (optional)")
    m_start = st.number_input("Mask start RT", value=0.0, step=0.5, format="%.2f")
    m_end = st.number_input("Mask end RT", value=0.0, step=0.5, format="%.2f")
    do_mask = st.checkbox("Apply masking to chromatograms", value=False)

    st.header("4) Gradient input")
    st.caption("Upload CSV with columns: start_time,end_time,start_B%,end_B% **or** edit below.")
    grad_upload = st.file_uploader("Gradient CSV", type=["csv"], accept_multiple_files=False, key="grad_csv")

    st.header("5) Flow rate")
    flow = st.number_input("Flow (mL/min)", value=1.0, step=0.1, format="%.2f")

default_gradient = pd.DataFrame({
    "start_time": [0, 5, 20, 25],
    "end_time":   [5, 20, 25, 30],
    "start_B%":   [5, 5, 95, 95],
    "end_B%":     [5, 95, 95, 5],
})

# ---- Gradient editor without pyarrow ----
if grad_upload is not None:
    gradient_df = pd.read_csv(grad_upload)
else:
    st.subheader("Gradient editor")
    example = default_gradient.to_csv(index=False)
    csv_text = st.text_area("Paste/edit gradient CSV here:", value=example, height=150)
    try:
        gradient_df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        gradient_df = default_gradient.copy()
        st.error("Could not parse the CSV text. Using default gradient.")

chrom_dfs = []
if uploads:
    for uf in uploads:
        try:
            df = parse_chrom_txt(uf)
            if df is not None:
                chrom_dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to parse {uf.name}: {e}")

combined = pd.DataFrame()
if chrom_dfs:
    combined = combine_chromatograms(chrom_dfs)
    combined = combined[(combined["RT(min)"] >= min(rt_min, rt_max)) & (combined["RT(min)"] <= max(rt_min, rt_max))]
    combined = combined.reset_index(drop=True)
    if do_mask and (m_start != m_end):
        combined = mask_region(combined, m_start, m_end)

st.subheader("Combined chromatograms")
if combined.empty:
    st.info("Upload chromatogram files to see the combined table.")
else:
    show_df_html(combined, max_rows=20)
    csv_bytes = combined.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download combined_data.csv", data=csv_bytes, file_name="combined_data.csv", mime="text/csv")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Gradient (B%)")
    fig_grad = plot_gradient_only(gradient_df)
    st.plotly_chart(fig_grad, use_container_width=True)
with col2:
    st.subheader("Overlay: Chromatograms + Gradient")
    if not combined.empty:
        fig_overlay = plot_overlay(combined, gradient_df)
        st.plotly_chart(fig_overlay, use_container_width=True)
    else:
        st.info("Upload chromatograms to see the overlay.")

st.subheader("Solvent volumes")
vol_df = compute_volumes(gradient_df, flow)
show_df_html(vol_df)

if not combined.empty:
    total_A = vol_df.loc[vol_df["segment"] == "TOTAL", "segment_A_mL"].values[0]
    total_B = vol_df.loc[vol_df["segment"] == "TOTAL", "segment_B_mL"].values[0]
    st.markdown(f"**Total volume A:** {total_A:.2f} mL &nbsp;&nbsp; **Total volume B:** {total_B:.2f} mL &nbsp;&nbsp; **Total (A+B):** {total_A + total_B:.2f} mL")

st.markdown("---")
st.caption("This app avoids PyArrow entirely for table rendering. For big tables and editing, install PyArrow to enable Streamlit's grid components.")

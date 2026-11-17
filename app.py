from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_mobile_price_4fitur.pkl"

FEATURE_COLUMNS = ["ram", "battery_power", "px_width", "px_height"]

FEATURE_SPECS = {
    "ram": {
        "label": "Kapasitas RAM (MB)",
        "min": 256,
        "max": 16384,
        "default": 2048,
        "step": 128,
        "help": "Kapasitas memori RAM ponsel dalam satuan Megabyte (256–16384 MB).",
    },
    "battery_power": {
        "label": "Kapasitas Baterai (mAh)",
        "min": 500,
        "max": 6000,
        "default": 1500,
        "step": 100,
        "help": "Daya tampung baterai ponsel dalam satuan mAh (500–6000 mAh).",
    },
    "px_width": {
        "label": "Lebar Resolusi Layar (piksel)",
        "min": 500,
        "max": 4000,
        "default": 1280,
        "step": 10,
        "help": "Jumlah piksel horizontal pada layar ponsel (500–4000 piksel).",
    },
    "px_height": {
        "label": "Tinggi Resolusi Layar (piksel)",
        "min": 0,
        "max": 4000,
        "default": 720,
        "step": 10,
        "help": "Jumlah piksel vertikal pada layar ponsel (0–4000 piksel).",
    },
}

CLASS_LABELS = {
    0: "Low Cost (Entry-Level)",
    1: "Medium Cost (Mid-Range)",
    2: "High Cost (Upper Mid-Range)",
    3: "Very High Cost (Flagship)",
}

CLASS_DESCRIPTIONS = {
    0: "Ponsel kelas pemula dengan spesifikasi dasar untuk kebutuhan harian ringan.",
    1: "Ponsel kelas menengah dengan performa seimbang untuk penggunaan sehari-hari.",
    2: "Ponsel kelas menengah atas dengan RAM besar dan layar beresolusi tinggi.",
    3: "Ponsel kelas unggulan (flagship) dengan spesifikasi tinggi di seluruh aspek.",
}

CLASS_STYLES = {
    0: {"accent": "#64748B", "tint": "rgba(100, 116, 139, 0.12)", "tag": "Kelas 0 • Entry-Level"},
    1: {"accent": "#0D9488", "tint": "rgba(13, 148, 136, 0.12)", "tag": "Kelas 1 • Mid-Range"},
    2: {"accent": "#2563EB", "tint": "rgba(37, 99, 235, 0.12)", "tag": "Kelas 2 • Upper Mid-Range"},
    3: {"accent": "#D97706", "tint": "rgba(217, 119, 6, 0.14)", "tag": "Kelas 3 • Flagship"},
}

PRESETS = {
    "entry": {
        "button_label": "Contoh HP Entry-Level",
        "values": {"ram": 512, "battery_power": 900, "px_width": 720, "px_height": 480},
    },
    "mid": {
        "button_label": "Contoh HP Mid-Range",
        "values": {"ram": 1536, "battery_power": 1400, "px_width": 1280, "px_height": 720},
    },
    "flagship": {
        "button_label": "Contoh HP Flagship",
        "values": {"ram": 8192, "battery_power": 5000, "px_width": 2400, "px_height": 1080},
    },
}

FONT_STACK = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def inject_theme_styles() -> None:
    """Load Plus Jakarta Sans font and clean editorial styles compatible with light/dark mode."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
    font-family: {FONT_STACK};
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp input, .stApp button,
.stApp [data-testid="stMarkdownContainer"] {{
    font-family: {FONT_STACK};
}}

/* Preserve Streamlit's Material icon font so built-in UI icons render properly */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {{
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}}

.block-container {{
    padding-top: 1.75rem;
    padding-bottom: 2.5rem;
    max-width: 1120px;
}}

.editorial-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 0 0 1.25rem 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.22);
    margin-bottom: 1.25rem;
}}

.editorial-kicker {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.35rem;
}}

.editorial-title {{
    margin: 0 0 0.4rem 0;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
}}

.editorial-subtitle {{
    margin: 0;
    font-size: 0.95rem;
    opacity: 0.8;
    line-height: 1.55;
    max-width: 660px;
}}

.section-heading {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0.2rem 0 0.25rem 0;
}}

.section-heading h3, .section-heading h4 {{
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.015em;
}}

.result-panel {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.85rem;
}}

.result-panel-inner {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}}

.tier-tag {{
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.45rem;
}}

.result-heading {{
    margin: 0 0 0.3rem 0;
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.3;
}}

.result-copy {{
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.8;
    line-height: 1.5;
}}

.blueprint-box {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 0.85rem 1rem;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 0.85rem;
}}

div.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def svg_header_schematic() -> str:
    """Return a clean minimalist hardware-classifier SVG schematic."""
    return f"""
<div class="editorial-header">
  <div style="flex: 1; min-width: 240px;">
    <div class="editorial-kicker">Proyek Akhir Pengenalan Pola</div>
    <h1 class="editorial-title">Klasifikasi Rentang Harga Ponsel</h1>
    <p class="editorial-subtitle">
      Prediksi segmen harga smartphone menggunakan <b>Random Forest Classifier</b>
      dari 4 parameter perangkat keras utama: <code>ram</code>, <code>battery_power</code>,
      <code>px_width</code>, dan <code>px_height</code>.
    </p>
  </div>
  <div style="flex-shrink: 0;">
    <svg width="164" height="92" viewBox="0 0 164 92" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Diagram Klasifikasi Ponsel">
      <rect x="10" y="10" width="44" height="72" rx="8" stroke="currentColor" stroke-opacity="0.75" stroke-width="1.8"/>
      <rect x="15" y="18" width="34" height="52" rx="3" stroke="currentColor" stroke-opacity="0.25" stroke-width="1.2"/>
      <line x1="26" y1="14" x2="38" y2="14" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.8" stroke-linecap="round"/>
      <rect x="20" y="48" width="5" height="16" rx="1.5" fill="#64748B"/>
      <rect x="27" y="40" width="5" height="24" rx="1.5" fill="#0D9488"/>
      <rect x="34" y="33" width="5" height="31" rx="1.5" fill="#2563EB"/>
      <rect x="41" y="26" width="5" height="38" rx="1.5" fill="#D97706"/>
      <path d="M58 46H78" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5" stroke-dasharray="3 3"/>
      <circle cx="88" cy="24" r="4" stroke="currentColor" stroke-opacity="0.7" stroke-width="1.6"/>
      <circle cx="88" cy="46" r="4" stroke="#2563EB" stroke-width="1.8"/>
      <circle cx="88" cy="68" r="4" stroke="currentColor" stroke-opacity="0.7" stroke-width="1.6"/>
      <path d="M92 24H108M92 46H108M92 68H108" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.4"/>
      <rect x="110" y="16" width="44" height="16" rx="4" stroke="currentColor" stroke-opacity="0.28" stroke-width="1.2"/>
      <text x="132" y="27" text-anchor="middle" fill="currentColor" opacity="0.82" font-size="8.5" font-weight="600" font-family="{FONT_STACK}">4 Fitur</text>
      <rect x="110" y="38" width="44" height="16" rx="4" stroke="#2563EB" stroke-opacity="0.65" stroke-width="1.2"/>
      <text x="132" y="49" text-anchor="middle" fill="#2563EB" font-size="8.5" font-weight="700" font-family="{FONT_STACK}">400 Tree</text>
      <rect x="110" y="60" width="44" height="16" rx="4" stroke="currentColor" stroke-opacity="0.28" stroke-width="1.2"/>
      <text x="132" y="71" text-anchor="middle" fill="currentColor" opacity="0.82" font-size="8.5" font-weight="600" font-family="{FONT_STACK}">4 Kelas</text>
    </svg>
  </div>
</div>
"""


def svg_icon(name: str, color: str = "currentColor") -> str:
    """Return a clean 18x18 line SVG icon."""
    icons = {
        "sliders": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "chart": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "table": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>',
        "check": f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    }
    return icons.get(name, icons["sliders"])


def svg_confidence_gauge(confidence: float, accent: str) -> str:
    """Return a clean SVG ring gauge representing model confidence."""
    radius = 32
    circumference = 2 * np.pi * radius
    clamped = max(0.0, min(100.0, confidence))
    offset = circumference * (1.0 - clamped / 100.0)
    return f"""
<svg width="84" height="84" viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg" aria-label="Skor Keyakinan">
  <circle cx="42" cy="42" r="{radius}" fill="none" stroke="rgba(128, 128, 128, 0.2)" stroke-width="6.5"/>
  <circle cx="42" cy="42" r="{radius}" fill="none" stroke="{accent}" stroke-width="6.5"
          stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"
          transform="rotate(-90 42 42)"/>
  <text x="42" y="40" text-anchor="middle" fill="currentColor" font-size="13.5" font-weight="700" font-family="{FONT_STACK}">{clamped:.1f}%</text>
  <text x="42" y="53" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="8" font-weight="600" font-family="{FONT_STACK}">KEYAKINAN</text>
</svg>
"""


def svg_phone_spec_blueprint(
    ram: int, battery_power: int, px_width: int, px_height: int, pred_class: int
) -> str:
    """Return a minimalist technical smartphone spec blueprint SVG."""
    style = CLASS_STYLES[pred_class]
    accent = style["accent"]
    megapixels = (px_width * px_height) / 1_000_000
    segments = "".join(
        f'<rect x="{118 + idx * 46}" y="128" width="40" height="7" rx="3.5" '
        f'fill="{accent if idx <= pred_class else "rgba(128, 128, 128, 0.22)"}"/>'
        for idx in range(4)
    )
    return f"""
<div class="blueprint-box">
  <svg width="100%" viewBox="0 0 316 148" xmlns="http://www.w3.org/2000/svg" aria-label="Spesifikasi Ponsel">
    <rect x="14" y="10" width="78" height="128" rx="12" fill="none" stroke="currentColor" stroke-opacity="0.7" stroke-width="2"/>
    <rect x="21" y="21" width="64" height="104" rx="6" fill="{style['tint']}" stroke="{accent}" stroke-width="1.2"/>
    <line x1="43" y1="15.5" x2="63" y2="15.5" stroke="currentColor" stroke-opacity="0.4" stroke-width="2" stroke-linecap="round"/>
    <text x="53" y="45" text-anchor="middle" fill="{accent}" font-size="10" font-weight="700" font-family="{FONT_STACK}">KELAS {pred_class}</text>
    <line x1="31" y1="54" x2="75" y2="54" stroke="{accent}" stroke-opacity="0.3" stroke-width="1"/>
    <text x="53" y="72" text-anchor="middle" fill="currentColor" font-size="9.5" font-weight="700" font-family="{FONT_STACK}">{ram} MB</text>
    <text x="53" y="83" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="7.5" font-family="{FONT_STACK}">RAM</text>
    <text x="53" y="103" text-anchor="middle" fill="currentColor" font-size="9.5" font-weight="700" font-family="{FONT_STACK}">{battery_power} mAh</text>
    <text x="53" y="114" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="7.5" font-family="{FONT_STACK}">BATERAI</text>

    <text x="118" y="25" fill="currentColor" font-size="11" font-weight="700" font-family="{FONT_STACK}">Ringkasan Spesifikasi Input</text>
    <text x="118" y="47" fill="currentColor" opacity="0.8" font-size="10" font-family="{FONT_STACK}">Memori RAM: {ram:,} MB ({ram / 1024:.2f} GB)</text>
    <text x="118" y="66" fill="currentColor" opacity="0.8" font-size="10" font-family="{FONT_STACK}">Baterai: {battery_power:,} mAh</text>
    <text x="118" y="85" fill="currentColor" opacity="0.8" font-size="10" font-family="{FONT_STACK}">Resolusi: {px_width} × {px_height} px ({megapixels:.2f} MP)</text>
    <text x="118" y="116" fill="{accent}" font-size="9.5" font-weight="700" font-family="{FONT_STACK}">Posisi Segmen Harga ({pred_class + 1} dari 4)</text>
    {segments}
  </svg>
</div>
"""


@st.cache_resource
def load_model_bundle(model_path: Path) -> tuple[Any, list[str]]:
    """Load the serialized RandomForest model and feature list from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di: {model_path}. "
            "Pastikan 'random_forest_mobile_price_4fitur.pkl' berada di direktori utama."
        )

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        features = list(loaded.get("fitur", FEATURE_COLUMNS))
    else:
        model = loaded
        features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))

    return model, features


def init_session_state() -> None:
    """Initialize default feature values in Streamlit session state."""
    for feature, spec in FEATURE_SPECS.items():
        if feature not in st.session_state:
            st.session_state[feature] = spec["default"]


def apply_preset(preset_key: str) -> None:
    """Populate session state with preset smartphone specifications."""
    for feature, value in PRESETS[preset_key]["values"].items():
        st.session_state[feature] = value


def render_sidebar(model: Any, features: list[str]) -> None:
    """Render sidebar with clean SVG mark, model summary, and target class reference."""
    with st.sidebar:
        st.html(
            f"""
<div class="sidebar-brand">
  <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="1" y="1" width="34" height="34" rx="8" stroke="currentColor" stroke-opacity="0.22" stroke-width="1.2"/>
    <circle cx="18" cy="10" r="2.8" fill="#2563EB"/>
    <circle cx="10" cy="25" r="2.8" fill="#0D9488"/>
    <circle cx="26" cy="25" r="2.8" fill="#D97706"/>
    <path d="M16.5 12.5L11.5 22.5M19.5 12.5L24.5 22.5" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.4" stroke-linecap="round"/>
  </svg>
  <div>
    <div style="font-weight:700; font-size:0.96rem; font-family:{FONT_STACK};">Informasi Model</div>
    <div style="font-size:0.78rem; opacity:0.7; font-family:{FONT_STACK};">Random Forest • 4 Fitur</div>
  </div>
</div>
            """
        )

        st.markdown(
            f"""
            - **Algoritma:** Random Forest (`RandomForestClassifier`)
            - **Jumlah Pohon:** `{getattr(model, "n_estimators", "-")}`
            - **Fitur Utama ({len(features)}):**
              1. `ram` (MB)
              2. `battery_power` (mAh)
              3. `px_width` (piksel)
              4. `px_height` (piksel)
            """
        )

        st.divider()
        st.subheader("4 Kelas Target")
        class_rows = "".join(
            f"""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:0.87rem; font-family:{FONT_STACK};">
  <span style="width:9px; height:9px; border-radius:999px; background:{CLASS_STYLES[cls_id]['accent']}; display:inline-block; flex-shrink:0;"></span>
  <span><b>{cls_id}</b> — {label}</span>
</div>
            """
            for cls_id, label in CLASS_LABELS.items()
        )
        st.html(class_rows)

        if hasattr(model, "feature_importances_"):
            st.divider()
            st.subheader("Tingkat Pengaruh Fitur")
            imp_df = pd.DataFrame(
                {
                    "Fitur": features,
                    "Bobot (%)": np.round(model.feature_importances_ * 100, 2),
                }
            ).sort_values("Bobot (%)", ascending=False)
            st.dataframe(imp_df, hide_index=True, width="stretch")


def build_probability_dataframe(model: Any, probabilities: np.ndarray) -> pd.DataFrame:
    """Create an ordered DataFrame of class probabilities across all 4 target classes."""
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2, 3])]
    prob_map = dict(zip(classes, probabilities))

    rows = []
    for cls_id in sorted(CLASS_LABELS.keys()):
        prob = float(prob_map.get(cls_id, 0.0))
        rows.append(
            {
                "Kelas": f"{cls_id} • {CLASS_LABELS[cls_id]}",
                "Probabilitas (%)": round(prob * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def render_single_prediction_tab(model: Any, features: list[str]) -> None:
    """Render Tab 1 for single smartphone price classification."""
    st.html(
        f"""
<div class="section-heading">
  {svg_icon("sliders")}
  <h3>Prediksi Satuan</h3>
</div>
        """
    )
    st.caption(
        "Isi 4 spesifikasi ponsel atau pilih salah satu contoh cepat untuk demonstrasi langsung."
    )

    st.markdown("**Contoh Cepat:**")
    preset_cols = st.columns(3)
    for col, (preset_key, preset_data) in zip(preset_cols, PRESETS.items()):
        with col:
            st.button(
                preset_data["button_label"],
                key=f"btn_preset_{preset_key}",
                on_click=apply_preset,
                args=(preset_key,),
                width="stretch",
            )

    st.markdown("")
    col_left, col_right = st.columns(2)

    with col_left:
        ram = st.number_input(
            FEATURE_SPECS["ram"]["label"],
            min_value=FEATURE_SPECS["ram"]["min"],
            max_value=FEATURE_SPECS["ram"]["max"],
            step=FEATURE_SPECS["ram"]["step"],
            key="ram",
            help=FEATURE_SPECS["ram"]["help"],
        )
        px_width = st.number_input(
            FEATURE_SPECS["px_width"]["label"],
            min_value=FEATURE_SPECS["px_width"]["min"],
            max_value=FEATURE_SPECS["px_width"]["max"],
            step=FEATURE_SPECS["px_width"]["step"],
            key="px_width",
            help=FEATURE_SPECS["px_width"]["help"],
        )

    with col_right:
        battery_power = st.number_input(
            FEATURE_SPECS["battery_power"]["label"],
            min_value=FEATURE_SPECS["battery_power"]["min"],
            max_value=FEATURE_SPECS["battery_power"]["max"],
            step=FEATURE_SPECS["battery_power"]["step"],
            key="battery_power",
            help=FEATURE_SPECS["battery_power"]["help"],
        )
        px_height = st.number_input(
            FEATURE_SPECS["px_height"]["label"],
            min_value=FEATURE_SPECS["px_height"]["min"],
            max_value=FEATURE_SPECS["px_height"]["max"],
            step=FEATURE_SPECS["px_height"]["step"],
            key="px_height",
            help=FEATURE_SPECS["px_height"]["help"],
        )

    input_df = pd.DataFrame(
        [
            {
                "ram": int(ram),
                "battery_power": int(battery_power),
                "px_width": int(px_width),
                "px_height": int(px_height),
            }
        ],
        columns=features,
    )

    pred_class = int(model.predict(input_df)[0])
    pred_probs = model.predict_proba(input_df)[0]
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2, 3])]
    confidence = float(dict(zip(classes, pred_probs)).get(pred_class, np.max(pred_probs))) * 100
    style = CLASS_STYLES[pred_class]

    st.divider()
    st.html(
        f"""
<div class="section-heading">
  {svg_icon("check", style["accent"])}
  <h3>Hasil Klasifikasi</h3>
</div>
        """
    )

    res_col1, res_col2 = st.columns([1.25, 1.75])
    with res_col1:
        st.html(
            f"""
<div class="result-panel" style="border-left: 4px solid {style['accent']};">
  <div class="result-panel-inner">
    <div>
      <span class="tier-tag" style="background:{style['tint']}; color:{style['accent']};">
        {style['tag']}
      </span>
      <h4 class="result-heading">{CLASS_LABELS[pred_class]}</h4>
      <p class="result-copy">{CLASS_DESCRIPTIONS[pred_class]}</p>
    </div>
    <div style="flex-shrink:0;">
      {svg_confidence_gauge(confidence, style['accent'])}
    </div>
  </div>
</div>
{svg_phone_spec_blueprint(int(ram), int(battery_power), int(px_width), int(px_height), pred_class)}
            """
        )

    with res_col2:
        st.html(
            f"""
<div class="section-heading">
  {svg_icon("chart", style["accent"])}
  <h4>Distribusi Probabilitas Tiap Kelas (%)</h4>
</div>
            """
        )
        prob_df = build_probability_dataframe(model, pred_probs)
        st.bar_chart(
            prob_df,
            x="Kelas",
            y="Probabilitas (%)",
            color=style["accent"],
            horizontal=True,
            sort=False,
            width="stretch",
        )

        with st.expander("Lihat Detail Tabel Probabilitas & Data Input"):
            st.dataframe(prob_df, hide_index=True, width="stretch")
            st.dataframe(input_df, hide_index=True, width="stretch")


def render_batch_prediction_tab(model: Any, features: list[str]) -> None:
    """Render Tab 2 for batch CSV upload, validation, prediction, and export."""
    st.html(
        f"""
<div class="section-heading">
  {svg_icon("table")}
  <h3>Prediksi Massal (Upload CSV)</h3>
</div>
        """
    )
    st.caption(
        "Unggah file `.csv` yang memuat kolom wajib: `ram`, `battery_power`, `px_width`, dan `px_height`."
    )

    sample_df = pd.DataFrame(
        [
            {"ram": 512, "battery_power": 900, "px_width": 720, "px_height": 480},
            {"ram": 1536, "battery_power": 1400, "px_width": 1280, "px_height": 720},
            {"ram": 2048, "battery_power": 1500, "px_width": 1280, "px_height": 720},
            {"ram": 8192, "battery_power": 5000, "px_width": 2400, "px_height": 1080},
        ]
    )
    st.download_button(
        label="Unduh Contoh Template CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="contoh_data_ponsel.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk diprediksi",
        type=["csv"],
        help="File CSV harus memiliki header kolom: ram, battery_power, px_width, px_height.",
    )

    if uploaded_file is None:
        return

    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Gagal membaca file CSV: {exc}")
        return

    if df_raw.empty:
        st.warning("File CSV yang diunggah tidak memiliki baris data.")
        return

    missing_cols = [col for col in features if col not in df_raw.columns]
    if missing_cols:
        st.error(
            "Kolom wajib berikut tidak ditemukan di file CSV: "
            + ", ".join(f"`{col}`" for col in missing_cols)
        )
        st.info("Kolom yang terdeteksi pada file Anda: " + ", ".join(f"`{c}`" for c in df_raw.columns))
        return

    df_features = df_raw[features].copy()
    for col in features:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    invalid_rows = df_features.isna().any(axis=1)
    if invalid_rows.any():
        invalid_count = int(invalid_rows.sum())
        st.error(
            f"Terdapat {invalid_count} baris dengan nilai kosong atau bukan angka pada kolom fitur. "
            "Harap perbaiki data terlebih dahulu."
        )
        st.dataframe(df_raw.loc[invalid_rows], width="stretch")
        return

    preds = model.predict(df_features)
    probs = model.predict_proba(df_features)
    confidence_scores = np.max(probs, axis=1) * 100

    df_result = df_raw.copy()
    df_result["prediksi_kelas"] = [int(p) for p in preds]
    df_result["kategori_harga"] = [CLASS_LABELS[int(p)] for p in preds]
    df_result["skor_kepercayaan (%)"] = np.round(confidence_scores, 2)

    st.success(f"Berhasil memprediksi **{len(df_result)}** data ponsel.")

    summary_counts = df_result["kategori_harga"].value_counts()
    metric_cols = st.columns(4)
    for col, cls_id in zip(metric_cols, sorted(CLASS_LABELS.keys())):
        label = CLASS_LABELS[cls_id]
        col.metric(label=f"Kelas {cls_id}", value=int(summary_counts.get(label, 0)), help=label)

    st.markdown("**Pratinjau Hasil Prediksi:**")
    st.dataframe(df_result, width="stretch")

    csv_output = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Unduh Hasil Prediksi (CSV)",
        data=csv_output,
        file_name="hasil_prediksi_harga_ponsel.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Klasifikasi Rentang Harga Ponsel",
        page_icon="📱",
        layout="wide",
    )

    inject_theme_styles()
    init_session_state()

    try:
        model, features = load_model_bundle(MODEL_PATH)
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    render_sidebar(model, features)
    st.html(svg_header_schematic())

    tab_single, tab_batch = st.tabs(["Prediksi Satuan", "Prediksi Massal (Upload CSV)"])

    with tab_single:
        render_single_prediction_tab(model, features)

    with tab_batch:
        render_batch_prediction_tab(model, features)


if __name__ == "__main__":
    main()

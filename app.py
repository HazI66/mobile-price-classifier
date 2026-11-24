"""
Klasifikasi Rentang Harga Ponsel - Random Forest (4 fitur)
Proyek UAS Pengenalan Pola, Kelompok 3 - Departemen Sistem Komputer, Universitas Sriwijaya

Cara menjalankan:  streamlit run app.py
Berkas model (random_forest_mobile_price_4fitur.pkl) harus berada satu folder dengan app.py.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Konfigurasi data & model
# --------------------------------------------------------------------------- #
MODEL_FILENAME = "random_forest_mobile_price_4fitur.pkl"
MODEL_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = ["ram", "battery_power", "px_width", "px_height"]

# Rentang slider dibulatkan dari rentang data latih (lihat Tabel 4.1 laporan):
# ram 256-3.998, battery_power 501-1.998, px_width 500-1.998, px_height 0-1.960.
# Nilai di luar rentang ini tidak pernah dilihat model saat pelatihan.
SPECS = {
    "ram": {
        "label": "RAM", "unit": "MB", "min": 256, "max": 4000, "step": 32, "default": 2048,
        "help": "Kapasitas RAM dalam megabyte. Rentang data latih: 256-3.998 MB.",
    },
    "battery_power": {
        "label": "Kapasitas baterai", "unit": "mAh", "min": 500, "max": 2000, "step": 10, "default": 1250,
        "help": "Kapasitas baterai dalam mAh. Rentang data latih: 501-1.998 mAh.",
    },
    "px_width": {
        "label": "Lebar resolusi layar", "unit": "px", "min": 500, "max": 2000, "step": 10, "default": 1250,
        "help": "Jumlah piksel horizontal. Rentang data latih: 500-1.998 piksel.",
    },
    "px_height": {
        "label": "Tinggi resolusi layar", "unit": "px", "min": 0, "max": 2000, "step": 10, "default": 650,
        "help": "Jumlah piksel vertikal. Rentang data latih: 0-1.960 piksel.",
    },
}

# Bobot fitur model 20 fitur (tuning pertama) pada Tabel 4.3 laporan, sebagai pembanding.
REPORT_IMPORTANCE_20 = {"ram": 53.03, "battery_power": 7.36, "px_width": 5.44, "px_height": 5.24}

# Angka dari laporan (bersifat eksploratif, lihat subbab 4.10 dan 4.11).
REPORT_METRICS = [
    ("94%", "Akurasi data uji", "model 4 fitur, 400 data uji"),
    ("0,9056", "F1-macro cross-validation", "5 lipatan, std 0,0262"),
    ("2.000", "Data pelatihan model final", "20 fitur asli, 4 dipakai"),
]

CLASSES = {
    0: {"name": "Entry-Level", "full": "Low Cost (Entry-Level)", "color": "#14b8a6", "tint": "rgba(20,184,166,.12)",
        "desc": "Ponsel kelas pemula dengan spesifikasi dasar untuk kebutuhan harian yang ringan."},
    1: {"name": "Mid-Range", "full": "Medium Cost (Mid-Range)", "color": "#3b82f6", "tint": "rgba(59,130,246,.12)",
        "desc": "Ponsel kelas menengah dengan performa seimbang untuk pemakaian sehari-hari."},
    2: {"name": "Upper Mid-Range", "full": "High Cost (Upper Mid-Range)", "color": "#8b5cf6", "tint": "rgba(139,92,246,.12)",
        "desc": "Ponsel menengah atas dengan RAM besar dan layar beresolusi tinggi."},
    3: {"name": "Flagship", "full": "Very High Cost (Flagship)", "color": "#f59e0b", "tint": "rgba(245,158,11,.14)",
        "desc": "Ponsel unggulan dengan spesifikasi tinggi di hampir semua aspek."},
}

PRESETS = {
    0: {"label": "🌱 Entry-Level", "values": {"ram": 512, "battery_power": 900, "px_width": 720, "px_height": 480}},
    1: {"label": "⚡ Mid-Range", "values": {"ram": 1536, "battery_power": 1400, "px_width": 1280, "px_height": 720}},
    2: {"label": "🔥 Upper Mid-Range", "values": {"ram": 2496, "battery_power": 1400, "px_width": 1200, "px_height": 700}},
    3: {"label": "👑 Flagship", "values": {"ram": 3488, "battery_power": 1800, "px_width": 1700, "px_height": 1200}},
}

FONT = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


# --------------------------------------------------------------------------- #
# Fungsi bantu
# --------------------------------------------------------------------------- #
def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def fmt_dec(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def load_model_bundle_uncached() -> tuple[Any, list[str]]:
    path = MODEL_DIR / MODEL_FILENAME
    if not path.exists():
        matches = sorted(MODEL_DIR.glob("random_forest_mobile_price_4fitur*.pkl"))
        if not matches:
            raise FileNotFoundError(
                f"Berkas model '{MODEL_FILENAME}' tidak ditemukan di {MODEL_DIR}. "
                "Letakkan berkas .pkl satu folder dengan app.py."
            )
        path = matches[0]
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        features = list(loaded.get("fitur", FEATURE_COLUMNS))
    else:
        model = loaded
        features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    return model, features


@st.cache_resource(show_spinner="Memuat model...")
def load_model_bundle() -> tuple[Any, list[str]]:
    return load_model_bundle_uncached()


def predict_proba_ordered(model: Any, df: pd.DataFrame) -> np.ndarray:
    """Probabilitas dengan urutan kolom tetap kelas 0..3."""
    raw = model.predict_proba(df)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1, 2, 3])]
    out = np.zeros((len(df), 4))
    for j, c in enumerate(classes):
        out[:, c] = raw[:, j]
    return out


def init_session_state() -> None:
    for key, spec in SPECS.items():
        st.session_state.setdefault(key, spec["default"])


def apply_preset(class_id: int) -> None:
    for key, value in PRESETS[class_id]["values"].items():
        st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Gaya (CSS)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
    font-family: __FONT__;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

.stApp {
    background:
        radial-gradient(900px 500px at 0% -5%, rgba(196,181,253,.38), transparent 60%),
        radial-gradient(800px 460px at 100% 0%, rgba(251,207,232,.42), transparent 55%),
        #faf8ff;
    color: #1e1b4b;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; color: #fff;
    border-radius: 28px; padding: 2.1rem 2.3rem; margin-bottom: 1.4rem;
    background: linear-gradient(118deg, #4338ca 0%, #7c3aed 52%, #db2777 100%);
    box-shadow: 0 22px 50px rgba(91,33,182,.28);
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
}
.hero:before, .hero:after { content: ""; position: absolute; border-radius: 50%; pointer-events: none; }
.hero:before { width: 340px; height: 340px; right: -90px; top: -150px; background: rgba(255,255,255,.10); }
.hero:after  { width: 220px; height: 220px; left: 38%; bottom: -150px; background: rgba(255,255,255,.07); }
.hero-text { position: relative; z-index: 1; max-width: 640px; }
.hero-kicker { font-size: .72rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; opacity: .82; margin-bottom: .55rem; }
.hero-title { margin: 0 0 .6rem 0; font-size: clamp(1.7rem, 3.2vw, 2.45rem); font-weight: 800; letter-spacing: -.035em; line-height: 1.12; }
.hero-sub { margin: 0 0 1.1rem 0; font-size: .98rem; line-height: 1.6; opacity: .92; }
.hero-sub code { background: rgba(255,255,255,.18); color: #fff; padding: .08rem .4rem; border-radius: 6px; font-size: .86em; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip { background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.28); backdrop-filter: blur(4px);
        padding: .28rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 700; }
.hero-art { position: relative; z-index: 1; flex-shrink: 0; }

/* ---------- kartu ---------- */
.card {
    background: #fff; border: 1px solid #ece9fe; border-radius: 22px;
    box-shadow: 0 10px 30px rgba(76,29,149,.07); padding: 1.25rem 1.4rem; margin-bottom: 1rem;
    color: #1e1b4b;
}
.card-title { font-size: .98rem; font-weight: 800; letter-spacing: -.01em; margin: 0 0 .85rem 0; display: flex; align-items: center; gap: .5rem; }
.card-title .bar { width: 4px; height: 1.05rem; border-radius: 4px; background: linear-gradient(180deg,#7c3aed,#db2777); display: inline-block; }
.muted { color: #6b7280; }
.note { font-size: .78rem; color: #6b7280; line-height: 1.55; margin: .85rem 0 0 0; }

.section-title { font-size: 1.15rem; font-weight: 800; letter-spacing: -.02em; margin: .3rem 0 .15rem 0; color: #1e1b4b; }
.section-sub { font-size: .88rem; color: #6b7280; margin: 0 0 .9rem 0; }

/* ---------- hasil klasifikasi ---------- */
.result { border-top-width: 5px; border-top-style: solid; }
.result-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.tier-tag { display: inline-block; font-size: .72rem; font-weight: 800; letter-spacing: .04em; padding: .28rem .75rem; border-radius: 999px; margin-bottom: .55rem; }
.result-title { margin: 0 0 .3rem 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.2; }
.result-copy { margin: 0; font-size: .9rem; line-height: 1.55; color: #5b6478; }
.spec-panel { display: flex; gap: 1.1rem; align-items: center; margin-top: 1.15rem; padding-top: 1.1rem; border-top: 1px dashed #e5e1fb; }
.spec-list { flex: 1; min-width: 0; }
.spec-row { display: flex; justify-content: space-between; gap: .75rem; font-size: .84rem; padding: .22rem 0; }
.spec-row span:first-child { color: #6b7280; }
.spec-row b { font-weight: 700; text-align: right; }
.segments { display: flex; gap: .35rem; margin-top: .65rem; }
.segments i { flex: 1; height: 7px; border-radius: 99px; background: #e9e5fb; display: block; }
.segments-cap { font-size: .72rem; font-weight: 700; margin-top: .35rem; }

/* ---------- peluang kelas ---------- */
.prob-row { margin-bottom: .95rem; padding: .55rem .7rem; border-radius: 14px; border: 1px solid transparent; }
.prob-row.active { background: #faf8ff; border-color: #ece9fe; }
.prob-head { display: flex; justify-content: space-between; align-items: center; font-size: .86rem; margin-bottom: .4rem; }
.prob-head span { display: flex; align-items: center; gap: .45rem; font-weight: 600; }
.prob-head b { font-weight: 800; font-variant-numeric: tabular-nums; }
.dot { width: 10px; height: 10px; border-radius: 99px; display: inline-block; }
.track { height: 9px; border-radius: 99px; background: #eeeafc; overflow: hidden; position: relative; }
.fill { height: 100%; border-radius: 99px; }

/* ---------- posisi input ---------- */
.pos-row { margin-bottom: .8rem; }
.pos-head { display: flex; justify-content: space-between; font-size: .8rem; margin-bottom: .3rem; }
.pos-head span { color: #6b7280; font-weight: 600; }
.pos-head b { font-weight: 700; }
.pos-track { height: 6px; border-radius: 99px; background: linear-gradient(90deg,#ddd6fe,#fbcfe8); position: relative; }
.pos-dot { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #fff;
           border: 3px solid #7c3aed; transform: translate(-50%, -50%); box-shadow: 0 2px 6px rgba(76,29,149,.3); }
.pos-scale { display: flex; justify-content: space-between; font-size: .68rem; color: #9ca3af; margin-top: .25rem; }

/* ---------- ringkasan massal ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: .4rem 0 1.1rem 0; }
.stat { background: #fff; border: 1px solid #ece9fe; border-radius: 18px; padding: .95rem 1.05rem; box-shadow: 0 8px 22px rgba(76,29,149,.06); border-top: 4px solid; }
.stat-n { font-size: 1.7rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.1; }
.stat-l { font-size: .78rem; font-weight: 700; margin-top: .15rem; }
.stat-s { font-size: .74rem; color: #6b7280; }
.stackbar { display: flex; height: 12px; border-radius: 99px; overflow: hidden; background: #eeeafc; margin: .2rem 0 .3rem 0; }

/* ---------- tentang model ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-bottom: 1rem; }
.kpi { background: linear-gradient(135deg,#ffffff,#f7f3ff); border: 1px solid #ece9fe; border-radius: 18px; padding: 1rem 1.1rem; }
.kpi-n { font-size: 1.65rem; font-weight: 800; letter-spacing: -.03em; background: linear-gradient(110deg,#4338ca,#db2777); -webkit-background-clip: text; background-clip: text; color: transparent; }
.kpi-l { font-size: .82rem; font-weight: 700; }
.kpi-s { font-size: .74rem; color: #6b7280; }
.imp-row { margin-bottom: 1.05rem; }
.imp-head { display: flex; justify-content: space-between; align-items: baseline; font-size: .88rem; margin-bottom: .35rem; }
.imp-head code { background: #f3efff; color: #5b21b6; padding: .06rem .45rem; border-radius: 6px; font-size: .84em; font-weight: 700; }
.imp-head b { font-weight: 800; }
.imp-ref { font-size: .73rem; color: #6b7280; margin-top: .3rem; }
.param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .5rem .9rem; }
.param { display: flex; justify-content: space-between; font-size: .84rem; padding: .35rem 0; border-bottom: 1px dashed #ece9fe; }
.param span { color: #6b7280; }
.param b { font-weight: 700; }
.legend-row { display: flex; align-items: flex-start; gap: .6rem; margin-bottom: .65rem; font-size: .86rem; }
.legend-row small { display: block; color: #6b7280; font-size: .76rem; line-height: 1.4; }

/* ---------- langkah ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-bottom: 1rem; }
.step { background: #fff; border: 1px solid #ece9fe; border-radius: 18px; padding: .9rem 1rem; font-size: .85rem; line-height: 1.5; }
.step b { display: block; margin-bottom: .1rem; }
.step-n { width: 26px; height: 26px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
          font-size: .78rem; font-weight: 800; color: #fff; background: linear-gradient(135deg,#7c3aed,#db2777); margin-bottom: .45rem; }

.footer { text-align: center; font-size: .78rem; color: #8b8fa3; margin-top: 1.6rem; line-height: 1.6; }

/* ---------- komponen bawaan Streamlit ---------- */
div[data-testid="stTabs"] > div[role="tablist"] { gap: .4rem; border-bottom: 1px solid #ece9fe; }
div[data-testid="stTabs"] button[role="tab"] { border-radius: 12px 12px 0 0; padding: .7rem 1.15rem; font-weight: 700; color: #6b7280; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: #6d28d9; background: rgba(124,58,237,.08); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: linear-gradient(90deg,#7c3aed,#db2777) !important; height: 3px; }

.stButton > button, .stDownloadButton > button {
    border-radius: 14px; font-weight: 700; border: 1px solid #ddd6fe; background: #fff; color: #5b21b6;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); border-color: #a78bfa; color: #6d28d9; box-shadow: 0 10px 22px rgba(124,58,237,.16); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: linear-gradient(110deg,#6d28d9,#db2777); color: #fff; border: 0; }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 22px; }
[data-testid="stSlider"] [role="slider"] { background-color: #7c3aed !important; box-shadow: 0 0 0 4px rgba(124,58,237,.18); }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #6d28d9; font-weight: 800; }
[data-testid="stFileUploader"] section { border: 2px dashed #c4b5fd; border-radius: 18px; background: rgba(245,243,255,.8); }
[data-testid="stDataFrame"] { border: 1px solid #ece9fe; border-radius: 16px; overflow: hidden; }
div[data-testid="stExpander"] { border: 1px solid #ece9fe; border-radius: 16px; background: rgba(255,255,255,.8); }

@media (max-width: 800px) {
    .hero { flex-direction: column; align-items: flex-start; padding: 1.5rem; }
    .hero-art { display: none; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .kpi-grid, .steps, .param-grid { grid-template-columns: 1fr; }
    .result-top, .spec-panel { flex-direction: column; align-items: flex-start; }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS.replace("__FONT__", FONT), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Potongan HTML / SVG
# --------------------------------------------------------------------------- #
def html_block(markup: str) -> None:
    st.html(markup)


def hero_html(model: Any, n_features: int) -> str:
    n_trees = getattr(model, "n_estimators", "-")
    return f"""
<div class="hero">
  <div class="hero-text">
    <div class="hero-kicker">Proyek UAS · Pengenalan Pola</div>
    <h1 class="hero-title">Klasifikasi Rentang Harga Ponsel</h1>
    <p class="hero-sub">Masukkan spesifikasi perangkat keras, lalu model <b>Random Forest</b> memperkirakan
    ponsel itu masuk ke segmen harga yang mana. Hanya butuh empat fitur:
    <code>ram</code> <code>battery_power</code> <code>px_width</code> <code>px_height</code></p>
    <div class="chips">
      <span class="chip">🌲 {n_trees} pohon</span>
      <span class="chip">🧩 {n_features} fitur</span>
      <span class="chip">🏷️ 4 kelas harga</span>
    </div>
  </div>
  <div class="hero-art">
    <svg width="190" height="150" viewBox="0 0 190 150" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Ilustrasi ponsel">
      <rect x="58" y="8" width="74" height="134" rx="16" fill="rgba(255,255,255,.14)" stroke="#fff" stroke-opacity=".85" stroke-width="2.2"/>
      <rect x="65" y="22" width="60" height="106" rx="8" fill="rgba(255,255,255,.12)"/>
      <line x1="84" y1="15" x2="106" y2="15" stroke="#fff" stroke-opacity=".7" stroke-width="2.4" stroke-linecap="round"/>
      <rect x="72" y="92" width="9" height="26" rx="3" fill="#5eead4"/>
      <rect x="85" y="76" width="9" height="42" rx="3" fill="#93c5fd"/>
      <rect x="98" y="58" width="9" height="60" rx="3" fill="#c4b5fd"/>
      <rect x="111" y="40" width="9" height="78" rx="3" fill="#fcd34d"/>
      <circle cx="22" cy="40" r="5" fill="#fff" fill-opacity=".85"/>
      <circle cx="168" cy="108" r="6" fill="#fff" fill-opacity=".85"/>
      <path d="M27 40H52M138 108H162" stroke="#fff" stroke-opacity=".5" stroke-width="1.6" stroke-dasharray="3 4"/>
    </svg>
  </div>
</div>
"""


def gauge_svg(confidence: float, color: str) -> str:
    radius = 34
    circumference = 2 * np.pi * radius
    pct = max(0.0, min(100.0, confidence))
    offset = circumference * (1 - pct / 100)
    return f"""
<svg width="96" height="96" viewBox="0 0 96 96" xmlns="http://www.w3.org/2000/svg" aria-label="Skor keyakinan">
  <circle cx="48" cy="48" r="{radius}" fill="none" stroke="#eeeafc" stroke-width="8"/>
  <circle cx="48" cy="48" r="{radius}" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round"
          stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" transform="rotate(-90 48 48)"/>
  <text x="48" y="48" text-anchor="middle" fill="#1e1b4b" font-size="16" font-weight="800" font-family="{FONT}">{fmt_dec(pct)}%</text>
  <text x="48" y="62" text-anchor="middle" fill="#6b7280" font-size="7.5" font-weight="700" letter-spacing=".08em" font-family="{FONT}">KEYAKINAN</text>
</svg>
"""


def phone_svg(pred_class: int, ram: int, battery: int) -> str:
    c = CLASSES[pred_class]["color"]
    return f"""
<svg width="92" height="150" viewBox="0 0 92 150" xmlns="http://www.w3.org/2000/svg" aria-label="Ponsel">
  <defs>
    <linearGradient id="scr" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{c}" stop-opacity=".16"/>
      <stop offset="1" stop-color="{c}" stop-opacity=".42"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="84" height="142" rx="16" fill="#fff" stroke="#1e1b4b" stroke-opacity=".75" stroke-width="2.2"/>
  <rect x="10" y="16" width="72" height="118" rx="9" fill="url(#scr)" stroke="{c}" stroke-width="1.2"/>
  <line x1="34" y1="10" x2="58" y2="10" stroke="#1e1b4b" stroke-opacity=".4" stroke-width="2.4" stroke-linecap="round"/>
  <text x="46" y="44" text-anchor="middle" fill="{c}" font-size="9.5" font-weight="800" letter-spacing=".06em" font-family="{FONT}">KELAS {pred_class}</text>
  <line x1="22" y1="54" x2="70" y2="54" stroke="{c}" stroke-opacity=".35"/>
  <text x="46" y="78" text-anchor="middle" fill="#1e1b4b" font-size="11" font-weight="800" font-family="{FONT}">{fmt_int(ram)}</text>
  <text x="46" y="89" text-anchor="middle" fill="#6b7280" font-size="7.5" font-weight="700" font-family="{FONT}">MB RAM</text>
  <text x="46" y="110" text-anchor="middle" fill="#1e1b4b" font-size="11" font-weight="800" font-family="{FONT}">{fmt_int(battery)}</text>
  <text x="46" y="121" text-anchor="middle" fill="#6b7280" font-size="7.5" font-weight="700" font-family="{FONT}">mAh BATERAI</text>
</svg>
"""


def result_card_html(pred_class: int, confidence: float, inputs: dict[str, int]) -> str:
    c = CLASSES[pred_class]
    megapixels = inputs["px_width"] * inputs["px_height"] / 1_000_000
    segments = "".join(
        f'<i style="background:{c["color"] if i <= pred_class else "#e9e5fb"}"></i>' for i in range(4)
    )
    return f"""
<div class="card result" style="border-top-color:{c['color']};">
  <div class="result-top">
    <div>
      <span class="tier-tag" style="background:{c['tint']};color:{c['color']};">KELAS {pred_class} · {c['name'].upper()}</span>
      <h3 class="result-title">{c['full']}</h3>
      <p class="result-copy">{c['desc']}</p>
    </div>
    <div style="flex-shrink:0;">{gauge_svg(confidence, c['color'])}</div>
  </div>
  <div class="spec-panel">
    <div style="flex-shrink:0;">{phone_svg(pred_class, inputs['ram'], inputs['battery_power'])}</div>
    <div class="spec-list">
      <div class="spec-row"><span>Memori RAM</span><b>{fmt_int(inputs['ram'])} MB ({fmt_dec(inputs['ram'] / 1024, 2)} GB)</b></div>
      <div class="spec-row"><span>Baterai</span><b>{fmt_int(inputs['battery_power'])} mAh</b></div>
      <div class="spec-row"><span>Resolusi layar</span><b>{fmt_int(inputs['px_width'])} × {fmt_int(inputs['px_height'])} px</b></div>
      <div class="spec-row"><span>Total piksel</span><b>{fmt_dec(megapixels, 2)} MP</b></div>
      <div class="segments">{segments}</div>
      <div class="segments-cap" style="color:{c['color']};">Segmen harga {pred_class + 1} dari 4</div>
    </div>
  </div>
</div>
"""


def probability_card_html(probs: np.ndarray, pred_class: int) -> str:
    rows = ""
    for cid in range(4):
        c = CLASSES[cid]
        pct = float(probs[cid]) * 100
        active = " active" if cid == pred_class else ""
        rows += f"""
<div class="prob-row{active}">
  <div class="prob-head"><span><i class="dot" style="background:{c['color']}"></i>Kelas {cid} · {c['name']}</span><b>{fmt_dec(pct)}%</b></div>
  <div class="track"><div class="fill" style="width:{max(pct, 0.6):.2f}%;background:linear-gradient(90deg,{c['color']},{c['color']}aa);"></div></div>
</div>"""
    return f"""
<div class="card">
  <div class="card-title"><span class="bar"></span>Peluang tiap kelas</div>
  {rows}
  <p class="note">Skor keyakinan adalah proporsi pohon dalam hutan yang memilih kelas tersebut. Nilainya
  menggambarkan keyakinan model, bukan jaminan bahwa prediksi pasti benar.</p>
</div>
"""


def position_card_html(inputs: dict[str, int]) -> str:
    rows = ""
    for key in FEATURE_COLUMNS:
        s = SPECS[key]
        pct = (inputs[key] - s["min"]) / (s["max"] - s["min"]) * 100
        pct = max(2.0, min(98.0, pct))
        rows += f"""
<div class="pos-row">
  <div class="pos-head"><span>{s['label']}</span><b>{fmt_int(inputs[key])} {s['unit']}</b></div>
  <div class="pos-track"><div class="pos-dot" style="left:{pct:.1f}%"></div></div>
  <div class="pos-scale"><span>{fmt_int(s['min'])}</span><span>{fmt_int(s['max'])}</span></div>
</div>"""
    return f"""
<div class="card">
  <div class="card-title"><span class="bar"></span>Posisi input dalam rentang data latih</div>
  {rows}
</div>
"""


def stat_cards_html(counts: dict[int, int], total: int) -> str:
    cards = ""
    for cid in range(4):
        c = CLASSES[cid]
        n = counts.get(cid, 0)
        share = n / total * 100 if total else 0
        cards += f"""
<div class="stat" style="border-top-color:{c['color']}">
  <div class="stat-n" style="color:{c['color']}">{fmt_int(n)}</div>
  <div class="stat-l">Kelas {cid} · {c['name']}</div>
  <div class="stat-s">{fmt_dec(share)}% dari seluruh data</div>
</div>"""
    stack = "".join(
        f'<div style="width:{counts.get(cid, 0) / total * 100:.2f}%;background:{CLASSES[cid]["color"]}"></div>'
        for cid in range(4) if total and counts.get(cid, 0)
    )
    return f'<div class="stat-grid">{cards}</div><div class="stackbar">{stack}</div>'


def importance_card_html(model: Any, features: list[str]) -> str:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return ""
    live = {f: float(v) * 100 for f, v in zip(features, imp)}
    rows = ""
    for rank, (feat, val) in enumerate(sorted(live.items(), key=lambda kv: kv[1], reverse=True)):
        ref = REPORT_IMPORTANCE_20.get(feat)
        ref_txt = f"Pada model 20 fitur (Tabel 4.3 laporan): {fmt_dec(ref, 2)}%" if ref is not None else ""
        grad = "linear-gradient(90deg,#6d28d9,#db2777)" if rank == 0 else "linear-gradient(90deg,#a78bfa,#f0abfc)"
        rows += f"""
<div class="imp-row">
  <div class="imp-head"><code>{feat}</code><b>{fmt_dec(val, 2)}%</b></div>
  <div class="track"><div class="fill" style="width:{val:.2f}%;background:{grad};"></div></div>
  <div class="imp-ref">{ref_txt}</div>
</div>"""
    return f"""
<div class="card">
  <div class="card-title"><span class="bar"></span>Bobot fitur model final (4 fitur)</div>
  {rows}
  <p class="note">Bobot di atas dihitung dari <b>model final yang dipakai aplikasi</b>, sehingga totalnya 100% di empat
  fitur. Pada model 20 fitur di laporan, bobot yang sama terbagi ke 20 fitur sehingga angkanya lebih kecil, tetapi
  <code>ram</code> tetap yang paling berpengaruh. Posisi <code>px_width</code> dan <code>px_height</code> bisa bertukar
  karena selisihnya tipis. Bobot ini menunjukkan peran relatif menurut model, bukan hubungan sebab-akibat.</p>
</div>
"""


def model_info_card_html(model: Any) -> str:
    params = [
        ("Algoritma", "RandomForestClassifier"),
        ("Jumlah pohon", getattr(model, "n_estimators", "-")),
        ("Kedalaman maksimum", getattr(model, "max_depth", "-")),
        ("min_samples_split", getattr(model, "min_samples_split", "-")),
        ("min_samples_leaf", getattr(model, "min_samples_leaf", "-")),
        ("max_features", getattr(model, "max_features", "-")),
    ]
    items = "".join(f'<div class="param"><span>{k}</span><b>{v}</b></div>' for k, v in params)
    return f"""
<div class="card">
  <div class="card-title"><span class="bar"></span>Konfigurasi model</div>
  <div class="param-grid">{items}</div>
</div>
"""


def legend_card_html() -> str:
    rows = "".join(
        f"""<div class="legend-row"><i class="dot" style="background:{c['color']};margin-top:.38rem"></i>
        <div><b>Kelas {cid} · {c['full']}</b><small>{c['desc']}</small></div></div>"""
        for cid, c in CLASSES.items()
    )
    return f"""
<div class="card">
  <div class="card-title"><span class="bar"></span>Arti keempat kelas</div>
  {rows}
</div>
"""


def kpi_html() -> str:
    cards = "".join(
        f'<div class="kpi"><div class="kpi-n">{n}</div><div class="kpi-l">{l}</div><div class="kpi-s">{s}</div></div>'
        for n, l, s in REPORT_METRICS
    )
    return f"""
<div class="kpi-grid">{cards}</div>
<p class="note" style="margin-top:-.3rem;margin-bottom:1rem;">Angka di atas diambil dari laporan dan bersifat eksploratif:
data uji ikut dipakai untuk memilih jumlah fitur, sehingga hasil pada data baru bisa lebih rendah.</p>
"""


# --------------------------------------------------------------------------- #
# Tab 1: prediksi satuan
# --------------------------------------------------------------------------- #
def render_single_tab(model: Any, features: list[str]) -> None:
    html_block('<div class="section-title">Atur spesifikasi ponsel</div>'
               '<p class="section-sub">Geser slider, atau mulai cepat dari salah satu contoh di bawah ini.</p>')

    cols = st.columns(4)
    for col, (cid, preset) in zip(cols, PRESETS.items()):
        with col:
            st.button(preset["label"], key=f"preset_{cid}", on_click=apply_preset, args=(cid,), width="stretch")

    with st.container(border=True):
        left, right = st.columns(2, gap="large")
        for col, keys in ((left, ("ram", "px_width")), (right, ("battery_power", "px_height"))):
            with col:
                for key in keys:
                    s = SPECS[key]
                    st.slider(
                        f"{s['label']} ({s['unit']})",
                        min_value=s["min"], max_value=s["max"], step=s["step"],
                        key=key, help=s["help"],
                    )

    inputs = {k: int(st.session_state[k]) for k in FEATURE_COLUMNS}
    input_df = pd.DataFrame([inputs], columns=features)
    probs = predict_proba_ordered(model, input_df)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class]) * 100

    html_block('<div class="section-title" style="margin-top:.6rem;">Hasil klasifikasi</div>')
    col_a, col_b = st.columns([1.15, 1], gap="large")
    with col_a:
        html_block(result_card_html(pred_class, confidence, inputs))
        with st.expander("Lihat tabel data input dan probabilitas"):
            prob_df = pd.DataFrame(
                {"Kelas": [f"{i} · {CLASSES[i]['full']}" for i in range(4)],
                 "Probabilitas (%)": np.round(probs * 100, 2)}
            )
            st.dataframe(prob_df, hide_index=True, width="stretch")
            st.dataframe(input_df, hide_index=True, width="stretch")
    with col_b:
        html_block(probability_card_html(probs, pred_class))
        html_block(position_card_html(inputs))


# --------------------------------------------------------------------------- #
# Tab 2: prediksi massal
# --------------------------------------------------------------------------- #
def sample_template() -> pd.DataFrame:
    return pd.DataFrame([p["values"] for p in PRESETS.values()], columns=FEATURE_COLUMNS)


def render_batch_tab(model: Any, features: list[str]) -> None:
    html_block("""
<div class="section-title">Prediksi banyak ponsel sekaligus</div>
<p class="section-sub">Unggah berkas CSV, aplikasi memeriksa datanya, memprediksi semua baris, lalu menyiapkan hasil untuk diunduh.</p>
<div class="steps">
  <div class="step"><span class="step-n">1</span><b>Siapkan CSV</b>Header wajib: <code>ram</code>, <code>battery_power</code>, <code>px_width</code>, <code>px_height</code>.</div>
  <div class="step"><span class="step-n">2</span><b>Unggah berkas</b>Seret ke kotak di bawah, atau pakai template contoh.</div>
  <div class="step"><span class="step-n">3</span><b>Unduh hasil</b>Kolom kelas, kategori harga, dan skor keyakinan ditambahkan.</div>
</div>
""")

    st.download_button(
        "⬇️ Unduh template CSV", data=sample_template().to_csv(index=False).encode("utf-8"),
        file_name="contoh_data_ponsel.csv", mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Pilih berkas CSV", type=["csv"],
        help="Kolom lain di berkas Anda dibiarkan apa adanya dan ikut muncul di hasil.",
    )
    if uploaded is None:
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal membaca berkas CSV: {exc}")
        return
    if df_raw.empty:
        st.warning("Berkas CSV tidak memiliki baris data.")
        return

    col_lookup = {str(c).strip().lower(): c for c in df_raw.columns}
    missing = [f for f in features if f not in col_lookup]
    if missing:
        st.error("Kolom wajib berikut tidak ditemukan: " + ", ".join(f"`{m}`" for m in missing))
        st.info("Kolom yang terbaca di berkas Anda: " + ", ".join(f"`{c}`" for c in df_raw.columns))
        return

    df_feat = pd.DataFrame({f: pd.to_numeric(df_raw[col_lookup[f]], errors="coerce") for f in features})
    bad = df_feat.isna().any(axis=1)
    if bad.any():
        st.error(f"Ada {int(bad.sum())} baris dengan nilai kosong atau bukan angka. Perbaiki dulu, lalu unggah ulang.")
        st.dataframe(df_raw.loc[bad], width="stretch")
        return

    out_of_range = pd.Series(False, index=df_feat.index)
    for f in features:
        if f in SPECS:
            out_of_range |= (df_feat[f] < SPECS[f]["min"]) | (df_feat[f] > SPECS[f]["max"])
    if out_of_range.any():
        st.warning(
            f"{int(out_of_range.sum())} baris punya nilai di luar rentang data latih "
            "(ram 256-4.000, baterai 500-2.000, lebar 500-2.000, tinggi 0-2.000). "
            "Hasil prediksinya tetap dihitung, tetapi kurang bisa diandalkan."
        )

    probs = predict_proba_ordered(model, df_feat[features])
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1) * 100

    df_out = df_raw.copy()
    df_out["prediksi_kelas"] = preds.astype(int)
    df_out["kategori_harga"] = [CLASSES[int(p)]["full"] for p in preds]
    df_out["skor_kepercayaan (%)"] = np.round(conf, 2)

    st.success(f"Berhasil memprediksi {len(df_out)} data ponsel.")
    counts = {cid: int((preds == cid).sum()) for cid in range(4)}
    html_block(stat_cards_html(counts, len(df_out)))

    st.markdown("**Pratinjau hasil prediksi**")
    st.dataframe(
        df_out, width="stretch", hide_index=True,
        column_config={
            "skor_kepercayaan (%)": st.column_config.ProgressColumn(
                "Keyakinan", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    st.download_button(
        "⬇️ Unduh hasil prediksi (CSV)", data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_harga_ponsel.csv", mime="text/csv", type="primary",
    )


# --------------------------------------------------------------------------- #
# Tab 3: tentang model
# --------------------------------------------------------------------------- #
def render_about_tab(model: Any, features: list[str]) -> None:
    html_block('<div class="section-title">Tentang model</div>'
               '<p class="section-sub">Ringkasan model yang dipakai aplikasi ini, dibaca langsung dari berkas model.</p>')
    html_block(kpi_html())
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        html_block(importance_card_html(model, features))
    with right:
        html_block(model_info_card_html(model))
        html_block(legend_card_html())
    html_block("""
<div class="card">
  <div class="card-title"><span class="bar"></span>Catatan penggunaan</div>
  <p class="note" style="margin-top:0;">Hasil klasifikasi hanyalah perkiraan segmen harga berdasarkan model dan dataset
  Mobile Price Classification, bukan harga aktual di pasaran. Dataset tidak memuat merek maupun tahun rilis, dan
  aplikasi hanya memakai empat fitur. Gunakan nilai di dalam rentang data latih supaya hasilnya masuk akal.</p>
</div>
""")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Klasifikasi Harga Ponsel", page_icon="📱", layout="wide")
    inject_styles()
    init_session_state()

    try:
        model, features = load_model_bundle()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    html_block(hero_html(model, len(features)))

    tab_single, tab_batch, tab_about = st.tabs(["📱 Prediksi satuan", "📂 Prediksi massal", "📊 Tentang model"])
    with tab_single:
        render_single_tab(model, features)
    with tab_batch:
        render_batch_tab(model, features)
    with tab_about:
        render_about_tab(model, features)

    html_block('<div class="footer">Proyek UAS Pengenalan Pola · Kelompok 3<br>'
               'Departemen Sistem Komputer, Fakultas Ilmu Komputer, Universitas Sriwijaya</div>')


if __name__ == "__main__":
    main()

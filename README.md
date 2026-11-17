# 📱 Klasifikasi Rentang Harga Ponsel (Proyek Akhir Pengenalan Pola)

Aplikasi web interaktif berbasis **Streamlit** untuk mengklasifikasikan rentang harga ponsel menggunakan model **Random Forest Classifier** (`random_forest_mobile_price_4fitur.pkl`) dengan 4 fitur perangkat keras utama.

## Spesifikasi Model
- **Algoritma:** `RandomForestClassifier` (400 pohon keputusan, tanpa scaler/encoder eksternal)
- **File Model:** `random_forest_mobile_price_4fitur.pkl` (diserialisasi dengan `joblib` pada `scikit-learn==1.6.1`)
- **4 Fitur Input (berurutan):**
  1. `ram` — Kapasitas RAM dalam MB (`256–16384`)
  2. `battery_power` — Kapasitas baterai dalam mAh (`500–6000`)
  3. `px_width` — Lebar resolusi layar dalam piksel (`500–4000`)
  4. `px_height` — Tinggi resolusi layar dalam piksel (`0–4000`)
- **4 Kelas Target:**
  - `0`: Low Cost (Entry-Level)
  - `1`: Medium Cost (Mid-Range)
  - `2`: High Cost (Upper Mid-Range)
  - `3`: Very High Cost (Flagship)

## Fitur Aplikasi
- **Prediksi Satuan:** Input 4 spesifikasi dalam 2 kolom, 3 tombol preset cepat (*Entry-Level*, *Mid-Range*, *Flagship*), hasil klasifikasi kelas harga, dan grafik batang distribusi probabilitas.
- **Prediksi Massal (Upload CSV):** Unggah file `.csv`, validasi kolom otomatis, tabel pratinjau beserta skor kepercayaan, dan tombol unduh hasil prediksi `.csv`.

## Struktur File
```text
.
├── app.py
├── random_forest_mobile_price_4fitur.pkl
├── requirements.txt
└── README.md
```

## Menjalankan Secara Lokal

1. Pastikan Python 3.10+ sudah terpasang, lalu buat dan aktifkan virtual environment (disarankan):
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # Linux/macOS
   source .venv/bin/activate
   ```

2. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

4. Buka alamat lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## Deploy ke Streamlit Community Cloud

1. Unggah `app.py`, `requirements.txt`, `README.md`, dan `random_forest_mobile_price_4fitur.pkl` ke repositori GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan masuk dengan akun GitHub Anda.
3. Klik **Create app** / **Deploy an app**, lalu pilih:
   - **Repository:** repositori GitHub proyek Anda
   - **Branch:** `main` (atau branch utama Anda)
   - **Main file path:** `app.py`
4. Klik **Deploy** dan tunggu hingga proses instalasi `requirements.txt` selesai.

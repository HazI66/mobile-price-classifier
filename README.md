# Klasifikasi Rentang Harga Ponsel

Aplikasi web sederhana berbasis Streamlit untuk tugas akhir mata kuliah Pengenalan Pola. Tujuannya untuk memprediksi kelas rentang harga ponsel dari empat spesifikasi utama menggunakan model Random Forest yang sudah dilatih sebelumnya (`random_forest_mobile_price_4fitur.pkl`).

## Tentang Model

Model dibuat menggunakan `RandomForestClassifier` (400 pohon keputusan, disimpan dengan `joblib` pada `scikit-learn==1.6.1`) dan tidak memakai scaler atau encoder tambahan—data angka langsung masuk ke model.

Input yang dipakai ada 4 fitur dengan urutan berikut:
- `ram` — kapasitas RAM dalam MB (256–16384)
- `battery_power` — kapasitas baterai dalam mAh (500–6000)
- `px_width` — lebar resolusi layar dalam piksel (500–4000)
- `px_height` — tinggi resolusi layar dalam piksel (0–4000)

Output prediksinya dibagi menjadi 4 kategori harga:
- `0`: Low Cost (Entry-Level)
- `1`: Medium Cost (Mid-Range)
- `2`: High Cost (Upper Mid-Range)
- `3`: Very High Cost (Flagship)

## Fitur di Aplikasi

1. **Prediksi Satuan** — Mengisi spesifikasi satu ponsel secara manual atau memakai tiga tombol contoh cepat (*Entry-Level*, *Mid-Range*, *Flagship*) untuk demo presentasi, lalu menampilkan kelas prediksi beserta grafik probabilitasnya.
2. **Prediksi Massal (Upload CSV)** — Mengunggah file `.csv` berisi empat kolom fitur di atas, mengecek kelengkapan kolom, menampilkan tabel hasil prediksi + skor keyakinan, dan menyediakan tombol unduh hasilnya ke CSV.

## Cara Menjalankan di Komputer Sendiri

Pastikan Python 3.10 atau yang lebih baru sudah terpasang, lalu jalankan perintah berikut di folder proyek:

```bash
python -m venv .venv
# Aktifkan virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Setelah berjalan, buka `http://localhost:8501` di browser.


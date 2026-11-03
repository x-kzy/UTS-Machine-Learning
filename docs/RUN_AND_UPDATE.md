# Menjalankan dan memperbarui proyek

## Mengunggah paket awal dari CMD

1. Unduh dan ekstrak seluruh paket repository.
2. Buka folder `UTS-Machine-Learning` sampai terlihat `PUSH_KE_GITHUB.cmd`.
3. Ketik `cmd` pada address bar File Explorer lalu tekan Enter.
4. Jalankan:

```bat
PUSH_KE_GITHUB.cmd
```

Script mengambil `main` dari `https://github.com/x-kzy/UTS-Machine-Learning`, menyalin file yang tercantum pada manifest, membuat commit, lalu melakukan push. Lisensi yang sudah ada dipertahankan. Dataset CSV/ZIP besar tidak termasuk manifest. Script menggunakan Git dan Windows PowerShell; Python tidak diperlukan untuk upload. Jika Git belum terpasang, pasang Git for Windows dari https://git-scm.com/download/win lalu buka CMD baru. Bila diminta login, gunakan akun yang mempunyai akses tulis ke repository tersebut.

Pesan `SELESAI` hanya dicetak setelah `git push` berhasil. Script tidak menggunakan force push dan tidak menghapus riwayat remote. Salinan Git yang dipakai dicetak setelah upload dan berada di `.upload-work/upload_<ID>`.

## Menjalankan eksperimen

Gunakan Google Colab untuk menjalankan notebook secara berurutan, atau Python 3.12 lokal:

```bat
JALANKAN_EKSPERIMEN.cmd
```

Cara terpisah:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_dataset.py
.venv\Scripts\python.exe scripts\run_project.py
```

Source Python mempertahankan seluruh sel kode notebook, dengan pemanggilan ekspor hasil di akhir. Setiap run mendapat direktori baru `hasil_b10/run_<ID>`. Fitur, model, ambang dan prediksi tidak diubah oleh modul ekspor. Angka dapat bergeser pada presisi sangat kecil antarversi pustaka; versi asli dan versi paket dicatat terpisah.

## Inferensi dengan hasil reproduksi yang tersedia

Jalankan contoh berikut dari root repository setelah dependensi terpasang:

```python
from pathlib import Path
import joblib
import pandas as pd
from src.b10_runtime import infer_raw_completed

folder = Path("results/reproduced/latest")
bundle = joblib.load(folder / "B10_model_bundle.joblib")
sensor = pd.read_csv(folder / "demo_raw_sensor.csv", index_col="timestamp", parse_dates=True)
prediction, quality, state = infer_raw_completed(
    sensor, bundle,
    observed_until=sensor.index.max().ceil(bundle["window"])
)
print(prediction)
```

Model reproduksi mengikuti versi `environment.json` pada folder yang sama. Batch sensor harus mencakup jendela lengkap; teruskan state persistensi jika data diproses dalam beberapa batch. Skor bukan probabilitas kerusakan.

## Membuat ulang galeri dari output notebook

```bat
python scripts\build_evidence.py
```

Skrip ini tidak menjalankan training: grafik PNG diekstrak dari output notebook, tabel diambil dari tampilan tersimpan, dan gambar tabel dirender dari nilai tersebut. Tabel yang sudah dibulatkan atau terpotong tidak dinyatakan sebagai ekspor numerik penuh. Tabel lengkap berpresisi penuh berasal dari run reproduksi.

## Pembaruan setelah unggahan awal

Gunakan salinan Git yang dicetak script, atau clone baru:

```bat
git clone https://github.com/x-kzy/UTS-Machine-Learning.git
cd UTS-Machine-Learning
git pull --ff-only origin main
```

Edit file atau jalankan eksperimen. Jika notebook final berubah, simpan notebook dengan output pada `notebooks/UTS_ML_B10_Kelompok5.ipynb`, lalu bangun ulang galeri. Pertahankan hasil lama bila hasil baru masih berupa eksperimen pengembangan. Tinjau perubahan sebelum commit:

```bat
git status
git add README.md docs notebooks src scripts config results reports data .gitignore .gitattributes requirements.txt
git commit -m "Perbarui eksperimen dan dokumentasi B10"
git push origin main
```

Folder `hasil_b10/`, dataset lengkap dan environment virtual diabaikan Git. Tambahkan hanya hasil run yang sudah ditinjau ke `results/`, lalu jelaskan run ID, perubahan parameter, metrik, bukti gambar, dan keputusan pada `docs/EXPERIMENTS.md`. Tidak perlu membuat commit jika tidak ada perubahan.

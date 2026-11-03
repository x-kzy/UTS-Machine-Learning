# Dataset MetroPT-3

Sumber resmi: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset  
DOI: https://doi.org/10.24432/C5VW3R  
Lisensi: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

Dataset dikembangkan oleh Narjes Davari, Bruno Veloso, Rita Ribeiro, dan Joao Gama. Rekaman sensor berasal dari Air Production Unit kompresor kereta. Atribusi sumber harus dipertahankan.

| Berkas | Fungsi |
| --- | --- |
| `sample_sensor_1000.csv` | 1.000 observasi pertama dari CSV sumber; contoh skema, bukan seluruh data eksperimen |
| `Data_Description_Metro.pdf` | Dokumentasi asli dari arsip dataset |
| `laporan_gangguan.csv` | Empat interval laporan yang digunakan untuk evaluasi |
| `MetroPT3(AirCompressor).csv` | CSV lengkap setelah unduhan; sengaja tidak dilacak Git |
| `metropt+3+dataset.zip` | Arsip resmi setelah unduhan; sengaja tidak dilacak Git |

CSV lengkap berukuran 218.300.507 byte. GitHub memblokir file di atas 100 MiB untuk Git biasa; sumber tetap dapat diperoleh melalui tautan resmi dan skrip berikut. Dokumentasi batas: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

```bash
python scripts/download_dataset.py
```

Jika ZIP sudah ada di komputer:

```bat
python scripts\download_dataset.py --source "C:\lokasi\metropt+3+dataset.zip"
```

SHA-256 snapshot yang diperiksa:

```text
ZIP: aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a
CSV: db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24
```

Jika checksum berubah, periksa versi sumber terlebih dahulu. Jangan mengganti hasil laporan dengan angka dari snapshot yang belum diperiksa.

Baris mentah memiliki 17 kolom termasuk timestamp dan ID, dengan 15 sensor. Model tidak menerima ID, timestamp, `y_report`, atau `event_number` sebagai fitur. Label 0 pada pembanding berarti di luar laporan, bukan sehat terverifikasi.

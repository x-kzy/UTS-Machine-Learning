# Hasil eksperimen

## Output final tersimpan

`recorded/` berisi 52 tabel yang diekstrak dari tampilan HTML notebook final dan manifest output. Grafik sumber ada pada `docs/images/`. Presisi tabel mengikuti tampilan tersimpan; sebagian tabel merupakan cuplikan karena pandas membatasi tampilan kolom atau teks. Manifest mencatat `display_excerpt` dan lokasi cell/output.

## Run reproduksi

`reproduced/latest/` berisi satu run source lengkap pada lingkungan paket yang diuji. Empat jumlah jendela serta metrik AP, precision, recall dan F1 pada empat angka desimal cocok dengan output notebook final. Berkas ini meliputi 40 tabel berpresisi penuh, grafik run, model joblib, modul runtime, contoh sensor demo, konfigurasi, log fit, lingkungan, skor/alarm dan SHA-256.

Versi pustaka run reproduksi berbeda dari catatan lingkungan run laporan; keduanya dipertahankan secara eksplisit. Waktu fit merupakan pengukuran pada lingkungan masing-masing, bukan benchmark perangkat keras yang dikontrol.

`scores_and_alarms.csv.gz` menyimpan prediksi validasi dan benchmark setiap model. `B10_model_bundle.joblib` menyimpan pipeline, konfigurasi fitur dan ambang yang digunakan, sehingga inferensi dapat dicoba tanpa fit ulang pada lingkungan paket yang sesuai.

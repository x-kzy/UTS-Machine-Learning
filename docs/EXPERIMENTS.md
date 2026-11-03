# Catatan eksperimen UTS B10

[README](../README.md) · [Galeri](GALLERY.md) · [Tabel lengkap run reproduksi](../results/reproduced/latest/tables/)

Catatan ini merangkum eksperimen yang ada pada notebook final. Log run berisi pemanggilan fit aktual, konfigurasi dan waktu fit; riwayat Git mencatat unggahan serta perubahan repository.

## Protokol yang dipertahankan

Train Februari awal, kalibrasi Februari akhir, validasi Maret–Mei, dan benchmark Juni–akhir. Fitur berupa jendela lima menit yang selesai, kualitas jendela diperiksa, dan satu jendela menjadi embargo antarbatas. Pipeline di-fit tanpa label laporan. AP proxy validasi menjadi kriteria utama pemilihan kandidat. Ambang utama P99 kalibrasi dan persistensi tiga jendela tidak diubah dari hasil sensitivitas.

## Eksperimen dan keputusan

| Eksperimen | Konfigurasi / pertanyaan | Temuan | Bukti |
| --- | --- | --- | --- |
| Baseline | K-Means K=3 setelah imputer, filter konstan dan StandardScaler | Menyediakan skor jarak centroid dan alarm persisten | [Section 6](stages/06.md) |
| Jumlah cluster | K=2,3,4,5,6 pada ruang train dan sampel silhouette yang sama | K=2 silhouette 0,6226; K=3 0,6217. K=3 tetap untuk pembanding terkontrol | [K scan](../results/reproduced/latest/tables/k_scan.csv) |
| Profil cluster | Statistik sensor dalam satuan asli | Ukuran cluster 4.108, 795, dan 79 jendela; nomor cluster bukan label kerusakan | [Profil](../results/reproduced/latest/tables/profile.csv) |
| PCA | StandardScaler lalu PCA minimal 95% varians, K-Means sama | 22 fitur menjadi tujuh komponen, varians 95,692%; AP validasi belum melampaui baseline | [Section 8](stages/08.md) |
| Ensemble | Isolation Forest 200 pohon, max_samples=256 | F1 validasi dan benchmark lebih tinggi pada konfigurasi alarm yang digunakan, tetapi AP validasi lebih rendah | [Validasi](../results/reproduced/latest/tables/validation_metrics.csv) |
| Lima fold temporal | Expanding window; fit dan kalibrasi ulang di dalam fold | Pelampauan berubah antarperiode; alarm persisten nol pada referensi tanpa laporan | [Fold](../results/reproduced/latest/tables/cv_results.csv) |
| Ablation fitur | 22 statistik dasar vs 25 fitur lengkap, model K-Means K=3 | AP sekitar 0,29749 vs 0,29745; penambahan paket tiga fitur belum memperbaiki AP | [Ablation](../results/reproduced/latest/tables/feature_ablation.csv) |
| Sensitivitas | Kuantil 0,975/0,990/0,995 × persistensi 1/3/6 | Sembilan konfigurasi memperlihatkan trade-off alarm dan recall; benchmark tetap P99 dan tiga jendela | [Sensitivitas](../results/reproduced/latest/tables/sensitivity.csv) |
| Benchmark | Seluruh kandidat beku pada jendela yang sama | K-Means AP 0,8685, precision 0,3362, recall 0,9872, F1 0,5016 | [Benchmark](../results/reproduced/latest/tables/test_metrics.csv) |
| Inferensi dan integritas | Sensor mentah → fitur → skor/alarm; serialisasi; state dan jeda | Enam pemeriksaan lulus; demo 12 jendela, satu pelampauan dan nol alarm | [Integritas](../results/reproduced/latest/tables/integrity_checks.csv) |

Angka 22 pada statistik dasar ablation adalah jumlah input sebelum filter. Angka 22 pada pipeline lengkap adalah jumlah fitur tersisa setelah filter dari 25 input; kedua himpunan tersebut berbeda.

## Interpretasi sensitivitas

Pada kuantil 0,99, persistensi tiga ke enam jendela meningkatkan F1 validasi dari sekitar 0,26037 ke 0,26767 dan mengurangi episode di luar laporan dari 0,67650 menjadi 0,29766 per hari ekuivalen terpantau. Temuan ini merupakan analisis trade-off, bukan izin memilih konfigurasi baru dari benchmark yang sama. Recall, delay dan biaya alarm harus dipertimbangkan bersama sebelum eksperimen akhir.

## Bukti hasil dan keterbatasan

Kedua interval benchmark terdeteksi, dengan alarm pertama yang tersedia lima menit setelah awal masing-masing laporan. F03 dan F04 hanya teramati sebagian menurut aturan kualitas; deteksi 2/2 bukan cakupan seluruh durasi. Sebanyak 1.218 jendela alarm berada di luar laporan, tetapi jumlah tersebut tidak identik dengan 30 episode yang seluruhnya tidak bertumpang tindih laporan. Periode benchmark telah diamati pada iterasi sebelumnya.

## Menambah eksperimen berikutnya

1. Nyatakan pertanyaan, fitur/model, ruang parameter dan kriteria pemilihan sebelum fit.
2. Gunakan pipeline dan pemisahan temporal yang sama untuk pembanding.
3. Simpan run ID, parameter, versi, hash dataset, tabel lengkap, grafik, dan keputusan.
4. Tambahkan ringkasan hasil ke dokumen ini, termasuk temuan negatif.
5. Buat commit dengan perubahan aktual dan evaluasi pada periode baru bila tersedia.

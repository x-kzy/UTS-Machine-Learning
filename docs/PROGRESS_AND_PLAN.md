# Progres Milestone 1 (UTS) dan Rencana Pengembangan UAS B10 Kelompok 5

> **Status:** Proyek Milestone 1 telah menghasilkan eksperimen awal. Bagian **Rencana UAS** di bawah merupakan pekerjaan **yang belum diklaim selesai** dan semua yang ada disini masih bersifat rencana dan belum tentu semuanya akan diterapkan pada UAS nantinya, bisa jadi ada yang dipakai/diganti/dihapus. Rujukan utama: *Laporan UTS Milestone 1 Kelompok 5*, BAB V, khususnya subbab 5.1–5.3, halaman tercetak 32–35.

## 1. Dasar keputusan: hasil dan keterbatasan UTS

- Dataset tetap **MetroPT-3 (UCI ID 791)** dari satu Air Production Unit (APU), tanpa target kerusakan pada tiap baris sensor. Empat interval gangguan F01–F04 digunakan sebagai **pembanding evaluasi yang lemah**, bukan label pelatihan.
- Sebanyak **1.516.948** observasi sensor (15 sensor) diolah menjadi **50.276 jendela lima menit yang layak**: 4.982 training, 2.093 kalibrasi, 21.286 validasi, dan 21.915 benchmark.
- Baseline **K-Means (K=3)**, kandidat **PCA + K-Means**, dan **Isolation Forest** dibandingkan memakai split temporal dan ambang P99 pada data kalibrasi, dengan alarm setelah tiga pelampauan berurutan.
- K-Means dipilih karena **AP proxy validasi 0,2974**. Pada benchmark yang telah pernah ditinjau sebelumnya: **AP 0,8685; precision 0,3362; recall 0,9872; F1 0,5016; deteksi 2/2 interval laporan yang teramati**. Terdapat **8 jendela laporan tanpa alarm** dan **1.218 jendela alarm di luar laporan**. Di luar laporan **belum tentu** sehat.
- Isolation Forest mempunyai F1 benchmark lebih tinggi (**0,6036**). Penambahan tiga fitur rekayasa dan PCA **belum terbukti meningkatkan AP** dalam pembandingan UTS. Data uji yang sudah ditinjau sebelumnya tidak boleh dipromosikan sebagai *hold-out baru*.

## 2. Rencana inti UAS berdasarkan BAB V subbab 5.2

| No. | Prioritas | Pelaksanaan yang direncanakan | Bukti penyelesaian yang harus dikumpulkan |
| --- | --- | --- | --- |
| 1 | **Validitas pembanding** | Memeriksa F01–F04, cakupan jendela layak, dan contoh alarm di luar laporan bersama pihak yang memahami mesin **jika akses tersedia**. Jika tidak memungkinkan, dokumentasikan ketidakpastian pembanding dan lakukan analisis sensitivitas batas interval. | Catatan keputusan verifikasi, daftar sampel diperiksa, dan alasan setiap penyesuaian. Tanpa bukti eksternal, jangan menyebut label sudah diverifikasi. |
| 2 | **Rekayasa dan seleksi fitur** | Uji kontribusi fitur secara individual dan menurut kondisi beban; khususnya tiga fitur tambahan serta sinyal digital yang konstan pada referensi. Pertimbangkan fitur *rolling* atau *lag* **hanya bila kausal dan tidak memakai data masa depan**. | Tabel *ablation* temporal, perbedaan AP/F1, beban alarm, serta penjelasan kapan fitur membantu atau tidak. |
| 3 | **Ambang dan biaya alarm** | Tetapkan dahulu batas episode alarm per hari dan toleransi keterlambatan yang ingin diuji; evaluasi kuantil/persistensi berdasarkan validasi, lalu **bekukan konfigurasi sebelum evaluasi akhir**. | Protokol pemilihan konfigurasi; tabel trade-off precision, recall, AP, F1, delay, dan episode alarm per hari. |

## 3. Pengembangan terkait materi setelah UTS — subbab 5.3, Tabel 5.2

| Minggu | Fokus yang direncanakan | Bukti yang ditargetkan | Status |
| --- | --- | --- | --- |
| **9** | Uji **One-Class SVM** sebagai kandidat tambahan, jika sumber daya komputasi memadai, dengan split temporal yang sama. | Tabel model, parameter, waktu komputasi, dan perbandingan yang adil dengan baseline. | Belum dikerjakan pada UTS |
| **10** | *Threshold tuning* pada validasi; interpretasi alarm dan *ablation* per fitur/kelompok sensor. | Tabel trade-off recall, precision, F1, delay, dan episode alarm. | Rencana |
| **12** | Analisis precision-recall; ROC sebagai pendamping; evaluasi per kejadian dan antarperiode. | Grafik, cakupan interval, analisis keterbatasan label dan variasi antarperiode. | Rencana |
| **13** | Pemilihan parameter dan pencatatan eksperimen; *data drift* dan reproduksibilitas. | Konfigurasi, log run, versi model, dan diagnosis distribusi yang dapat diperiksa. | Rencana |
| **14** | Penjelasan alarm berdasarkan sensor; **AutoML opsional** hanya apabila mendukung unsupervised dan validasi temporal. | Contoh alarm, alasan berbasis fitur/sensor, batas interpretasi. | Rencana |
| **15** | Pemilihan konfigurasi akhir dan pembelaan ilmiah dalam presentasi/demonstrasi UAS. | Laporan UAS, notebook/kode, repository, dan demo yang sesuai hasil aktual. | Rencana |

**Batas ruang lingkup:** Proyek tetap memakai *classical machine learning* dan dataset yang sama kecuali ada persetujuan dosen. One-Class SVM adalah kandidat, **bukan kewajiban yang telah dipenuhi**. Materi NLP Minggu 11 bukan prioritas untuk sensor numerik/digital. SMOTE dan Stratified K-Fold acak tidak langsung diterapkan pada data unsupervised temporal. *Probability calibration* berbeda dari kalibrasi kuantil P99 untuk skor anomali.

## 4. Aturan eksperimen UAS agar sebanding dengan UTS

1. Jangan mengubah angka hasil UTS atau menghapus output historis: simpan sebagai **baseline pembanding**.
2. Pisahkan **training**, **kalibrasi**, **validasi/model selection**, dan evaluasi baru bila tersedia. Gunakan pipeline yang mempelajari statistik hanya dari bagian fit yang sesuai.
3. Jangan memakai label laporan dalam fitting estimator tanpa mengubah formulasi masalah secara terbuka dan memperoleh persetujuan yang diperlukan.
4. Untuk setiap percobaan, catat nama metode, parameter, seed, versi pustaka, pembagian waktu, pengukuran, grafik, kesimpulan, dan keterbatasan.
5. Jika tidak tersedia periode baru yang belum pernah disentuh, nyatakan keterbatasan dengan jelas; **jangan mengklaim hasil UAS sebagai benchmark independen**.
6. Jangan menyebut model sudah siap industri, mampu diagnosis penyebab kebocoran, atau mampu memberi peringatan dini sebelum gangguan tanpa pembuktian yang sesuai.

## 5. Organisasi perkembangan repository

- Hasil **UTS** tetap pada `notebooks/`, `src/`, `results/recorded/`, dan `results/reproduced/latest/` (sebagai jejak eksperimen awal).
- Log percobaan **UAS** dicatat melalui pembaruan `docs/EXPERIMENTS.md` atau dokumen tambahan khusus UAS setelah benar-benar dijalankan.
- Gunakan [Checklist UAS](UAS_CHECKLIST.md) untuk mencatat progres yang nyata. Jangan mengubah status menjadi selesai sebelum bukti disimpan.
- [Laporan PDF UTS final](../Laporan_UTS_MILESTONE_1_ML_KELOMPOK_5.pdf) adalah rujukan historis Milestone 1. Laporan UAS nanti dibuat sebagai dokumen berbeda, bukan mengganti hasil UTS.

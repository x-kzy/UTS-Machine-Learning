# Progres UTS dan rencana UAS

## Tahapan UTS yang tersedia

| Tahap | Bukti pada repository |
| --- | --- |
| Perumusan masalah dan sumber | README, pembuka notebook, Bab I laporan |
| Data understanding dan EDA | Section 2–4, galeri, kamus sensor dan statistik |
| Preprocessing dan fitur | Section 5–6, audit jendela dan pipeline |
| Baseline dan kandidat | Section 6–8, K scan dan PCA |
| Validasi dan eksperimen | Section 9, lima fold, ablation dan sensitivitas |
| Benchmark dan diagnosis | Section 10–11, cakupan interval, confusion matrix dan error slicing |
| Inferensi dan reproduksi | Section 12, modul runtime, enam pemeriksaan integritas |
| Ringkasan dan penyerahan | Section 13–14, laporan final dan paket hasil |

Tabel di atas mengikuti urutan kerja yang terdokumentasi pada notebook final. Commit repository mencatat unggahan dan pembaruan aktual sejak repository digunakan.

## Rencana Pengembangan Untuk UAS
- Verifikasi empat interval gangguan dan contoh alarm di luar laporan bersama pihak yang memahami mesin bila akses tersedia.
- Uji kontribusi fitur secara individual dan perilaku sinyal digital yang konstan pada referensi tetapi berubah kemudian.
- Sepakati anggaran episode alarm serta toleransi delay sebelum memilih ambang operasional.
- Bandingkan kandidat lanjutan seperti One-Class SVM dengan protokol temporal yang sesuai.
- Periksa perubahan distribusi dan lakukan evaluasi pada periode/data baru bila tersedia.
- Dokumentasikan penjelasan alarm, keputusan model final, reproduksi dan demonstrasi UAS.

Rencana ini belum dihitung sebagai eksperimen yang selesai. Dataset/masalah tidak diganti hanya untuk mengejar skor.

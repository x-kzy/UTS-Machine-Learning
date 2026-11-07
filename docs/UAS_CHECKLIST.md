# Checklist Pelaksanaan UAS — Proyek B10 Kelompok 5

> **Daftar rencana.** Belum ada tugas UAS yang dinyatakan selesai berdasarkan laporan Milestone 1. Tandai item hanya setelah ada hasil eksperimen/bukti yang dapat diperiksa.

- [ ] **Verifikasi pembanding:** dokumentasikan tinjauan F01–F04, ketidakpastian label, jendela tidak layak, dan pemeriksaan batas interval.
- [ ] **Ablation individual:** bandingkan kontribusi fitur dan kelompok sensor, termasuk fitur digital yang konstan pada training.
- [ ] **Fitur temporal kausal:** uji *rolling* atau *lag* bila relevan, dengan pemeriksaan khusus kebocoran informasi.
- [ ] **Anggaran alarm:** tetapkan target episode/hari dan toleransi *delay* sebelum *threshold tuning*.
- [ ] **Kandidat One-Class SVM:** jalankan bila cocok dengan waktu/komputasi; bandingkan pada protokol temporal yang sama.
- [ ] **Evaluasi lanjut:** AP, precision, recall, F1, episode/hari, *delay*, cakupan interval, dan perbedaan antarperiode.
- [ ] **Drift dan penjelasan alarm:** sertakan bukti fitur/sensor dominan serta batas penafsiran.
- [ ] **Eksperimen dapat direproduksi:** simpan kode, konfigurasi, versi paket, checksum, tabel, dan grafik tiap run yang dipakai untuk keputusan.
- [ ] **Pemilihan final:** bekukan konfigurasi berdasarkan validasi, bukan memilih ulang memakai hasil benchmark yang telah dilihat.
- [ ] **Laporan dan demonstrasi UAS:** tulis kesimpulan berdasarkan eksperimen UAS yang benar-benar dijalankan; kaitkan dengan baseline UTS.

Referensi: [BAB V Laporan UTS](../Laporan_UTS_MILESTONE_1_ML_KELOMPOK_5.pdf) dan [Rencana UAS](PROGRESS_AND_PLAN.md).

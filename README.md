# Laporan Proyek Machine Learning - Azril Bagas

## Domain Proyek

Asuransi adalah salah satu industri yang mengandalkan data dan analitik untuk membuat keputusan yang tepat, termasuk dalam menangani klaim. Membayar klaim yang tidak valid atau overestimating bisa merugikan perusahaan, sementara underestimating bisa merugikan pelanggan. Oleh karena itu, perusahaan asuransi memerlukan model yang akurat untuk memprediksi biaya klaim.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menangani klaim dengan tepat adalah kunci untuk mempertahankan kepercayaan pelanggan dan juga menjaga kestabilan keuangan perusahaan. [Predictive Modeling for Life Insurance](https://www.soa.org/4934c0/globalassets/assets/library/newsletters/product-development-news/2018/february/pro-2018-iss109-stehno-guszcza.pdf)
- Dengan menggunakan model prediktif, perusahaan asuransi dapat mengurangi risiko dan meningkatkan efisiensi operasional. [Towards Explainability of Machine Learning Models in Insurance Pricing]([https://scholar.google.com/](https://arxiv.org/abs/2003.10674)) 

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana mengevaluasi risiko klaim yang diajukan oleh pelanggan?
- Bagaimana menentukan biaya klaim yang paling mungkin?
- Bagaimana mendeteksi klaim yang kemungkinan besar akan menjadi mahal?


### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model yang bisa memprediksi risiko klaim dengan akurat.
- Menentukan perkiraan biaya klaim yang tepat.
- Mengidentifikasi faktor-faktor yang paling mempengaruhi besar atau kecilnya biaya klaim.


**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 
    ### Solution statements
    - Menggunakan algoritma Random Forest, LightGBM, dan CatBoost untuk membangun model prediktif.
    - Melakukan feature engineering dan handling outliers untuk meningkatkan performa model.

## Data Understanding
Dataset berasal dari sebuah kompetisi kaggle yang diselenggarakan oleh The Actuaries Institute of Australia, Institute and Faculty of Actuaries dan the Singapore Actuarial Society. Berjudul **Actuarial loss prediction competition 2020/21**  - [Actuarial loss prediction](https://www.kaggle.com/competitions/actuarial-loss-estimation/data).

### Variabel-variabel pada dataset adalah sebagai berikut:
- ClaimNumber: Pengidentifikasi polisi yang unik
- DateTimeOfAccident: Tanggal dan waktu kejadian kecelakaan
- DateReported: Tanggal kecelakaan dilaporkan
- Age: Umur pekerja
- Gender: Jenis kelamin pekerja
- MaritalStatus: Status perkawinan pekerja. (M) menikah, (S) single, (U) tidak diketahui.
- DependentChildren: Jumlah anak yang menjadi tanggungan
- DependentsOther: Jumlah tanggungan selain anak
- WeeklyWages: Upah mingguan total
- PartTimeFullTime: Biner (P) atau (F)
- HoursWorkedPerWeek: Total jam kerja per minggu
- DaysWorkedPerWeek: Jumlah hari kerja per minggu
- ClaimDescription: Deskripsi klaim dalam teks bebas
- InitialIncurredClaimCost: Estimasi awal dari biaya klaim oleh perusahaan asuransi
- UltimateIncurredClaimCost: Total pembayaran klaim oleh perusahaan asuransi. Ini adalah kolom yang Anda diminta untuk memprediksi dalam set tes.


**Rubrik/Kriteria Tambahan (Opsional)**:

## Teknik Visualisasi Data
EDA dilakukan untuk memahami distribusi data dan hubungan antar variabel. Outliers dan skewness ditemukan pada beberapa fitur.

## Data Preparation
Proses data preparation melibatkan beberapa tahapan seperti handling missing values, encoding categorical features, dan feature engineering. Alasan dilakukan data preparation sebagai berikut :
- Missing Values: Untuk menghindari bias dan meningkatkan akurasi model.
- Encoding: Algoritma ML membutuhkan input numerik.
- Feature Engineering: Untuk ekstrak informasi yang lebih baik dari data.

## Modeling
Tiga model yang digunakan adalah Random Forest, LightGBM, dan CatBoost. Hyperparameter tuning dilakukan untuk meningkatkan performa model.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Kelebihan dan Kekurangan Algoritma
    - Random Forest: Robust terhadap outliers tetapi membutuhkan waktu pelatihan yang lebih lama.
    - LightGBM: Cepat tetapi sensitif terhadap noise.
    - CatBoost: Dapat menangani categorical features tetapi lebih lambat dibanding LightGBM.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Metrik yang digunakan adalah RMSE (Root Mean Squared Error) dan MAE (Mean Absolute Error) dan saya membentuk business metrics sendiri. 
- Hasil Proyek
  Model terbaik adalah model blended yang mengkombinasikan tiga model dengan RMSE 1.23 dan MAE 0.85.
- Metrik Evaluasi
  RMSE memberikan penalti yang lebih besar pada error besar.
  MAE memberikan penalti yang sama pada semua level error.
- Business Metrics
  - Cost-Weighted Error: Menunjukkan dampak finansial dari kesalahan prediksi.
  - Claim Cost Ratio: Menunjukkan efisiensi model dalam memprediksi biaya klaim.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

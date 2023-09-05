# Laporan Proyek Machine Learning - Azril Bagas

## Domain Proyek

Asuransi adalah salah satu industri yang mengandalkan data dan analitik untuk membuat keputusan yang tepat, termasuk dalam menangani klaim. Membayar klaim yang tidak valid atau overestimating bisa merugikan perusahaan, sementara underestimating bisa merugikan pelanggan. Oleh karena itu, perusahaan asuransi memerlukan model yang akurat untuk memprediksi biaya klaim. Menangani klaim dengan tepat adalah kunci untuk mempertahankan kepercayaan pelanggan dan juga menjaga kestabilan keuangan perusahaan. Dengan menggunakan model prediktif, perusahaan asuransi dapat mengurangi risiko dan meningkatkan efisiensi operasional. 

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
- Histogram
![Histogram dari data](/assets/images/eda.png)
- Korelasi antar variabel
![Korelasi](/assets/images/correlation.png)  

## Data Preparation
Proses data preparation melibatkan beberapa tahapan seperti handling missing values, encoding categorical features, dan feature engineering. Alasan dilakukan data preparation sebagai berikut :
- Missing Values: Untuk menghindari bias dan meningkatkan akurasi model.
  ```
  # Checking for missing values in each column
  missing_values = train_data.isnull().sum()
  ```
- Encoding: Algoritma ML membutuhkan input numerik.
  ```
  def encode_categoricals(df):
    encoder = OneHotEncoder(drop='first', sparse=False)
    for col in ['Gender', 'MaritalStatus', 'PartTimeFullTime']:
        one_hot = pd.DataFrame(encoder.fit_transform(df[[col]]))
        one_hot.columns = [col + '_' + str(i) for i in range(one_hot.shape[1])]
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, one_hot], axis=1)
    return df
  ```
- Feature Engineering: Untuk ekstrak informasi yang lebih baik dari data.
  ```
  def add_interaction_features(df):
    ### For now, just add Age multiplied by WeeklyWages
    df['Age_times_WeeklyWages'] = df['Age'] * df['WeeklyWages']
    return df

  def prepare_for_modeling(df):
      drop_cols = ['ClaimNumber', 'ClaimDescription', 'DateTimeOfAccident', 'DateReported', 'Corrected_ClaimDescription']
      df.drop(drop_cols, axis=1, inplace=True)
      return df

    def additional_steps(df):
      # One-hot-encoding for WeekdayOfAccident
      if 'WeekdayOfAccident' in df.columns:
        df = pd.get_dummies(df, columns=['WeekdayOfAccident'], drop_first=True)

    # If ClaimDescription exists, perform TF-IDF Vectorization
    if 'Corrected_ClaimDescription' in df.columns:
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(df['Corrected_ClaimDescription'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df = pd.concat([df, tfidf_df], axis=1)
    
    return df
  ```

## Modeling
Tiga model yang digunakan adalah XGBoost, LightGBM, dan CatBoost. Hyperparameter tuning dilakukan untuk meningkatkan performa model. Model - model tersebut secara singkat dapat dijelaskan sebagai berikut :
- XGboost : Menggunakan pendekatan yang disebut "gradient boosting" pada pohon keputusan. Ini berarti model dibangun dengan cara iteratif, di mana setiap pohon baru berusaha memperbaiki kesalahan yang dibuat oleh kumpulan pohon yang sudah ada. XGBoost menggunakan fungsi tujuan yang dapat disesuaikan dan teknik regularisasi untuk menghindari overfitting. Ini membuatnya sangat fleksibel dan memungkinkan untuk mengoptimalkan berbagai jenis masalah prediksi.
  > Bayangkan kamu sedang bermain permainan tebak-tebakan. Kamu punya banyak teman yang membantu kamu menebak jawabannya. Setiap teman mencoba menebak sedikit, dan kemudian teman berikutnya melihat apa yang sudah ditebak sebelumnya dan mencoba memperbaiki tebakan. Akhirnya, semua tebakan digabungkan untuk mendapatkan jawaban yang paling tepat.
- LightGBM : Menggunakan pohon keputusan seperti XGBoost dan CatBoost, tetapi dengan pendekatan yang disebut "histogram-based learning". Ini mempercepat proses pelatihan dan mengurangi penggunaan memori. LightGBM dirancang untuk efisiensi dan kecepatan, sehingga sangat cocok untuk dataset yang besar atau dimensi yang tinggi. LightGBM juga memiliki dukungan untuk pelatihan menggunakan GPU, yang bisa mempercepat proses lebih lanjut.
  > Bayangkan kamu memiliki banyak kucing yang bisa berbicara, dan setiap kucing ini bisa membantu kamu menjawab pertanyaan dengan melihat apa yang ada di sekitar—seperti gambar atau kata-kata. Mereka semua membantu satu sama lain untuk memberikan jawaban yang paling akurat.
- CatBoost : Juga menggunakan pohon keputusan, tetapi dengan fokus khusus pada fitur kategorikal. Tidak perlu melakukan "one-hot encoding" atau transformasi serupa pada fitur kategorikal karena CatBoost dapat menanganinya secara otomatis. CatBoost memiliki mekanisme regularisasi bawaan yang membantu model menjadi lebih robust terhadap overfitting. Ini memungkinkan model untuk memberikan performa yang baik meskipun dengan data yang lebih sedikit atau lebih banyak noise.
  > Bayangkan kamu membangun menara dari balok kayu. Tapi, alih-alih menumpuk balok satu per satu dari bawah ke atas, kamu mulai dari balok yang paling penting dulu, lalu menambahkan balok lainnya untuk membuat menara semakin kuat dan tinggi. Ini membantu kamu membangun menara lebih cepat dan lebih kuat.

Instalasi ketiga model:
```
pip install xgboost
pip install lightgbm
pip install catboost
```

- Kelebihan dan Kekurangan Algoritma
    - XGBoost
          - *Kelebihan*: Tinggi performa, efisien, fleksibel.
          - *Kekurangan*: Banyak hyperparameter, memerlukan sumber daya besar.
    - LightGBM
          - *Kelebihan*: Cepat dan efisien, rendah penggunaan memori.
          - *Kekurangan*: Overfitting pada data kecil, beberapa parameter kompleks.
    - CatBoost
          - *Kelebihan*: Penanganan fitur kategorikal, robust terhadap overfitting.
          - *Kekurangan*:  Lambat dalam pelatihan, beberapa parameter kompleks.

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
- _Stehno, Chris, et al. "Predictive Modeling for Life Insurance." 2018.[tautan](https://www.soa.org/4934c0/globalassets/assets/library/newsletters/product-development-news/2018/february/pro-2018-iss109-stehno-guszcza.pdf)_
- _Kuo, Kevin, and Daniel Lupton. "Towards explainability of machine learning models in insurance pricing." arXiv preprint arXiv:2003.10674. 2020. [tautan](https://arxiv.org/abs/2003.10674)_


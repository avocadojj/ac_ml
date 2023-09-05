# Laporan Proyek Machine Learning - Azril Bagas

## Domain Proyek

Asuransi adalah salah satu industri yang mengandalkan data dan analitik untuk membuat keputusan yang tepat, termasuk dalam menangani klaim. Membayar klaim yang tidak valid atau overestimating bisa merugikan perusahaan, sementara underestimating bisa merugikan pelanggan. Oleh karena itu, perusahaan asuransi memerlukan model yang akurat untuk memprediksi biaya klaim. Menangani klaim dengan tepat adalah kunci untuk mempertahankan kepercayaan pelanggan dan juga menjaga kestabilan keuangan perusahaan  [1]. Dengan menggunakan model prediktif, perusahaan asuransi dapat mengurangi risiko dan meningkatkan efisiensi operasional [2]. 

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

Model yang telah dikembangkan akan di-deploy menggunakan Flask untuk implementasi lokal, atau menggunakan _App Engine_ pada Google Cloud Platform untuk implementasi skala besar.

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
![Histogram dari data](https://github.com/avocadojj/ac_ml/blob/1c4e2a267000c406553dcafa4e8d3f119a5eee22/images/eda.png)
1. AnakTanggungan: Rentangnya dari 0 hingga 9, dengan sebagian besar klaiman tidak memiliki anak tanggungan.
2. TanggunganLainnya: Rentangnya dari 0 hingga 5, dengan sebagian besar klaiman tidak memiliki tanggungan lain.
3. HariBekerjaPerMinggu: Rentangnya dari 1 hingga 7, dengan sebagian besar klaiman bekerja 5 hari dalam seminggu.
4. BiayaKlaimAwal: Rentangnya dari 1 hingga 2.000.000, menunjukkan variasi biaya klaim awal yang signifikan.
5. BiayaKlaimAkhir: Rentangnya dari sekitar 121,89 hingga 4.027.136, menunjukkan variasi biaya klaim akhir yang signifikan.

- Korelasi antar variabel
![images/correlation.png](https://github.com/avocadojj/ac_ml/blob/ac472bd04081b79fdddec5c21f5a496f150cf955/images/correlation.png) 
1. Fitur seperti "UpahMingguan," "JamBekerjaPerMinggu," "BiayaKlaimAwal," dan "BiayaKlaimAkhir" menunjukkan kecondongan (skewness) yang signifikan dan kemungkinan adanya pencilan (outliers).
2. "JamBekerjaPerMinggu" memiliki beberapa contoh di mana nilai-nilainya jauh lebih tinggi, yang bisa jadi adalah pencilan. Nilai maksimum 640 jam jelas merupakan pencilan, karena jauh lebih tinggi daripada nilai persentil ke-99 yaitu 60 jam. Metode yang saya sarankan untuk memperbaiki pencilan adalah Metode Penutupan (Capping Method).
Kelebihan: Dapat menggunakan pengetahuan domain atau persentil (misalnya, persentil ke-99) untuk membatasi nilai.
Tidak bergantung pada asumsi distribusi normal.
Kekurangan: Memerlukan pemilihan manual untuk nilai batas (cap value), yang mungkin bersifat sewenang-wenang.
Mungkin kehilangan beberapa informasi jika penutupan terlalu agresif.
```
def capping(train_data, test_data, variable):
    # Calculate the 1st and 99th percentiles
    lower_bound = train_data[variable].quantile(0.01)
    upper_bound = train_data[variable].quantile(0.99)

    # Replace the outliers
    train_data[variable] = train_data[variable].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    test_data[variable] = test_data[variable].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

    return train_data, test_data
```

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
Pada preprocess, digunakan TF-IDF untuk membantu pemrosessan teks. TF-IDF menghasilkan sebuah vektor dari teks dokumen yang mencerminkan kepentingan relatif dari kata-kata dalam dokumen tersebut. Terbagi menjadi dua
- _Term Frequency_ (TF): Mengukur seberapa sering sebuah kata muncul dalam dokumen. Ini biasanya dinormalisasi (misalnya, dibagi oleh jumlah total kata dalam dokumen).
- _Inverse Document Frequency_ (IDF): Mengukur seberapa informatif sebuah kata adalah di seluruh kumpulan dokumen (corpus). Kata yang sering muncul di banyak dokumen akan memiliki IDF yang lebih rendah.

Setelah dimasukan ke fungsi tersebut, kemudian data di bagi menjadi 80% untuk pelatihan dan 20% untuk validasi. Adapun permasalahan yaitu terdapat perbedaan kolom yang terjadi akibat _encoding_, hal tersebut diatasi dengan code berikut :
```
# Columns present in train but not in test
missing_in_test = set(processed_train_data.columns) - set(processed_test_data.columns)
missing_in_train = set(processed_test_data.columns) - set(processed_train_data.columns)
for col in missing_in_test - {'UltimateIncurredClaimCost'}:  # We exclude the target variable
    processed_test_data[col] = 0  # or another default value

for col in missing_in_train - {'UltimateIncurredClaimCost'}:  # We exclude the target variable
    processed_train_data[col] = 0  # or another default value

# Re-ordering the columns in the test data to match the order in the training data
processed_test_data = processed_test_data[processed_train_data.drop('UltimateIncurredClaimCost', axis=1).columns]
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

Dari ketiga model dilakukan pencarian parameter optimal menggunakan metode _RandomizedSearchCV_ dengan ketentuan sebagai berikut
```
param_grids = {
    'XGBRegressor': {'learning_rate': [0.02, 0.025], 'max_depth': [5, 6, 7], 'n_estimators': [500], 'colsample_bytree': [0.9], 'subsample': [0.7], 'tree_method': ['hist']},
    'LGBMRegressor': {'learning_rate': [0.02, 0.025], 'n_estimators': [500], 'num_leaves': [30, 40, 50], 'feature_fraction': [0.7]},
    'CatBoostRegressor': {'iterations': [100, 200], 'depth': [6, 8, 10]}
}
```
Dalam pelatihan ini, XGBoost, LightGBM, dan CatBoost adalah algoritma yang sering digunakan dengan variasi parameter untuk meningkatkan performa. Misalnya, dalam XGBoost, kita menyetel learning_rate antara 0,02 dan 0,025 untuk mengontrol seberapa cepat model belajar, dan menggunakan max_depth antara 5 dan 7 untuk membatasi kedalaman pohon keputusan. Jumlah pohon (n_estimators) diatur menjadi 500, dan kita juga mengontrol persentase fitur dan sampel yang digunakan di setiap pohon melalui colsample_bytree dan subsample. Metode pelatihan pohon diatur ke 'hist'. LightGBM, di sisi lain, juga menggunakan learning_rate dan n_estimators yang mirip tetapi menambahkan parameter num_leaves untuk mengontrol jumlah daun maksimum di setiap pohon dan feature_fraction untuk bagian dari fitur yang digunakan. CatBoost juga memiliki pendekatan yang serupa; jumlah iterasi atau pohon diatur antara 100 dan 200, dan kedalaman pohon bisa 6, 8, atau 10. Untuk menghindari output log, logging_level di CatBoost diatur ke 'Silent'. Dengan penyetelan parameter ini, kita berusaha mendapatkan model yang paling optimal untuk data yang kita miliki.
Untuk percobaan, digunakan model gabungan yang terdiri atas 0.4 XGBoost + 0.3 LightGBM + 0.3 CatBoost disebut Blended Model.


## Evaluation
Metrik yang digunakan adalah RMSE (_Root Mean Squared Error_) dan MAE (_Mean Absolute Error_) dan saya membentuk business metrics sendiri. 
RMSE adalah akar kuadrat dari rata-rata perbedaan antara nilai yang diprediksi dan nilai sebenarnya. Ini memberikan ide tentang seberapa besar kesalahan model dalam unit yang sama dengan variabel target. Nilai RMSE yang lebih rendah menunjukkan model yang lebih baik.
MSE adalah rata-rata dari kuadrat perbedaan antara nilai yang diprediksi dan nilai sebenarnya. Sama seperti RMSE, nilai yang lebih rendah menunjukkan model yang lebih baik, tetapi MSE lebih sensitif terhadap outlier karena perbedaan dikuadratkan.

|               | RMSE  | MSE       |
|---------------|-------|-----------|
| XGBoost       | 24619 | 606140463 |
| LightGBM      | 24797 | 614927777 |
| CatBoost      | 24713 | 610742046 |
| Blended Model | 24515 | 600994166 |
Berdasarkan hasil RMSE & MSE dipilih Blended Model yang terbaik karene memiliki nilai terendah dari keseluruhan model. 
- Business Metrics
  - Cost-Weighted Error: Menunjukkan dampak finansial dari kesalahan prediksi.
  - Claim Cost Ratio: Menunjukkan efisiensi model dalam memprediksi biaya klaim.
  Dengan menggunakan dua metrik bisnis tersebut, _blended model_ dilakukan pengujian dengan hasil yakni sebagai berikut pada data validasi
|               | RMSE  | MSE  | Total Overestimation | Total Underestimation | Cost-Weighted Error | Claim Cost Ratio |
|---------------|-------|------|----------------------|-----------------------|---------------------|------------------|
| Blended Model | 24515 | 7407 | 1151423              | -1151423              | 174706947           | 1.008                |
- Total Overestimation dan Underestimation: Model overestimate sekitar 1,151,423 dan underestimate sebesar -1,151,423. Ini menunjukkan bahwa model memiliki kecenderungan yang seimbang antara overestimation dan underestimation, tetapi jumlah absolut dari kedua kesalahan ini adalah aspek yang perlu diperhatikan.
- Cost-Weighted Error: Nilai ini adalah sekitar 174,706,948, yang dihitung berdasarkan biaya yang diterapkan untuk overestimation dan underestimation. Ini adalah indikator penting dari sejauh mana kesalahan prediksi model akan mempengaruhi keuangan perusahaan.
- Claim Cost Ratio: Rasio ini adalah sekitar 1.008, menunjukkan bahwa model memprediksi biaya klaim yang hampir sebanding dengan biaya klaim sebenarnya.
Dengan mempertimbangkan metrik-metrik tersebut, dapat mengatakan bahwa model telah melakukan pekerjaan yang relatif baik dalam memprediksi 'Ultimate Incurred Claim Cost'. Namun, terdapat ruang untuk perbaikan, terutama dalam mengurangi Cost-Weighted Error untuk meminimalkan dampak finansial dari kesalahan prediksi.





**---Ini adalah bagian akhir laporan---**

_Catatan:_
- [1] _Stehno, Chris, et al. "Predictive Modeling for Life Insurance." 2018.[tautan](https://www.soa.org/4934c0/globalassets/assets/library/newsletters/product-development-news/2018/february/pro-2018-iss109-stehno-guszcza.pdf)_
- [2] _Kuo, Kevin, and Daniel Lupton. "Towards explainability of machine learning models in insurance pricing." arXiv preprint arXiv:2003.10674. 2020. [tautan](https://arxiv.org/abs/2003.10674)_


# Laporan Proyek Machine Learning - Nauval Dwi Primadya

## Project Overview

Dalam dunia kuliner yang terus berkembang, konsumen dihadapkan pada pilihan makanan yang semakin banyak, baik di restoran maupun melalui platform layanan pengantaran makanan. Hal ini membuat proses pemilihan makanan seringkali menjadi tugas yang membingungkan. Sistem rekomendasi makanan yang akurat dan relevan dapat membantu pengguna menemukan pilihan yang sesuai dengan preferensi mereka secara efisien. Metode **Content-Based Filtering** dan **Collaborative Filtering** adalah dua teknik utama yang digunakan untuk membangun sistem rekomendasi makanan yang efektif. Content-Based Filtering berfokus pada kesamaan antara makanan berdasarkan atribut seperti jenis, bahan, dan deskripsi makanan, sementara Collaborative Filtering memanfaatkan data interaksi pengguna lain untuk memberikan rekomendasi berdasarkan pola kesamaan preferensi. Kombinasi kedua metode ini dapat menghasilkan rekomendasi yang lebih akurat dan personal, memberikan pengalaman yang lebih baik bagi pengguna dalam memilih makanan yang sesuai dengan selera mereka.

Proyek ini penting karena dapat meningkatkan pengalaman pengguna dalam memilih makanan dengan memberikan saran yang lebih terarah dan relevan, baik berdasarkan kesamaan makanan yang telah dipilih sebelumnya (Content-Based Filtering) maupun preferensi pengguna lain yang serupa (Collaborative Filtering). Dengan menggabungkan kedua teknik ini, sistem rekomendasi dapat mengatasi kekurangan yang ada pada masing-masing metode, seperti masalah **cold-start** pada pengguna atau item baru. Hal ini akan sangat bermanfaat baik bagi konsumen yang ingin menemukan makanan baru sesuai dengan keinginan mereka, maupun bagi bisnis kuliner yang ingin meningkatkan engagement dan penjualan dengan menyarankan pilihan yang lebih personal dan tepat. (Ricci, F., Rokach, L., & Shapira, B., 2015; Zhang, Y., & Chen, J., 2020). 

### Referensi:
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Zhang, Y., & Chen, J. (2020). "A Hybrid Food Recommendation System Based on Collaborative and Content-Based Filtering," *International Journal of Computer Applications*.

## Business Understanding

### Problem Statements

Dalam pengembangan sistem rekomendasi makanan, terdapat beberapa tantangan yang harus diselesaikan untuk memastikan pengalaman pengguna yang lebih baik dan kinerja bisnis yang lebih optimal. Berikut adalah beberapa pernyataan masalah yang relevan:

1. **Pernyataan Masalah 1: Pengguna kesulitan menemukan makanan yang sesuai dengan selera mereka di antara banyaknya pilihan yang tersedia.**
   - Banyaknya pilihan makanan yang tersedia sering kali membuat pengguna merasa kebingungan atau terjebak dalam proses pencarian yang memakan waktu. Tanpa sistem yang membantu menyaring dan memberikan rekomendasi yang relevan, pengguna mungkin merasa frustrasi dan akhirnya memilih untuk tidak melanjutkan pencarian atau bahkan tidak melakukan pemesanan.

2. **Pernyataan Masalah 2: Pengguna yang mencari makanan berdasarkan jenis atau kategori tertentu (seperti makanan Chinese, Thai, atau Mexican) kesulitan menemukan pilihan yang tepat.**
   - Meskipun ada berbagai jenis makanan berdasarkan kategori tertentu, tanpa sistem yang bisa menyaring makanan berdasarkan jenis atau kategori ini, pengguna harus menelusuri banyak pilihan untuk menemukan makanan yang sesuai dengan keinginan mereka. Hal ini bisa memperlambat proses pemilihan makanan dan menurunkan kepuasan pengguna.

3. **Pernyataan Masalah 3: Platform kuliner kesulitan dalam memperkenalkan variasi atau makanan baru kepada pengguna yang sudah terbiasa dengan pilihan tertentu.**
   - Restoran atau platform pengantaran makanan sering kali kesulitan untuk mempromosikan makanan baru atau variasi menu yang belum dikenal oleh pengguna. Tanpa rekomendasi yang cerdas, pengguna mungkin terus memilih makanan yang sama berulang-ulang dan tidak terbuka untuk mencoba variasi baru.

### Goals

Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi yang dapat mengatasi pernyataan masalah di atas dan memberikan manfaat lebih bagi pengguna dan bisnis kuliner. Berikut adalah beberapa tujuan yang ingin dicapai:

1. **Tujuan 1: Membantu pengguna menemukan makanan yang sesuai dengan selera mereka secara lebih efisien dan relevan.**
   - Sistem rekomendasi akan menyaring pilihan makanan berdasarkan input yang diberikan oleh pengguna, seperti nama makanan atau jenis makanan (misalnya Chinese, Thai, atau Mexican), sehingga pengguna dapat dengan cepat menemukan makanan yang sesuai dengan preferensi mereka tanpa harus mencari secara manual.

2. **Tujuan 2: Menyediakan rekomendasi makanan berdasarkan kategori atau jenis tertentu, seperti Chinese, Thai, atau Mexican, untuk memudahkan pencarian.**
   - Dengan memungkinkan pengguna untuk mencari makanan berdasarkan kategori atau jenis yang lebih spesifik, sistem ini akan mempercepat pencarian dan membantu pengguna menemukan makanan yang mereka inginkan dengan lebih mudah dan tanpa kerumitan.

3. **Tujuan 3: Meningkatkan engagement dengan memberikan rekomendasi yang lebih personal dan memperkenalkan variasi atau makanan baru yang relevan.**
   - Sistem ini akan mendorong pengguna untuk mencoba makanan baru berdasarkan jenis atau kategori makanan yang sudah mereka pilih sebelumnya, sehingga meningkatkan variasi pemilihan makanan dan mempromosikan makanan baru atau menu yang jarang dipilih.

### Solution Approach

Untuk mencapai tujuan di atas, ada beberapa pendekatan yang dapat digunakan dalam sistem rekomendasi makanan ini. Berikut adalah solusi yang dapat diterapkan:

#### 1. **Content-Based Filtering (Berbasis Konten)**
   - **Deskripsi Solusi**: Menggunakan informasi tentang makanan seperti nama, jenis makanan (Chinese, Thai, Mexican), dan deskripsi untuk menemukan makanan yang mirip dengan makanan yang telah dipilih oleh pengguna. Sistem ini akan menghitung kesamaan antara makanan berdasarkan fitur-fitur tersebut.
   - **Langkah Implementasi**:
     1. Mengekstraksi fitur dari dataset, termasuk nama makanan, jenis, dan deskripsi.
     2. Menghitung kesamaan antar makanan menggunakan teknik **Cosine Similarity** atau **TF-IDF**.
     3. Memberikan rekomendasi berdasarkan makanan dengan kemiripan tertinggi dengan yang sudah dipilih atau dicari oleh pengguna.

#### 2. **Collaborative Filtering (Berbasis Pengguna)**
   - **Deskripsi Solusi**: Menggunakan data interaksi pengguna, seperti makanan yang telah dipilih sebelumnya atau rating yang diberikan, untuk memberikan rekomendasi berdasarkan pola preferensi pengguna lain yang serupa.
   - **Langkah Implementasi**:
     1. Membuat **user-item interaction matrix** yang menunjukkan interaksi pengguna dengan makanan.
     2. Menggunakan algoritma seperti **k-nearest neighbors (KNN)** atau **matrix factorization** untuk menemukan kesamaan antar pengguna dan memberikan rekomendasi berdasarkan makanan yang dipilih oleh pengguna lain dengan preferensi serupa.

### Kesimpulan

Dengan pendekatan **Content-Based Filtering** dan **Collaborative Filtering**, sistem rekomendasi ini dapat membantu pengguna menemukan makanan yang sesuai dengan preferensi mereka, baik berdasarkan jenis atau kategori makanan yang mereka pilih (seperti Chinese, Thai, atau Mexican), maupun berdasarkan pola perilaku pengguna lain. Kedua metode ini memungkinkan sistem untuk memberikan rekomendasi yang lebih personal dan relevan, sehingga meningkatkan kepuasan pengguna dan membantu bisnis kuliner dalam memperkenalkan variasi makanan baru atau menu yang jarang dipilih.

## Data Understanding

Dataset yang digunakan dalam proyek ini terdiri dari dua tabel: **data makanan** dan **rating makanan**. Dataset makanan berisi informasi tentang berbagai hidangan, termasuk nama makanan, kategori (seperti "Healthy Food", "Snack", dll.), apakah makanan tersebut vegan atau non-vegan, serta deskripsi bahan-bahan yang digunakan. Dataset rating berisi penilaian yang diberikan oleh pengguna terhadap makanan, yang mencakup ID pengguna, ID makanan, dan nilai rating (1-10). Dataset ini dapat diunduh melalui [Kaggle Food Dataset](https://www.kaggle.com/datasets).

### Deskripsi Variabel

1. **Tabel Makanan (`data`)**:
   - **`food_id`**: ID unik untuk setiap makanan (integer).
   - **`name`**: Nama makanan (string).
   - **`c_type`**: Kategori makanan, seperti "Healthy Food", "Snack", "Dessert" (string).
   - **`veg_non`**: Vegetarian atau non-vegetarian ("veg" atau "non-veg") (string).
   - **`describe`**: Deskripsi bahan-bahan makanan (string).

2. **Tabel Rating (`rating`)**:
   - **`user_id`**: ID unik pengguna (angka desimal).
   - **`food_id`**: ID unik makanan yang diberi rating (angka desimal).
   - **`rating`**: Nilai rating yang diberikan oleh pengguna (1-10, angka desimal).

### Exploratory Data Analysis (EDA)

1. **Jumlah Jenis Makanan (`c_type`)**:
   - Menghitung frekuensi kemunculan tiap kategori makanan dan memvisualisasikannya dengan **bar plot** untuk mengetahui distribusi jenis makanan.
   
   ```python
   plt.figure(figsize=(10, 5))
   sns.countplot(data=data, x=data['c_type'])
   plt.xticks(rotation=90)
   plt.show()
   ```

2. **Distribusi Vegan vs Non-Vegan (`veg_non`)**:
   - Menampilkan distribusi makanan vegan dan non-vegan dengan **bar plot** untuk mengetahui proporsi makanan berbasis tanaman atau hewan.
   
   ```python
   plt.figure(figsize=(5, 3))
   sns.countplot(data=data, x=data['veg_non'])
   plt.xticks(rotation=90)
   plt.show()
   ```

3. **Distribusi Rating**:
   - Menghitung jumlah kemunculan setiap rating dan menampilkannya dalam **bar plot** untuk melihat bagaimana rating terdistribusi.
   
   ```python
   plt.figure(figsize=(5, 3))
   sns.countplot(data=rating, x=rating['rating'])
   plt.show()
   ```

4. **Jumlah Data**:
   - Menghitung jumlah kategori unik dalam **`c_type`** dan **`veg_non`** untuk mendapatkan wawasan tentang keragaman kategori makanan.
   
   ```python
   print('Jumlah Type: ', len(data['c_type'].unique()))
   print('Jumlah Data: ', len(data['veg_non'].unique()))
   ```

### Kesimpulan

Melalui EDA, kita mendapatkan gambaran tentang distribusi jenis makanan dan rating pengguna, serta proporsi makanan vegan/non-vegan. Ini membantu dalam merancang sistem rekomendasi yang sesuai dengan preferensi pengguna dan memberikan wawasan tentang kategori makanan yang paling banyak atau kurang populer.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

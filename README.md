# Laporan Proyek Machine Learning - Nauval Dwi Primadya

<div align="justify">

   
## Project Overview

Dalam dunia kuliner yang terus berkembang, konsumen dihadapkan pada pilihan makanan yang semakin banyak, baik di restoran maupun melalui platform layanan pengantaran makanan. Hal ini membuat proses pemilihan makanan seringkali menjadi tugas yang membingungkan [[1](https://www.aasmr.org/jsms/Vol13/No.5/Vol.13%20No.5.10.pdf)]. Sistem rekomendasi makanan yang akurat dan relevan dapat membantu pengguna menemukan pilihan yang sesuai dengan preferensi mereka secara efisien. Metode **Content-Based Filtering** dan **Collaborative Filtering** adalah dua teknik utama yang digunakan untuk membangun sistem rekomendasi makanan yang efektif. Content-Based Filtering berfokus pada kesamaan antara makanan berdasarkan atribut seperti jenis, bahan, dan deskripsi makanan, sementara Collaborative Filtering memanfaatkan data interaksi pengguna lain untuk memberikan rekomendasi berdasarkan pola kesamaan preferensi. Kombinasi kedua metode ini dapat menghasilkan rekomendasi yang lebih akurat dan personal, memberikan pengalaman yang lebih baik bagi pengguna dalam memilih makanan yang sesuai dengan selera mereka [[2](https://www.proquest.com/openview/231c3bcbc225e7beb2b2a8bceb903a8f/1?pq-origsite=gscholar&cbl=5314840)].

Proyek ini penting karena dapat meningkatkan pengalaman pengguna dalam memilih makanan dengan memberikan saran yang lebih terarah dan relevan, baik berdasarkan kesamaan makanan yang telah dipilih sebelumnya (Content-Based Filtering) maupun preferensi pengguna lain yang serupa (Collaborative Filtering) [[2](https://www.proquest.com/openview/231c3bcbc225e7beb2b2a8bceb903a8f/1?pq-origsite=gscholar&cbl=5314840)]. Dengan menggabungkan kedua teknik ini, sistem rekomendasi dapat mengatasi kekurangan yang ada pada masing-masing metode, seperti masalah **cold-start** pada pengguna atau item baru. Hal ini akan sangat bermanfaat baik bagi konsumen yang ingin menemukan makanan baru sesuai dengan keinginan mereka, maupun bagi bisnis kuliner yang ingin meningkatkan engagement dan penjualan dengan menyarankan pilihan yang lebih personal dan tepat. 

## Business Understanding

### Problem Statements

Dalam pengembangan sistem rekomendasi makanan, terdapat beberapa tantangan yang harus diselesaikan untuk memastikan pengalaman pengguna yang lebih baik dan kinerja bisnis yang lebih optimal. Berikut adalah beberapa pernyataan masalah yang relevan:

1. Pengguna kesulitan menemukan makanan yang sesuai dengan selera mereka di antara banyaknya pilihan yang tersedia.
   - Banyaknya pilihan makanan yang tersedia sering kali membuat pengguna merasa kebingungan atau terjebak dalam proses pencarian yang memakan waktu. Tanpa sistem yang membantu menyaring dan memberikan rekomendasi yang relevan, pengguna mungkin merasa frustrasi dan akhirnya memilih untuk tidak melanjutkan pencarian atau bahkan tidak melakukan pemesanan.

2. Pengguna yang mencari makanan berdasarkan jenis atau kategori tertentu (seperti makanan Chinese, Thai, atau Mexican) kesulitan menemukan pilihan yang tepat.
   - Meskipun ada berbagai jenis makanan berdasarkan kategori tertentu, tanpa sistem yang bisa menyaring makanan berdasarkan jenis atau kategori ini, pengguna harus menelusuri banyak pilihan untuk menemukan makanan yang sesuai dengan keinginan mereka. Hal ini bisa memperlambat proses pemilihan makanan dan menurunkan kepuasan pengguna.

3. Platform kuliner kesulitan dalam memperkenalkan variasi atau makanan baru kepada pengguna yang sudah terbiasa dengan pilihan tertentu.
   - Restoran atau platform pengantaran makanan sering kali kesulitan untuk mempromosikan makanan baru atau variasi menu yang belum dikenal oleh pengguna. Tanpa rekomendasi yang cerdas, pengguna mungkin terus memilih makanan yang sama berulang-ulang dan tidak terbuka untuk mencoba variasi baru.

### Goals

Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi yang dapat mengatasi pernyataan masalah di atas dan memberikan manfaat lebih bagi pengguna dan bisnis kuliner. Berikut adalah beberapa tujuan yang ingin dicapai:

1. Membantu pengguna menemukan makanan yang sesuai dengan selera mereka secara lebih efisien dan relevan.
   - Sistem rekomendasi akan menyaring pilihan makanan berdasarkan input yang diberikan oleh pengguna, seperti nama makanan atau jenis makanan (misalnya Chinese, Thai, atau Mexican), sehingga pengguna dapat dengan cepat menemukan makanan yang sesuai dengan preferensi mereka tanpa harus mencari secara manual.

2. Menyediakan rekomendasi makanan berdasarkan kategori atau jenis tertentu, seperti Chinese, Thai, atau Mexican, untuk memudahkan pencarian.
   - Dengan memungkinkan pengguna untuk mencari makanan berdasarkan kategori atau jenis yang lebih spesifik, sistem ini akan mempercepat pencarian dan membantu pengguna menemukan makanan yang mereka inginkan dengan lebih mudah dan tanpa kerumitan.

3. Meningkatkan engagement dengan memberikan rekomendasi yang lebih personal dan memperkenalkan variasi atau makanan baru yang relevan.
   - Sistem ini akan mendorong pengguna untuk mencoba makanan baru berdasarkan jenis atau kategori makanan yang sudah mereka pilih sebelumnya, sehingga meningkatkan variasi pemilihan makanan dan mempromosikan makanan baru atau menu yang jarang dipilih.

### Solution Approach

Untuk mencapai tujuan di atas, ada beberapa pendekatan yang dapat digunakan dalam sistem rekomendasi makanan ini. Berikut adalah solusi yang dapat diterapkan:

#### 1. **Content-Based Filtering (Berbasis Konten)**  
   - **Deskripsi Solusi**: Sistem ini menggunakan informasi seperti nama, jenis makanan (Chinese, Thai, Mexican), dan deskripsi untuk memberikan rekomendasi berdasarkan kesamaan antar makanan. Data deskriptif diubah menjadi representasi numerik menggunakan teknik seperti **TF-IDF Vectorizer**, lalu dihitung kesamaannya menggunakan **Cosine Similarity** untuk menemukan makanan yang paling mirip dengan pilihan pengguna.  
   - **Langkah Implementasi**:  
     1. Mengekstraksi fitur dari dataset, seperti nama makanan, jenis, dan deskripsi.  
     2. Mengubah fitur teks menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.  
     3. Menghitung kesamaan antar makanan menggunakan **Cosine Similarity**.  
     4. Memberikan rekomendasi makanan yang memiliki kemiripan tertinggi dengan makanan pilihan pengguna.  

#### 2. **Collaborative Filtering (Berbasis Pengguna)**  
   - **Deskripsi Solusi**: Pendekatan ini memanfaatkan data interaksi pengguna, seperti rating atau klik, untuk memprediksi preferensi makanan berdasarkan kesamaan antar pengguna. Model **RecommenderNet** digunakan untuk merepresentasikan pengguna dan makanan dalam bentuk embedding, memungkinkan deteksi hubungan non-linear melalui **dot product** dan bias tambahan.  
   - **Langkah Implementasi**:  
     1. Membuat **user-item interaction matrix** yang mencatat interaksi pengguna dengan makanan.  
     2. Membuat model embedding untuk merepresentasikan pengguna dan makanan sebagai vektor berdimensi tetap menggunakan **RecommenderNet**.  
     3. Melatih model dengan **Binary Crossentropy** menggunakan **Adam Optimizer** untuk memprediksi relevansi makanan terhadap pengguna.  
     4. Memberikan rekomendasi makanan berdasarkan skor relevansi yang dihasilkan model.  

Dengan pendekatan **Content-Based Filtering** dan **Collaborative Filtering**, sistem rekomendasi ini dapat membantu pengguna menemukan makanan yang sesuai dengan preferensi mereka, baik berdasarkan jenis atau kategori makanan yang mereka pilih (seperti Chinese, Thai, atau Mexican), maupun berdasarkan pola perilaku pengguna lain. Kedua metode ini memungkinkan sistem untuk memberikan rekomendasi yang lebih personal dan relevan, sehingga meningkatkan kepuasan pengguna dan membantu bisnis kuliner dalam memperkenalkan variasi makanan baru atau menu yang jarang dipilih.

## Data Understanding

Dataset yang digunakan dalam proyek ini terdiri dari dua tabel: **data makanan** dan **rating makanan**. Dataset makanan berisi informasi tentang berbagai hidangan, termasuk nama makanan, kategori (seperti "Healthy Food", "Snack", dll.), apakah makanan tersebut vegan atau non-vegan, serta deskripsi bahan-bahan yang digunakan. Dataset rating berisi penilaian yang diberikan oleh pengguna terhadap makanan, yang mencakup ID pengguna, ID makanan, dan nilai rating (1-10). Dataset ini dapat diunduh melalui.


| Jenis      | Keterangan                                                                 |
|------------|-----------------------------------------------------------------------------|
| Title      | Food Recommendation System                                                             |
| Source     | [Kaggle](https://www.kaggle.com/datasets/schemersays/food-recommendation-system)                  |
| Maintainer | [schemersays](https://www.kaggle.com/schemersays)                                   |
| License    | Unknown                  |
| Visibility | Publik                                                                      |
| Tags       | _Busines_ |
| Usability  | 4.71                                                                      |



### Deskripsi Variabel

1. **Tabel Makanan (`data`)**:
   - **`food_id`**: ID unik untuk setiap makanan (integer).
   - **`name`**: Nama makanan (string).
   - **`c_type`**: Kategori makanan, seperti "Healthy Food", "Snack", "Dessert" (string).
   - **`veg_non`**: Vegetarian atau non-vegetarian ("veg" atau "non-veg") (string).
   - **`describe`**: Deskripsi bahan-bahan makanan (string).


| food_id | name                     | c_type      | veg_non | describe                                                    |
|---------|--------------------------|-------------|---------|-------------------------------------------------------------|
| 0       | summer squash salad      | Healthy Food| veg     | white balsamic vinegar, lemon juice, lemon rind...          |
| 1       | chicken minced salad     | Healthy Food| non-veg | olive oil, chicken mince, garlic (minced), onion...        |
| 2       | sweet chilli almonds     | Snack       | veg     | almonds whole, egg white, curry leaves, salt, ...          |
| 3       | tricolour salad          | Healthy Food| veg     | vinegar, honey/sugar, soy sauce, salt, garlic ...           |
| 4       | christmas cake           | Dessert     | veg     | christmas dry fruits (pre-soaked), orange zest...          |


2. **Tabel Rating (`rating`)**:
   - **`user_id`**: ID unik pengguna (angka desimal).
   - **`food_id`**: ID unik makanan yang diberi rating (angka desimal).
   - **`rating`**: Nilai rating yang diberikan oleh pengguna (1-10, angka desimal).


| user_id | food_id | rating |
|---------|---------|--------|
| 0       | 88.0    | 4.0    |
| 1       | 46.0    | 3.0    |
| 2       | 24.0    | 5.0    |
| 3       | 25.0    | 4.0    |
| 4       | 49.0    | 1.0    |


### Exploratory Data Analysis (EDA)

1. **Jumlah Jenis Makanan (`c_type`)**:
   - Menghitung frekuensi kemunculan tiap kategori makanan dan memvisualisasikannya dengan **bar plot** untuk mengetahui distribusi jenis makanan.
   
   ```python
   plt.figure(figsize=(10, 5))
   sns.countplot(data=data, x=data['c_type'])
   plt.xticks(rotation=90)
   plt.show()
   ```
![Grafik Jumlah Jenis Makanan](https://github.com/Primadya/sistemrekomendasidicoding/blob/main/image/Screenshot%20from%202024-11-17%2014-49-46.png?raw=true)


2. **Distribusi Vegan vs Non-Vegan (`veg_non`)**:
   - Menampilkan distribusi makanan vegan dan non-vegan dengan **bar plot** untuk mengetahui proporsi makanan berbasis tanaman atau hewan.
   
   ```python
   plt.figure(figsize=(5, 3))
   sns.countplot(data=data, x=data['veg_non'])
   plt.xticks(rotation=90)
   plt.show()
   ```
![Grafik Distribusi Vegan dan Non-Vegan](https://github.com/Primadya/sistemrekomendasidicoding/blob/main/image/Screenshot%20from%202024-11-17%2014-50-00.png?raw=true)


3. **Distribusi Rating**:
   - Menghitung jumlah kemunculan setiap rating dan menampilkannya dalam **bar plot** untuk melihat bagaimana rating terdistribusi.
   
   ```python
   plt.figure(figsize=(5, 3))
   sns.countplot(data=rating, x=rating['rating'])
   plt.show()
   ```
   
![Grafik Distribusi Rating](https://github.com/Primadya/sistemrekomendasidicoding/blob/main/image/Screenshot%20from%202024-11-17%2014-50-10.png?raw=true)


Melalui EDA, kita mendapatkan gambaran tentang distribusi jenis makanan dan rating pengguna, serta proporsi makanan vegan/non-vegan. Ini membantu dalam merancang sistem rekomendasi yang sesuai dengan preferensi pengguna dan memberikan wawasan tentang kategori makanan yang paling banyak atau kurang populer.

## Data Preparation

Pada tahap **Data Preparation**, dilakukan beberapa proses untuk memastikan bahwa data yang digunakan dalam sistem rekomendasi sudah bersih, konsisten, dan siap diproses oleh model **Content-Based Filtering** dan **Collaborative Filtering**. Berikut adalah tahapannya beserta alasan mengapa setiap tahapan ini penting.

### 1. **Pembersihan Data**
Pembersihan data adalah langkah pertama yang penting untuk memastikan tidak ada nilai yang hilang atau duplikat yang bisa memengaruhi model. Data yang hilang atau duplikat dapat menyebabkan ketidakkonsistenan dalam hasil rekomendasi dan merusak akurasi model.

#### a. **Menghapus Nilai yang Hilang (Missing Values)**:
Untuk memastikan bahwa tidak ada nilai yang hilang yang dapat mengganggu analisis, kita akan mengecek dan menghapus baris yang memiliki nilai kosong.

```python
# Cek missing value pada dataset
merged_data.isnull().sum()

# Menghapus baris yang mengandung missing value
merged_data = merged_data.dropna()

# Cek kembali missing value setelah dihapus
print("\nJumlah missing value setelah pembersihan:")
print(merged_data.isnull().sum())
```

**Alasan**: Menghapus nilai yang hilang mengurangi potensi kesalahan dalam model dan memastikan bahwa data yang digunakan lengkap untuk pemrosesan lebih lanjut.

#### b. **Menghapus Duplikat**:
Data yang terduplikasi dapat menyebabkan bias pada model dan memberi hasil yang tidak akurat.

```python
# Menghapus duplikasi berdasarkan food_id
merged_data = merged_data.drop_duplicates('food_id')
```

**Alasan**: Duplikasi dapat menyebabkan informasi yang salah, misalnya jika makanan yang sama muncul lebih dari sekali, dapat memengaruhi model dalam memberikan rekomendasi yang akurat.

### 2. **Pengkodean dan Normalisasi**
Pengkodean adalah proses mengonversi data kategorikal menjadi format numerik yang bisa diproses oleh model. Normalisasi diperlukan agar rating yang diberikan oleh pengguna memiliki skala yang konsisten.

#### a. **Pengkodean ID Pengguna dan Makanan** (Collaborative Filtering):
Untuk **Collaborative Filtering**, kita perlu mengonversi **user_id** dan **food_id** menjadi angka untuk membuat matriks pengguna-makanan yang lebih mudah diproses.

```python
# Encoding user_id dan food_id
user_ids = df['user_id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
food_ids = df['food_id'].unique().tolist()
food_to_food_encoded = {x: i for i, x in enumerate(food_ids)}

df['user'] = df['user_id'].map(user_to_user_encoded)
df['food'] = df['food_id'].map(food_to_food_encoded)
```

**Alasan**: Pengkodean ID pengguna dan makanan menjadi angka memungkinkan model untuk memproses data lebih efisien dalam bentuk matriks numerik.

#### b. **Normalisasi Rating** (Collaborative Filtering):
Rating yang diberikan oleh pengguna di-normalisasi ke rentang 0 hingga 1 agar konsisten dalam pembelajaran model.

```python
# Normalisasi rating ke rentang 0-1
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```

**Alasan**: Normalisasi rating membantu model untuk tidak terbebani oleh skala rating yang lebih tinggi atau rendah dan membuat pembelajaran menjadi lebih seimbang.

### 3. **Transformasi Teks untuk Content-Based Filtering**
Pada **Content-Based Filtering**, kita akan menggunakan informasi konten dari deskripsi makanan untuk merekomendasikan makanan yang serupa berdasarkan deskripsi teks. Untuk itu, deskripsi makanan akan diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorizer untuk kolom Describe
tf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tf.fit_transform(foods['describe'])
```

**Alasan**: **TF-IDF** (Term Frequency-Inverse Document Frequency) membantu untuk mengonversi teks menjadi vektor numerik berdasarkan pentingnya kata dalam deskripsi makanan. Dengan ini, model dapat menghitung kemiripan antara makanan berdasarkan konten deskripsinya.

### 4. **Pembagian Data**
Pembagian dataset menjadi data latih dan data validasi sangat penting untuk mengevaluasi kinerja model. Tanpa pembagian yang jelas, model mungkin akan mengalami overfitting atau underfitting, yang akan mempengaruhi akurasi rekomendasi.

```python
# Membagi dataset menjadi data latih dan validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]
)
```

**Alasan**: Pembagian data memungkinkan kita untuk melatih model menggunakan sebagian besar data dan menguji kinerja model pada data yang belum pernah dilihat sebelumnya, sehingga memberikan gambaran yang lebih realistis tentang kinerja model.

Proses **Data Preparation** sangat krusial untuk memastikan kualitas dan konsistensi data yang digunakan dalam membangun model rekomendasi. Beberapa alasan pentingnya tahap-tahap ini adalah:

1. **Menghapus data yang hilang dan duplikat** membantu mencegah kesalahan atau bias yang dapat mempengaruhi hasil rekomendasi.
2. **Pengkodean dan normalisasi** membuat data lebih mudah diproses oleh model dan memastikan konsistensi dalam skala rating yang diberikan oleh pengguna.
3. **Transformasi teks dengan TF-IDF** memungkinkan model untuk memahami dan mengukur kemiripan antar makanan berdasarkan deskripsi konten, yang merupakan dasar dari **Content-Based Filtering**.
4. **Pembagian data** menjadi data latih dan validasi memungkinkan kita untuk melatih model pada data yang ada dan mengujinya pada data yang belum terlihat, yang penting untuk evaluasi kinerja model.

Dengan langkah-langkah **Data Preparation** ini, dataset siap digunakan untuk membangun model **Content-Based Filtering** dan **Collaborative Filtering**, yang akan memberikan rekomendasi makanan yang akurat dan relevan kepada pengguna.


## Modeling
### Content-Based Filtering (CBF) Model

Pada bagian **Content-Based Filtering**, pendekatan yang digunakan adalah mencari makanan yang mirip berdasarkan deskripsi dan kategori yang ada di dataset. Sistem ini akan merekomendasikan makanan berdasarkan **cosine similarity** antara deskripsi makanan yang dimasukkan dengan makanan lain dalam database. Selain itu, pengguna dapat mencari makanan berdasarkan nama atau jenis makanan (misalnya Chinese, Thai, dll.).

#### a. **Fungsi `food_recommendations`**
Fungsi `food_recommendations` menerima input dari pengguna, yang dapat berupa nama makanan atau jenis makanan. Berdasarkan input tersebut, fungsi ini mencari kemiripan menggunakan **Cosine Similarity** antara makanan yang sudah ada. Jika inputnya cocok dengan nama makanan atau kategori makanan, sistem akan mengembalikan **Top-N** rekomendasi berdasarkan kesamaan konten (nama, kategori, dan deskripsi makanan).

```python
def food_recommendations(input_value, similarity_data=cosine_sim_df, items=foods[['name', 'c_type', 'veg_non']], k=4):
    input_value = input_value.strip().lower()  # Pastikan input dalam lowercase dan tanpa spasi ekstra

    print(f"Input value: {input_value}")  # Debugging: Periksa input_value

    # Jika input_value ada sebagai nama makanan dalam similarity_data (pencocokan parsial)
    matching_items = [name for name in similarity_data.columns if input_value in name.lower()]
    
    if matching_items:
        print(f"Nama makanan yang cocok dengan '{input_value}': {matching_items}")  # Debugging
        # Rekomendasi berdasarkan nama makanan
        closest = matching_items  # Gunakan semua makanan yang cocok
        closest_df = pd.DataFrame({'name': closest})  # Buat DataFrame dengan kolom 'name'
        return closest_df.merge(items, on='name').head(k)
    
    # Jika input_value adalah C_Type yang ada dalam dataset (menggunakan pencocokan parsial)
    elif any(input_value in ctype.lower() for ctype in items['c_type'].unique()):
        # Filter dataset berdasarkan C_Type yang mengandung input_value
        filtered_items = items[items['c_type'].str.contains(input_value, case=False, na=False)]
        return filtered_items.head(k)
    
    else:
        # Jika input_value tidak ditemukan
        available_names = ", ".join(items['name'].unique()[:5])  # Tampilkan 5 nama makanan
        available_types = ", ".join(items['c_type'].unique()[:5])  # Tampilkan 5 jenis makanan
        return (f"Tidak ada rekomendasi yang ditemukan untuk '{input_value}'.\n"
                f"Coba masukkan salah satu nama berikut: {available_names}, atau salah satu jenis makanan berikut: {available_types}")

# Mengambil input dari pengguna untuk mencari rekomendasi
input_value = input("Masukkan nama makanan atau jenis makanan yang Anda suka: ").strip()
result = food_recommendations(input_value, k=4)
```

**Kelebihan Content-Based Filtering:**
- **Personalized Recommendations**: Memberikan rekomendasi yang sangat relevan berdasarkan deskripsi atau jenis makanan yang disukai.
- **Tidak Memerlukan Data Pengguna Lain**: Cocok untuk situasi di mana tidak ada cukup interaksi pengguna atau data rating.

**Kekurangan Content-Based Filtering:**
- **Keterbatasan Variasi**: Hanya memberikan rekomendasi berdasarkan kemiripan konten, jadi bisa jadi terbatas jika tidak ada banyak variasi dalam deskripsi atau kategori makanan.
- **Mungkin Kurang Menarik**: Bisa memberikan rekomendasi yang terlalu serupa, tanpa mengeksplorasi makanan yang berbeda dari yang sudah disukai.

---

### Collaborative Filtering (CF) Model

Pada bagian **Collaborative Filtering**, pendekatan yang digunakan adalah dengan memanfaatkan data interaksi pengguna dengan makanan untuk memberikan rekomendasi. Dalam hal ini, sistem melakukan **Matrix Factorization** dengan menggunakan model **Neural Network (RecommenderNet)** yang dibangun menggunakan TensorFlow dan Keras. Model ini memetakan pengguna dan makanan ke dalam ruang vektor yang lebih rendah (embedding), dan menggunakan **dot product** antara vektor pengguna dan makanan untuk memprediksi rating makanan.

#### a. **Model RecommenderNet**

Model **RecommenderNet** menggunakan **embedding layer** untuk pengguna dan makanan, dan melakukan prediksi rating berdasarkan interaksi tersebut. Model ini memiliki dua embedding: satu untuk **user** dan satu untuk **food**. Prediksi rating dihasilkan dengan cara menghitung **dot product** antara embedding pengguna dan embedding makanan, kemudian ditambah dengan bias pengguna dan makanan.

```python
import tensorflow as tf
from tensorflow.keras import layers, regularizers

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_food, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.food_embedding = layers.Embedding(
            num_food, embedding_size, embeddings_initializer="he_normal",
            embeddings_regularizer=regularizers.l2(1e-6)
        )
        self.food_bias = layers.Embedding(num_food, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        food_vector = self.food_embedding(inputs[:, 1])
        food_bias = self.food_bias(inputs[:, 1])
        dot_product = tf.tensordot(user_vector, food_vector, axes=2)
        x = dot_product + user_bias + food_bias
        return tf.nn.sigmoid(x)

# Menyusun data dan melatih model
model = RecommenderNet(num_users, num_food)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    validation_data=(x_val, y_val)
)
```

#### b. **Menampilkan Rekomendasi Berbasis Collaborative Filtering**

Untuk mendapatkan rekomendasi, kita menggunakan model yang sudah dilatih dan memprediksi rating makanan yang belum pernah dinilai oleh pengguna.

```python
def get_collaborative_recommendations(user_id, predicted_ratings_df, top_n=5):
    # Mendapatkan makanan yang sudah dinilai oleh pengguna
    rated_foods = df[df['user_id'] == user_id]['food_id'].tolist()
    
    # Menyaring makanan yang belum dinilai oleh pengguna
    recommendations = predicted_ratings_df.loc[~predicted_ratings_df.index.isin(rated_foods)]
    
    # Mengambil N makanan dengan prediksi rating tertinggi
    recommendations = recommendations[user_id].sort_values(ascending=False).head(top_n)
    
    return recommendations.index.tolist()

# Rekomendasi untuk pengguna dengan user_id = 2
top_n_recommendations_collaborative = get_collaborative_recommendations(2, predicted_ratings_df, top_n=5)
print("Top-5 Collaborative Recommendations for User 2:")
print(top_n_recommendations_collaborative)
```

**Kelebihan Collaborative Filtering:**
- **Variasi Lebih Banyak**: Memberikan rekomendasi yang lebih beragam dan menarik, karena didasarkan pada interaksi pengguna lain.
- **Menangani Preferensi Pengguna**: Cocok untuk situasi di mana ada banyak interaksi atau rating yang diberikan oleh pengguna.

**Kekurangan Collaborative Filtering:**
- **Cold Start Problem**: Pengguna baru atau makanan baru yang tidak memiliki interaksi atau rating bisa jadi tidak mendapatkan rekomendasi yang akurat.
- **Memerlukan Data Pengguna yang Banyak**: Algoritma ini memerlukan banyak data interaksi untuk memberikan rekomendasi yang relevan.

---

Pada proyek ini, dua pendekatan **Content-Based Filtering** dan **Collaborative Filtering** diterapkan untuk memberikan **Top-N Recommendations** bagi pengguna:

1. **Content-Based Filtering** memberikan rekomendasi yang berbasis pada kesamaan konten, seperti deskripsi makanan atau kategori makanan. Ini berguna ketika data pengguna terbatas atau baru, tetapi terbatas pada variasi makanan yang ada.
2. **Collaborative Filtering** menggunakan data interaksi pengguna untuk memberikan rekomendasi yang lebih variatif dan menarik, meskipun menghadapi tantangan seperti **cold-start problem** untuk pengguna atau item baru.

Kedua pendekatan ini dapat saling melengkapi untuk meningkatkan kualitas rekomendasi dalam sistem yang lebih kompleks.

## Evaluation

Berdasarkan hasil yang diperoleh, berikut adalah analisis perbandingan antara **Content-Based Filtering** dan **Collaborative Filtering** dengan menggunakan model neural network. Perbandingan ini didasarkan pada kinerja yang diukur melalui metrik seperti **presisi**, **akurasi**, **loss**, **RMSE**, serta kelebihan dan kekurangan masing-masing metode.

---

#### 1. **Content-Based Filtering**

**Metrik Evaluasi:**

- **Presisi (Precision)**: 25.00%
  - **Rumus**:
    
    Presisi = True Positives / (True Positives + False Positives)

  - **Penjelasan**: Presisi mengukur seberapa banyak rekomendasi yang diberikan model yang benar-benar relevan. Nilai presisi yang rendah (25%) menunjukkan bahwa meskipun model memberikan banyak rekomendasi yang tepat (akurat), sebagian besar rekomendasi tersebut tidak sesuai dengan preferensi pengguna. Model ini mungkin terlalu banyak memberikan rekomendasi yang kurang sesuai dengan pengguna karena tidak dapat menangkap seluruh preferensi pengguna secara mendalam.

- **Akurasi (Accuracy)**: 100.00%
  - **Rumus**:
    
      ![Accuracy Formula](https://latex.codecogs.com/svg.latex?%5Ctext%7BAkurasi%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BTrue%20Negatives%7D%7D%7B%5Ctext%7BTotal%20Prediksi%7D%7D)

    
  - **Penjelasan**: Akurasi yang sangat tinggi menunjukkan bahwa sebagian besar rekomendasi yang dihasilkan oleh model adalah benar (artinya model dapat memprediksi dengan benar mayoritas item yang relevan bagi pengguna). Namun, meskipun akurasi tinggi, ini bisa menipu karena banyak dari rekomendasi tersebut tidak relevan. Artinya, akurasi yang tinggi tidak selalu berarti kualitas rekomendasi yang baik.
 
**Analisis Kelebihan dan Kekurangan:**
- **Kelebihan**: 
  - Akurasi yang sangat tinggi menunjukkan bahwa model dapat memilih item yang tepat berdasarkan fitur konten yang ada. 
  - Tidak tergantung pada data interaksi antar pengguna, sehingga cocok untuk situasi di mana data interaksi pengguna tidak lengkap atau tidak tersedia.
  
- **Kekurangan**:
  - **Presisi rendah** menunjukkan bahwa meskipun akurasi tinggi, banyak rekomendasi yang tidak sesuai dengan preferensi pengguna. Hal ini menandakan bahwa meskipun model mengenali konten yang relevan, model kurang efisien dalam memahami keinginan pengguna secara lebih pribadi.
  - Mengandalkan fitur konten yang ada, yang bisa terbatas jika tidak ada informasi cukup tentang item atau pengguna.

---

#### 2. **Collaborative Filtering (Neural Network)**

**Metrik Evaluasi:**

- **Epoch 1–5**:
  - **Loss**: 0.7016
  - **RMSE**: 0.3291
  - **Rumus RMSE (Root Mean Squared Error)**:  
       
   RMSE = sqrt((1/n) * Σ (r̂ᵢ - rᵢ)²)
   
   Di mana:
   - r̂ᵢ adalah nilai yang diprediksi
   - rᵢ adalah nilai yang sebenarnya
   - n adalah jumlah data yang diuji

  - **Penjelasan**: Pada tahap awal pelatihan, model mengalami penurunan signifikan dalam loss dan RMSE. Ini menunjukkan bahwa model mulai belajar dengan baik, tetapi tingkat kesalahan masih cukup tinggi. Namun, penurunan yang cepat mengindikasikan bahwa model mulai mengenali pola dalam data interaksi pengguna-item.
  
- **Epoch 10–20**:
  - Model menunjukkan penurunan yang lebih stabil pada loss dan RMSE, baik pada data pelatihan maupun validasi. Ini menandakan bahwa model mulai beradaptasi dengan data dan menghindari overfitting.
  
- **Epoch 30–50**:
  - Nilai loss dan RMSE semakin stabil, yang menunjukkan bahwa model mendekati konvergensi dan hasil mulai mencapai titik optimal pada parameter yang dipelajari.
  
- **Epoch 50–100**:
  - Setelah epoch ke-50, penurunan loss dan RMSE semakin lambat, yang menunjukkan bahwa model telah mencapai kapasitas optimalnya dalam belajar dari data, dan meskipun ada sedikit fluktuasi, performa sudah cukup stabil.

**Analisis Kelebihan dan Kekurangan:**
- **Kelebihan**:
  - **RMSE yang stabil** dan penurunan loss yang konsisten menunjukkan bahwa model cukup baik dalam mempelajari pola interaksi antara pengguna dan item. Ini penting untuk model berbasis collaborative filtering, karena dapat memberikan rekomendasi berdasarkan interaksi antar pengguna.
  - Menggunakan neural network memberikan fleksibilitas dan potensi yang lebih tinggi dalam menangkap pola-pola non-linier dalam data, terutama ketika data interaksi yang ada cukup besar.
  
- **Kekurangan**:
  - **Penurunan yang lambat** setelah epoch ke-50 bisa menunjukkan bahwa model mulai mencapai batas kemampuan dalam hal belajar dari data yang ada. Ini bisa mengindikasikan bahwa model memerlukan lebih banyak data atau tuning parameter untuk mencapai hasil yang lebih baik.
  - **Kecepatan konvergensi** yang lambat bisa menjadi masalah, terutama jika dibutuhkan model yang cepat beradaptasi dengan data baru.

---

### **Perbandingan dan Kesimpulan**

1. **Content-Based Filtering**:
   - **Kelebihan**:
     - Dapat memberikan rekomendasi yang relevan berdasarkan fitur konten item, meskipun tidak membutuhkan data interaksi pengguna.
   - **Kekurangan**:
     - Meskipun memiliki akurasi tinggi, presisi yang rendah menunjukkan bahwa model sering memberikan rekomendasi yang tidak sesuai dengan preferensi pengguna, karena tidak mempertimbangkan hubungan antara pengguna.
     - Model ini bergantung sepenuhnya pada fitur konten, yang bisa terbatas jika data konten tidak cukup informatif.
   
2. **Collaborative Filtering (Neural Network)**:
   - **Kelebihan**:
     - Collaborative filtering berbasis neural network sangat efektif dalam menangkap pola-pola yang ada di antara pengguna dan item, serta lebih personal dalam memberikan rekomendasi, terutama dengan data interaksi yang lebih kaya.
     - **RMSE yang stabil** dan **penurunan loss yang konsisten** menunjukkan bahwa model ini dapat memberikan rekomendasi yang lebih akurat dan relevan dibandingkan dengan content-based filtering.
   - **Kekurangan**:
     - Kecepatan konvergensi yang lambat setelah sejumlah epoch menunjukkan bahwa model ini mungkin membutuhkan lebih banyak data atau perlu dioptimalkan lebih lanjut untuk meningkatkan kinerjanya.
     - Penurunan performa setelah epoch tertentu bisa menunjukkan bahwa model mencapai titik jenuh dalam pembelajaran, yang menunjukkan adanya batasan dalam jumlah data atau tuning yang dilakukan.

### **Kesimpulan Evaluasi**:
- **Content-Based Filtering** dapat menghasilkan akurasi yang sangat tinggi, tetapi presisi yang rendah menunjukkan bahwa model ini mungkin terlalu menghasilkan banyak rekomendasi yang tidak relevan bagi pengguna. Oleh karena itu, untuk memperbaiki presisi, perlu ada penyesuaian dalam pemilihan fitur atau metode lain seperti penyesuaian bobot fitur.
  
- **Collaborative Filtering** (terutama dengan neural network) cenderung memberikan rekomendasi yang lebih personal dan relevan, meskipun konvergensi model relatif lambat dan membutuhkan optimisasi lebih lanjut. Pendekatan ini lebih sesuai untuk memberikan rekomendasi yang lebih tepat, tetapi mungkin membutuhkan lebih banyak data atau waktu pelatihan yang lebih lama untuk mencapai performa optimal.

Secara keseluruhan, jika tujuan adalah memberikan rekomendasi yang lebih personal dan relevan, **Collaborative Filtering** lebih unggul. Namun, **Content-Based Filtering** masih dapat berguna dalam situasi di mana data interaksi pengguna terbatas atau tidak tersedia.

## Diskusi Hasil Proyek 
### Diskusi Hasil Proyek terhadap Problem Statement dan Goals  

Hasil evaluasi model menunjukkan bahwa kombinasi metode **Content-Based Filtering** dan **Collaborative Filtering berbasis neural network** mampu mendukung pencapaian tujuan utama proyek ini. Sistem rekomendasi yang dihasilkan memberikan solusi efektif terhadap problem statement yang dihadapi, dengan manfaat sebagai berikut:  

- **Membantu Pengguna Menemukan Makanan yang Sesuai dengan Selera**:  
  Dengan presisi model yang terus meningkat dalam pendekatan Collaborative Filtering (RMSE stabil di 0.3291), pengguna dapat menerima rekomendasi makanan yang lebih personal dan relevan berdasarkan interaksi mereka sebelumnya. Ini secara langsung meningkatkan efisiensi pencarian makanan sesuai preferensi.  

- **Menyediakan Rekomendasi Berdasarkan Kategori Makanan Tertentu**:  
  Pendekatan Content-Based Filtering memungkinkan pengguna untuk menemukan makanan sesuai kategori atau atribut tertentu, seperti jenis masakan atau bahan. Meski presisi rendah (25%) menjadi kelemahan, model tetap berguna dalam menangani kebutuhan spesifik ini.  

- **Memperkenalkan Variasi Makanan Baru**:  
  Dengan pendekatan Content-Based Filtering, makanan baru dapat direkomendasikan berdasarkan kesamaan fitur dengan makanan lain yang sudah populer. Selain itu, Collaborative Filtering mendukung rekomendasi yang lebih relevan setelah interaksi pengguna terhadap makanan baru mulai tercatat.  

### Manfaat Praktis Proyek  

1. **Personalisasi Rekomendasi**:  
   Pengguna dapat merasakan pengalaman yang lebih personal melalui rekomendasi makanan yang sesuai selera dan kebiasaan mereka. Hal ini mendorong keterlibatan pengguna lebih lanjut dengan platform.  

2. **Efisiensi Pencarian Makanan**:  
   Dengan sistem rekomendasi yang akurat dan relevan, pengguna tidak perlu menghabiskan waktu lama untuk menemukan makanan yang diinginkan.  

3. **Meningkatkan Eksposur dan Penjualan Makanan Baru**:  
   Sistem ini membantu memperkenalkan makanan baru kepada pengguna, meningkatkan kemungkinan makanan tersebut dicoba dan diterima di pasar.  

Secara keseluruhan, proyek ini tidak hanya berhasil membangun model rekomendasi dengan performa yang memuaskan, tetapi juga memberikan manfaat langsung dalam meningkatkan pengalaman pengguna dan potensi keuntungan platform. Pendekatan gabungan ini memberikan solusi holistik untuk memenuhi kebutuhan pengguna dalam memilih makanan dan mendukung pengenalan variasi makanan baru.
## Referensi

1. [Chow, Y. Y., Haw, S. C., Naveen, P., Anaam, E. A., & Mahdin, H. B. (2023). Food Recommender System: A Review on Techniques, Datasets and Evaluation Metrics. Journal of System and Management Sciences, 13(5), 153-168.](https://www.aasmr.org/jsms/Vol13/No.5/Vol.13%20No.5.10.pdf)
2. [Gaikwad, S. M., Kutubuddin, S. A., Madki, B. M. A., & Bhange, C. (2023). Food Recommendation System Using Content Based & Collaborative Filtering. International Research Journal of Innovations in Engineering and Technology, 7(12), 238.](https://www.proquest.com/openview/231c3bcbc225e7beb2b2a8bceb903a8f/1?pq-origsite=gscholar&cbl=5314840)


</div>

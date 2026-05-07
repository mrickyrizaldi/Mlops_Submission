# Submission 2: Prediksi Diabetes Menggunakan TFX

Nama: **Muhammad Ricky Rizaldi**  
Username dicoding: **mrickyr**

| | Deskripsi |
| ----------- | ----------- |
| Dataset | Dataset prediksi diabetes dari Kaggle: [Diabetes-dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data), berisi 768 data pasien perempuan dengan 8 fitur medis dan 1 label target. |
| Masalah | Proyek mengangkat isu deteksi dini diabetes, yang umumnya membutuhkan pemeriksaan laboratorium yang memakan waktu dan biaya. Tujuannya adalah menyediakan model prediktif yang mampu mengidentifikasi pasien berisiko tinggi secara cepat berdasarkan indikator medis dasar.|
| Solusi machine learning | Dibangun pipeline end-to-end menggunakan **TensorFlow Extended (TFX)** yang mencakup ingest data, validasi, transformasi, hyperparameter tuning, training, evaluasi, dan pusher untuk model yang lolos validasi.|
| Metode pengolahan | Pengolahan dilakukan melalui modul Transform: normalisasi fitur numerik (`tft.scale_to_z_score`), imputasi nilai tidak logis (0) menggunakan median, encoding label biner, serta pembagian dataset menjadi train-test 80:20. |
| Arsitektur model | Model berupa **Dense Neural Network (DNN)** dengan dua hidden layer (baseline 64 & 32, hasil tuning 128 & 64), aktivasi ReLU, dan sigmoid pada output. Hyperparameter tuning mencakup neuron, learning rate, batch size, dan dropout. |
| Metrik evaluasi | Model dievaluasi menggunakan TFMA dengan metrik klinis: AUC ≥ 0.80, Recall ≥ 0.50, Precision ≥ 0.50, Accuracy ≥ 0.70 serta analisis confusion matrix (TP, FP, TN, FN). |
| Performa model |  Model tuned mencapai **AUC 0.88**, **Accuracy 0.78**, **Precision 0.76**, **Recall 0.62**, sehingga dinyatakan *blessed* dan layak untuk deployment. |
| Opsi deployment | Model dideploy menggunakan **TensorFlow Serving** dalam Docker container dan di-*host* pada platform **Railway**. Dockerfile memuat SavedModel, konfigurasi monitoring Prometheus, dan penyesuaian port Railway. |
| Web app | • Root URL: `https://diabetespredictionmlops-production.up.railway.app/` <br> • Endpoint prediksi: `https://diabetespredictionmlops-production.up.railway.app/v1/models/diabetes-prediction:predict`|
| Monitoring | Monitoring dilakukan menggunakan **Prometheus** dan **Grafana**. Metrik yang dipantau mencakup total request, status code, throughput, error rate, average latency, p95–p99 latency, low-traffic detection, dan spike detection. Model berjalan stabil dengan latency aman dan throughput konsisten. |

# Dokumentasi Lengkap
Dokumentasi dari proyek lebih lengkapnya dapat dilihat dibawah ini:

## 1. Informasi Dataset
Dataset yang digunakan dalam proyek ini merupakan **dataset prediksi diabetes** yang banyak digunakan dalam penelitian medis dan tersedia secara publik di Kaggle. Dataset ini terdiri dari **768 data pasien perempuan** dengan berbagai parameter medis yang dapat memengaruhi risiko diabetes. Setiap baris data merepresentasikan satu pasien dengan 8 fitur prediktor dan satu kolom target.

Berikut deskripsi fitur yang digunakan:
- **Pregnancies:** Jumlah kehamilan yang pernah dialami pasien.
- **Glucose:** Konsentrasi glukosa plasma setelah 2 jam dalam uji toleransi glukosa.
- **BloodPressure:** Tekanan darah diastolik (mm Hg).
- **SkinThickness:** Ketebalan lipatan kulit triceps (mm).
- **Insulin:** Kadar insulin serum 2 jam (mu U/ml).
- **BMI:** Indeks massa tubuh (kg/m²).
- **DiabetesPedigreeFunction:** Indeks yang menggambarkan riwayat keluarga terhadap diabetes.
- **Age:** Usia pasien.
- **Outcome:** Label target (1 = positif diabetes, 0 = negatif diabetes).

Sumber dataset: [Diabetes-dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data)

Dataset ini digunakan untuk membangun model prediktif yang dapat membantu mendeteksi dini risiko diabetes pada pasien berdasarkan indikator medis.

---

## 2. Permasalahan yang Ingin Diselesaikan
Diabetes merupakan salah satu penyakit kronis yang memerlukan deteksi dini untuk mencegah komplikasi serius. Dalam praktik medis, proses deteksi biasanya dilakukan dengan pemeriksaan laboratorium yang memerlukan waktu dan biaya. Oleh karena itu, proyek ini berfokus pada **pembuatan sistem prediksi berbasis machine learning** untuk membantu tenaga medis mengidentifikasi pasien berisiko tinggi berdasarkan data kesehatan dasar.

Tujuan utama proyek ini adalah:
- Membangun model machine learning yang dapat memprediksi probabilitas seseorang menderita diabetes.
- Mengintegrasikan pipeline end-to-end menggunakan **TensorFlow Extended (TFX)** untuk menjamin reprodusibilitas, validasi otomatis, dan deployment model secara konsisten.

---

## 3. Solusi Machine Learning
Solusi yang dikembangkan adalah **pipeline TFX** yang menangani seluruh tahapan workflow machine learning, mulai dari pengambilan data, pembersihan, transformasi, pelatihan, evaluasi, hingga penyimpanan model yang telah lolos validasi (*blessed*). Pipeline ini terdiri dari komponen berikut:

1. **ExampleGen:** Mengimpor dataset dan membagi menjadi data latih dan data uji.
2. **StatisticsGen & SchemaGen:** Menghasilkan statistik deskriptif dan mendeteksi anomali atau outlier pada fitur medis.
3. **ExampleValidator:** Memvalidasi data terhadap skema untuk memastikan tidak ada nilai hilang atau inkonsistensi.
4. **Transform:** Melakukan pembersihan, normalisasi, dan transformasi fitur menggunakan `tft` agar model dapat beroperasi secara optimal.
5. **Tuner:** Melakukan pencarian *hyperparameter* terbaik secara otomatis.
6. **Trainer:** Melatih model neural network berdasarkan parameter hasil tuning.
7. **Evaluator:** Mengevaluasi performa model menggunakan metrik medis yang disesuaikan dengan konteks klinis.
8. **Pusher:** Menyimpan model ke direktori serving jika dinyatakan *blessed* oleh evaluator.

---

## 4. Metode Pengolahan Data
Tahapan pengolahan data dilakukan menggunakan modul `Transform` pada TFX dengan langkah-langkah berikut:

1. **Normalisasi Fitur Numerik:** Semua fitur numerik seperti `Glucose`, `BMI`, `Age`, dan `Insulin` dinormalisasi menggunakan metode `tft.scale_to_z_score()` untuk memastikan distribusi data seragam.
2. **Penanganan Nilai Nol:** Fitur seperti `Insulin` dan `SkinThickness` yang memiliki nilai 0 (tidak logis secara medis) diimputasi menggunakan nilai median dari distribusi data.
3. **Encoding Label:** Kolom `Outcome` diubah menjadi format biner (0 dan 1) untuk mendukung klasifikasi.
4. **Pembagian Data:** Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan `ExampleGen` secara acak namun terkontrol.

Transformasi ini memastikan model memiliki data bersih dan terstandarisasi sesuai praktik terbaik dalam pemrosesan data medis.

---

## 5. Arsitektur Model dan Hyperparameter Tuning

### 5.1 Arsitektur Model
Model dikembangkan menggunakan **Dense Neural Network (DNN)** yang terdiri dari beberapa lapisan *fully connected* dengan fungsi aktivasi ReLU dan sigmoid di output layer. Arsitektur awal model adalah sebagai berikut:

- **Input Layer:** Menyesuaikan jumlah fitur (8 fitur input medis).
- **Hidden Layer 1:** Dense 64 neuron, aktivasi ReLU.
- **Hidden Layer 2:** Dense 32 neuron, aktivasi ReLU.
- **Output Layer:** Dense 1 neuron, aktivasi sigmoid.

Model dikompilasi dengan:
```python
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
```

### 5.2 Hyperparameter Tuning
Komponen **Tuner** digunakan untuk mencari kombinasi terbaik dari:
- Jumlah neuron pada tiap layer.
- Learning rate (0.001 – 0.0001).
- Batch size (16 – 64).
- Dropout rate (0.1 – 0.4).

Setelah tuning, diperoleh hasil terbaik dengan konfigurasi:
- Hidden Layer 1: 128 neuron
- Hidden Layer 2: 64 neuron
- Learning rate: 0.001
- Batch size: 32

Hasil tuning menunjukkan peningkatan signifikan dibanding model baseline.

| Tahap | AUC (Val) | Accuracy (Val) | Precision (Val) | Recall (Val) |
|-------|------------|----------------|------------------|---------------|
| Sebelum Tuning | 0.853 | 0.755 | 0.760 | 0.520 |
| Setelah Tuning | **0.883** | **0.781** | **0.762** | **0.616** |

Peningkatan **Recall** menunjukkan model lebih sensitif dalam mengenali pasien dengan diabetes tanpa mengorbankan presisi secara berlebihan.

---

## 6. Evaluasi Model dan Metrik
Evaluasi model dilakukan menggunakan **TensorFlow Model Analysis (TFMA)** dengan konfigurasi metrik medis yang realistis:

- **AUC (≥ 0.80):** Mengukur kemampuan model membedakan antara kasus positif dan negatif.
- **Recall (≥ 0.50):** Meminimalkan *false negative* agar kasus diabetes tidak terlewat.
- **Precision (≥ 0.50):** Mengontrol *false positive* agar pasien sehat tidak salah deteksi.
- **BinaryAccuracy (≥ 0.70):** Indikator umum performa keseluruhan model.

Selain itu, ditambahkan metrik **True Positives (TP)**, **False Positives (FP)**, **True Negatives (TN)**, dan **False Negatives (FN)** untuk membentuk confusion matrix yang membantu analisis kesalahan prediksi.

Hasil evaluasi akhir:
- **AUC:** 0.88
- **Accuracy:** 0.78
- **Precision:** 0.76
- **Recall:** 0.62

Hasil evaluasi menunjukkan bahwa model melewati semua ambang batas evaluasi yang ditetapkan dan dinyatakan *blessed* oleh evaluator.

---

## 7. Performa Model
Model yang telah dinyatakan *blessed* secara otomatis dikirim ke direktori serving oleh komponen **Pusher**. Lokasi penyimpanan model:
```
mrickyr-pipeline\serving_model\mrickyr-pipeline\1762251112
```
Model ini kemudian diuji dengan TensorFlow Serving via Docker. Pengujian dilakukan dengan dua skenario:
1. Data dari dataset validasi (index dan batch) – Model berhasil memberikan prediksi konsisten terhadap ground truth.
2. Data input baru (manual) – Model mampu memprediksi dengan benar pasien berisiko diabetes dan non-diabetes berdasarkan indikator medis.

---

Berhubung dokumentasi di README kamu **sudah lengkap untuk bagian dataset, masalah, solusi, metode pengolahan, arsitektur, metrik, dan performa model**, maka aku tinggal menambahkan **tiga bagian yang belum ada**:

* **Opsi Deployment**
* **Web App (Model Serving URL)**
* **Monitoring**

Di bawah ini adalah teks final yang bisa kamu **copy-paste langsung** ke README / laporan submission tanpa perlu modifikasi lagi. Sudah disesuaikan dengan **apa yang benar-benar kamu kerjakan**.

---

## 8. Opsi Deployment

Pada proyek ini, model yang telah dinyatakan *blessed* oleh TFX Evaluator dideploy menggunakan **TensorFlow Serving** dalam sebuah **Docker container**. Selanjutnya, container ini dijalankan dan di-*host* di platform cloud **Railway**.

Alasan pemilihan Railway sebagai platform hosting:

* Mendukung **deploy Docker image** secara langsung.
* Memiliki manajemen container yang sederhana.
* Menyediakan domain publik sehingga model dapat diakses melalui internet.
* Menyediakan dashboard metrik bawaan (CPU, memory, request) sebagai pelengkap monitoring.
* Lebih mudah di akses dan digunakan.

Untuk deployment, dilakukan langkah berikut:

1. **Membangun Dockerfile TensorFlow Serving**
   Dockerfile memuat:

   * Model hasil TFX (`serving_model/`)
   * Konfigurasi monitoring (`prometheus.config`)
   * Custom entrypoint agar port REST mengikuti environment Railway (`$PORT`)

2. **Push Docker Image ke Repository GitHub**
   Railway otomatis membangun dan menjalankan container dari Dockerfile tersebut.

3. **Model Serving Aktif** di endpoint publik Railway melalui REST API TensorFlow Serving.

Deployment dilakukan menggunakan image resmi `tensorflow/serving` sehingga kompatibel dengan format SavedModel yang dihasilkan TFX.

---

## 9. Web App (Model Serving URL)

Setelah deployment di Railway berhasil, model dapat diakses melalui endpoint publik berikut:

```
https://diabetespredictionmlops-production.up.railway.app/
```

Endpoint untuk melakukan prediksi (REST API):

```
https://diabetespredictionmlops-production.up.railway.app/v1/models/diabetes-prediction:predict
```

Pengguna dapat mengirimkan data dalam format **TF Example (base64)** seperti yang dipersyaratkan TensorFlow Serving.

Contoh akses menggunakan Python:

```python
res = requests.post(
    "https://diabetespredictionmlops-production.up.railway.app/v1/models/diabetes-prediction:predict",
    json={"instances": [{"b64": "<serialized_base64_tfexample>"}]}
)
```

Endpoint ini digunakan untuk semua proses inferensi model pada tahap deployment.

---

## 10. Monitoring

Monitoring dilakukan menggunakan kombinasi:

* **Prometheus** (mengambil metrik dari TensorFlow Serving)
* **Grafana** (untuk visualisasi dashboard)

### Monitoring dengan Prometheus

TensorFlow Serving menyediakan endpoint metrik Prometheus melalui konfigurasi:

```
/monitoring/prometheus/metrics
```

Konfigurasi monitoring diaktifkan melalui file:

```
prometheus.config
```

Prometheus kemudian dijalankan secara lokal (Docker) dan melakukan **scraping ke Railway** menggunakan konfigurasi:

```yaml
scrape_configs:
  - job_name: "tf-serving-railway"
    scrape_interval: 5s
    metrics_path: /monitoring/prometheus/metrics
    scheme: https
    static_configs:
      - targets: ['diabetespredictionmlops-production.up.railway.app']
```

Status target pada dashboard Prometheus ([http://localhost:9090/targets](http://localhost:9090/targets)) menunjukkan state **UP**, artinya scraping berjalan dengan baik.

### Monitoring dengan Grafana

Grafana dijalankan melalui Docker:

```
docker run -d -p 3000:3000 grafana/grafana
```

Prometheus diset sebagai data source, dan berbagai panel dibuat untuk memantau performa model.

Metrik penting yang dimonitor antara lain:

#### 1. Total API Requests

**Alasan:**
Memberikan gambaran total penggunaan model sejak service berjalan.

---

#### 2. API Request by Status Code

**Alasan:**
Memudahkan deteksi peningkatan request gagal dan memonitor distribusi status request.

---

#### 3. API Throughput (Requests per Second / RPS)

**Alasan:**
Menunjukkan intensitas traffic, membantu melihat pola penggunaan, dan mendeteksi potensi bottleneck.

---

#### 4. API Errors (jumlah & persentase)

**Alasan:**
Memantau gangguan dan reliabilitas model meskipun error saat ini kecil/0.

---

#### 5. Average API Latency

**Alasan:**
Mengukur performa waktu respon rata-rata dari perspektif pengguna (end-to-end latency).

---

#### 6. API Latency Tail (p95 / p99)

**Alasan:**
Menangkap "latency terburuk" yang tidak terlihat pada nilai rata-rata, penting untuk kualitas pengalaman pengguna.

---

#### 7. Low Traffic Detector

**Alasan:**
Menandakan berkurangnya penggunaan atau potensi gangguan pada client.

---

#### 8. Request Spike Detector

**Alasan:**
Mendeteksi lonjakan traffic yang tidak wajar dan membantu antisipasi kebutuhan scaling atau investigasi.

---

### Hasil Monitoring

Berdasarkan hasil visualisasi Grafana:  
* Model berhasil menerima beberapa request inferensi dari pengguna.
* Latency rata-rata stabil dan berada dalam batas normal.
* Dashboard bisa mendeteksi request OK dan Error
* Throughput rendah namun stabil, sesuai karena project bersifat demonstrasi.
* p95 latency berada di kisaran aman.
* Terdapat Traffic Anomaly Detection untuk mendeteksi kasus anomali dari user.

Monitoring menunjukkan bahwa model berjalan stabil pada Railway dan dapat diakses dengan lancar melalui REST API.

---

## 11. Kesimpulan
Proyek ini berhasil membangun pipeline machine learning end-to-end menggunakan TFX untuk memprediksi risiko diabetes secara terstruktur dan reproducible. Model yang dihasilkan melalui tahapan data processing, training, tuning, dan evaluasi kemudian berhasil dideploy menggunakan TensorFlow Serving di platform Railway sehingga dapat diakses melalui REST API publik. Selain itu, sistem monitoring berhasil diterapkan menggunakan Prometheus dan Grafana untuk memantau metrik penting seperti jumlah request, throughput, dan latency. Secara keseluruhan, proyek ini menunjukkan implementasi lengkap alur MLOps yang stabil, skalabel, dan siap digunakan dalam lingkungan produksi.
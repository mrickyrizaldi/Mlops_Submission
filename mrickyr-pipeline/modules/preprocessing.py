"""TFX Transform module for diabetes prediction.

Modul ini dipanggil oleh komponen TFX Transform untuk melakukan
preprocessing berbasis TensorFlow Transform (TFT), sehingga
proses preprocessing akan konsisten baik saat training maupun serving.

Transform meliputi:
- Imputasi nilai 0 pada fitur tertentu (karena dianggap missing),
- Normalisasi z-score untuk semua fitur numerik,
- Konversi label menjadi int64.
"""

from typing import Dict, Any

import tensorflow as tf
import tensorflow_transform as tft


# Fitur numerik dalam dataset diabetes
NUMERICAL_FEATURE_KEYS = [
    "Age",
    "BMI",
    "BloodPressure",
    "DiabetesPedigreeFunction",
    "Glucose",
    "Insulin",
    "Pregnancies",
    "SkinThickness",
]

# Nama kolom label
LABEL_KEY = "Outcome"

# Fitur yang ketika bernilai 0 dianggap missing dan perlu imputasi
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def transformed_name(key: str) -> str:
    """Menambahkan suffix '_xf' pada nama fitur hasil transformasi."""
    return f"{key}_xf"


def _impute_zero_with_nonzero_mean(x: tf.Tensor) -> tf.Tensor:
    """Mengganti nilai 0 dengan rata-rata nilai non-zero menggunakan TFT.

    TFT akan menghitung mean non-zero secara global saat fase analisis,
    lalu nilai 0 pada data diisi dengan mean tersebut.
    """
    x = tf.cast(x, tf.float32)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    is_zero = tf.equal(x, 0.0)

    # Hanya nilai non-zero yang dipakai untuk menghitung mean
    masked_sum = tf.where(is_zero, tf.zeros_like(x), x)

    sum_all = tft.mean(masked_sum)
    zero_ratio = tft.mean(tf.cast(is_zero, tf.float32))
    nonzero_ratio = tf.maximum(1.0 - zero_ratio, 1e-6)

    mean_nonzero = sum_all / nonzero_ratio

    # Replace nilai 0 dengan mean non-zero
    return tf.where(is_zero, mean_nonzero, x)


def preprocessing_fn(inputs: Dict[str, Any]) -> Dict[str, tf.Tensor]:
    """Preprocess fitur mentah menjadi fitur tertransformasi.

    Args:
        inputs: Mapping nama fitur ke tensor mentah dari ExampleGen.

    Returns:
        Dictionary berisi fitur numerik yang telah di-scale serta
        label yang di-cast ke int64.
    """
    outputs: Dict[str, tf.Tensor] = {}

    # Normalisasi dan imputasi untuk seluruh fitur numerik
    for key in NUMERICAL_FEATURE_KEYS:
        value = tf.cast(inputs[key], tf.float32)
        value = tf.where(tf.math.is_nan(value), tf.zeros_like(value), value)

        if key in ZERO_AS_MISSING:
            value = _impute_zero_with_nonzero_mean(value)

        outputs[transformed_name(key)] = tft.scale_to_z_score(value)

    # Label di-cast ke int64 agar cocok untuk model Keras
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.int64
    )

    return outputs

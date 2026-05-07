import tensorflow as tf
import tensorflow_transform as tft

# fitur numerik
NUMERIC_FEATURE_KEYS = [
    'Age',
    'BMI',
    'BloodPressure',
    'DiabetesPedigreeFunction',
    'Glucose',
    'Insulin',
    'Pregnancies',
    'SkinThickness'
]

# label (target)
LABEL_KEY = 'Outcome'

# fitur bernilai 0 tidak masuk akal (anggap saja missing)
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def _xf(name): 
    '''Tambahkan suffix _xf pada nama fitur hasil transformasi.'''
    return f"{name}_xf"

def _impute_zero_with_nonzero_mean(x: tf.Tensor) -> tf.Tensor:
    """
    Mengganti nilai nol dengan rata-rata dari nilai bukan nol menggunakan analyzer TFT. 
    ...
    """
    x = tf.cast(x, tf.float32)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    is_zero = tf.equal(x, 0.0)
    masked_sum = tf.where(is_zero, tf.zeros_like(x), x)

    sum_all = tft.mean(masked_sum)
    zero_ratio = tft.mean(tf.cast(is_zero, tf.float32))
    nonzero_ratio = 1.0 - zero_ratio
    mean_nonzero = sum_all / tf.maximum(nonzero_ratio, 1e-6)

    return tf.where(is_zero, mean_nonzero, x)


def preprocessing_fn(inputs):
    """
    Fungsi preprocessing...
    """
    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        val = tf.cast(inputs[key], tf.float32)
        val = tf.where(tf.math.is_nan(val), tf.zeros_like(val), val)

        if key in ZERO_AS_MISSING:
            val = _impute_zero_with_nonzero_mean(val)

        # Standardisasi
        outputs[_xf(key)] = tft.scale_to_z_score(val)

    # Label wajib int64
    outputs[_xf(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs

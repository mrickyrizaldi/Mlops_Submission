import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Konstanta dan Helper Function
LABEL_KEY = "Outcome"  # label asli


def transformed_name(key: str) -> str:
    """Menambahkan suffix '_xf' untuk menandai fitur hasil transformasi."""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Membaca TFRecord hasil transformasi (kompresi GZIP)."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


# Input Function
def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64) -> tf.data.Dataset:
    """
    Membuat dataset TF dari file TFRecord hasil transformasi.

    Args:
        file_pattern (str): Pola path file TFRecord hasil transformasi.
        tf_transform_output (tft.TFTransformOutput): Objek hasil komponen Transform.
        num_epochs (int): Jumlah epoch untuk membaca data (None = infinite).
        batch_size (int): Ukuran batch.

    Returns:
        tf.data.Dataset: Dataset yang siap digunakan untuk training/evaluasi.
    """
    # Mendapatkan spesifikasi fitur hasil transformasi
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Membuat dataset dalam bentuk batch
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    ).repeat()
    return dataset


def model_builder(hparams=None):
    """
    Membangun model jaringan saraf tiruan (MLP) sederhana untuk klasifikasi biner.

    Args:
        hparams (dict, optional): Hyperparameter dari komponen Tuner (jika ada).

    Returns:
        tf.keras.Model: Model Keras yang sudah dikompilasi.
    """
    # deteksi tipe parameter (dict atau HyperParameters)
    if hparams is None:
        # jika dipanggil tanpa tuner (Trainer biasa)
        hidden_units = 64
        dropout_rate = 0.2
        learning_rate = 1e-3

    elif hasattr(hparams, "Choice"):  # dipanggil oleh keras_tuner
        hp = hparams
        hidden_units = hp.Int("units", min_value=32, max_value=128, step=32)
        dropout_rate = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        # turunkan range learning rate biar tidak terlalu agresif
        learning_rate = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

    else:  # dipanggil oleh Trainer dengan dict hasil dari tuner_fn
        hidden_units = int(hparams.get("units", 64))
        dropout_rate = float(hparams.get("dropout", 0.2))
        learning_rate = float(hparams.get("learning_rate", 1e-3))

    # definisi input layer sesuai fitur hasil Transform
    inputs = {
        name: layers.Input(shape=(1,), name=name, dtype=tf.float32)
        for name in [
            'Age_xf', 'BMI_xf', 'BloodPressure_xf', 'DiabetesPedigreeFunction_xf',
            'Glucose_xf', 'Insulin_xf', 'Pregnancies_xf', 'SkinThickness_xf'
        ]
    }

    # Concatenate seluruh fitur numerik
    x = layers.Concatenate(name="concatenate_inputs")(list(inputs.values()))
    
    # hidden layers
    x = layers.Dense(hidden_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units // 2, activation='relu')(x)

    # output layer
    outputs = layers.Dense(1, activation='sigmoid', name='probability')(x)

    # compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # tampilkan ringkasan model
    model.summary()
    return model


# Serving Function (untuk model deployment)
def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Mendefinisikan signature function agar model dapat menerima input mentah
    dalam format tf.Example saat deployment (TensorFlow Serving).
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """
    Fungsi utama untuk melatih model.
    Fungsi ini dijalankan oleh komponen Trainer TFX dan meliputi:
    - Membaca hasil transformasi,
    - Menyusun pipeline input,
    - Melatih model dengan callback,
    - Menyimpan model beserta serving signature.
    """
    # Muat hasil Transform
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Siapkan dataset training & evaluasi
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_set  = input_fn(fn_args.eval_files,  tf_transform_output, num_epochs=10)

    # Ambil hyperparameter jika Tuner digunakan
    hparams = None
    if getattr(fn_args, "hyperparameters", None):
        try:
            hparams = fn_args.hyperparameters.get("values") or fn_args.hyperparameters
        except Exception:
            hparams = None

    # Bangun model
    model = model_builder(hparams=hparams)

    # Siapkan callback
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

    # Latih model
    model.fit(
        x=train_set,
        validation_data=eval_set,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=50,
        callbacks=[tensorboard_callback, early_stopping, reduce_lr],
        verbose=2
    )

    # Buat serving signature
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    # Simpan model dalam format TensorFlow SavedModel
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

"""TFX Trainer module for diabetes prediction.

Berisi tiga bagian utama:
- input_fn: membaca TFRecord hasil Transform menjadi tf.data.Dataset.
- model_builder: membangun arsitektur model Keras untuk klasifikasi diabetes.
- run_fn: entry point yang dipanggil komponen TFX Trainer untuk training.
"""

# pylint: disable=import-error

from typing import Any, Dict, Iterable, Mapping, Optional, Union
import os

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

# Nama kolom label asli di dataset
LABEL_KEY = "Outcome"


def transformed_name(key: str) -> str:
    """Menambahkan suffix '_xf' untuk menandai fitur hasil transformasi."""
    return f"{key}_xf"


def gzip_reader_fn(filenames: Iterable[str]) -> tf.data.TFRecordDataset:
    """Reader untuk membaca TFRecord hasil Transform yang dikompresi GZIP."""
    return tf.data.TFRecordDataset(
        filenames,
        compression_type="GZIP",
    )


def input_fn(
    file_pattern: Union[str, Iterable[str]],
    tf_transform_output: tft.TFTransformOutput,
    num_epochs: Optional[int] = None,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Membuat tf.data.Dataset dari file TFRecord hasil Transform.

    Dataset yang dihasilkan sudah:
    - memakai feature spec hasil Transform (transformed_feature_spec),
    - membentuk batch,
    - memisahkan fitur dan label berdasarkan label_key.
    """
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def _get_default_hparams() -> Dict[str, float]:
    """Mengembalikan default hyperparameter ketika tuner tidak digunakan."""
    return {
        "units": 64.0,
        "dropout": 0.2,
        "learning_rate": 1e-3,
    }


def _parse_hparams(
    hparams: Optional[Union[kt.HyperParameters, Mapping[str, Any]]],
) -> Dict[str, float]:
    """Menyamakan format hyperparameter menjadi dict sederhana.

    Mendukung tiga kasus:
    - hparams is None → pakai default,
    - hparams adalah objek HyperParameters dari KerasTuner,
    - hparams adalah dict/mapping biasa (hasil dari TFX Tuner).
    """
    if hparams is None:
        return _get_default_hparams()

    if isinstance(hparams, kt.HyperParameters):
        units = float(
            hparams.Int(
                "units",
                min_value=32,
                max_value=128,
                step=32,
            )
        )
        dropout = float(
            hparams.Float(
                "dropout",
                min_value=0.1,
                max_value=0.5,
                step=0.1,
            )
        )
        learning_rate = float(
            hparams.Choice(
                "learning_rate",
                [1e-3, 5e-4, 1e-4],
            )
        )
        return {
            "units": units,
            "dropout": dropout,
            "learning_rate": learning_rate,
        }

    # Asumsikan mapping/dict biasa (misalnya dari TFX tuner)
    units = float(hparams.get("units", 64.0))
    dropout = float(hparams.get("dropout", 0.2))
    learning_rate = float(hparams.get("learning_rate", 1e-3))

    return {
        "units": units,
        "dropout": dropout,
        "learning_rate": learning_rate,
    }


def model_builder(
    hparams: Optional[Union[kt.HyperParameters, Mapping[str, Any]]] = None,
) -> tf.keras.Model:
    """Membangun model MLP sederhana untuk klasifikasi biner (diabetes).

    Model:
    - Input: 8 fitur numerik tertransformasi (hasil Transform),
    - Dua hidden layer fully-connected dengan ReLU + dropout,
    - Output: 1 neuron sigmoid (probabilitas diabetes).
    """
    parsed_hparams = _parse_hparams(hparams)

    hidden_units = int(parsed_hparams["units"])
    dropout_rate = float(parsed_hparams["dropout"])
    learning_rate = float(parsed_hparams["learning_rate"])

    # Nama fitur input yang sudah diberi suffix '_xf' oleh Transform
    input_feature_names = [
        "Age_xf",
        "BMI_xf",
        "BloodPressure_xf",
        "DiabetesPedigreeFunction_xf",
        "Glucose_xf",
        "Insulin_xf",
        "Pregnancies_xf",
        "SkinThickness_xf",
    ]

    # Satu input layer per fitur, semua bertipe float32
    inputs: Dict[str, tf.keras.Input] = {
        name: layers.Input(
            shape=(1,),
            name=name,
            dtype=tf.float32,
        )
        for name in input_feature_names
    }

    # Gabungkan seluruh fitur numerik menjadi satu vektor
    x = layers.Concatenate(name="concatenate_inputs")(
        list(inputs.values())
    )

    # Hidden layer pertama
    x = layers.Dense(
        hidden_units,
        activation="relu",
        name="dense_1",
    )(x)
    x = layers.Dropout(
        dropout_rate,
        name="dropout_1",
    )(x)

    # Hidden layer kedua (lebih kecil)
    x = layers.Dense(
        hidden_units // 2,
        activation="relu",
        name="dense_2",
    )(x)

    # Output layer sigmoid untuk probabilitas
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="diabetes_classifier",
    )

    # Kompilasi model dengan loss & metrik yang relevan untuk klasifikasi biner
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # Menampilkan ringkasan model (membantu saat debugging)
    model.summary()

    return model


def _get_serve_tf_examples_fn(
    model: tf.keras.Model,
    tf_transform_output: tft.TFTransformOutput,
):
    """Mendefinisikan serving signature yang menerima tf.Example mentah.

    Fungsi ini akan:
    - Mem-parse serialized tf.Example sesuai raw_feature_spec,
    - Mengaplikasikan layer transform TFT yang sama seperti saat training,
    - Memanggil model untuk menghasilkan prediksi.
    """
    # Tambahkan layer transform TFT ke dalam model
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(
        serialized_tf_examples: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Fungsi serving yang menerima serialized tf.Example."""
        feature_spec = tf_transform_output.raw_feature_spec()
        # Label tidak dipakai saat serving
        feature_spec.pop(LABEL_KEY, None)

        # Parse tf.Example mentah → fitur mentah
        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            feature_spec,
        )
        # Terapkan transform yang sama seperti di komponen Transform
        transformed_features = model.tft_layer(parsed_features)

        # Kembalikan output model (probabilitas)
        return model(transformed_features)

    return serve_tf_examples_fn


def _extract_hparams_from_fn_args(
    fn_args: FnArgs,
) -> Optional[Mapping[str, Any]]:
    """Mengambil hyperparameter dari FnArgs jika tersedia.

    TFX biasanya menyimpan hyperparameters sebagai:
    - Dict dengan key "values", atau
    - Objek langsung di fn_args.hyperparameters.
    """
    if not getattr(fn_args, "hyperparameters", None):
        return None

    try:
        values = fn_args.hyperparameters.get("values")
        return values or fn_args.hyperparameters
    except Exception:  # pylint: disable=broad-except
        # Jika format tidak sesuai harapan, hilangkan saja hparams
        return None


def run_fn(fn_args: FnArgs) -> None:
    """Fungsi utama training yang dipanggil oleh komponen TFX Trainer.

    Alur:
    1. Muat hasil Transform (tf_transform_output).
    2. Bangun dataset train/eval dari TFRecord hasil Transform.
    3. Ambil hyperparameter terbaik (jika Tuner digunakan).
    4. Bangun dan latih model Keras.
    5. Simpan model dalam format SavedModel dengan serving signature.
    """
    # Muat artefak Transform (feature spec & transform graph)
    tf_transform_output = tft.TFTransformOutput(
        fn_args.transform_graph_path,
    )

    # Dataset training dan evaluasi dari TFRecord hasil Transform
    train_set = input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_epochs=10,
    )
    eval_set = input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        num_epochs=10,
    )

    # Ambil hyperparameters jika tersedia dari Tuner
    hparams_mapping = _extract_hparams_from_fn_args(fn_args)

    # Bangun model dengan hyperparameters (atau default)
    model = model_builder(hparams=hparams_mapping)

    # Logging ke TensorBoard
    log_dir = os.path.join(
        os.path.dirname(fn_args.serving_model_dir),
        "logs",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq="epoch",
    )

    # Early stopping dan penurunan learning rate berbasis metrik val_auc
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    # Training model
    model.fit(
        x=train_set,
        validation_data=eval_set,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=50,
        callbacks=[tensorboard_callback, early_stopping, reduce_lr],
        verbose=2,
    )

    # Buat serving function yang menerima tf.Example mentah
    serving_fn = _get_serve_tf_examples_fn(
        model=model,
        tf_transform_output=tf_transform_output,
    )
    signatures = {
        "serving_default": serving_fn.get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples",
            ),
        ),
    }

    # Simpan model dalam format SavedModel dengan signature untuk serving
    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures,
    )

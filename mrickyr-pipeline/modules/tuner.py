"""TFX Tuner module for diabetes prediction.

Modul ini berisi fungsi `tuner_fn` yang dipanggil oleh komponen
TFX Tuner untuk melakukan hyperparameter tuning menggunakan KerasTuner.

Tuner akan:
- Membaca data TFRecord hasil Transform via input_fn.
- Mencari kombinasi hyperparameter terbaik untuk model_builder.
- Mengembalikan objek TunerFnResult ke komponen Tuner.
"""

from typing import Any, Dict

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

from modules.trainer import input_fn, model_builder


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama untuk menjalankan hyperparameter tuning.

    Fungsi ini akan:
    - Memuat artefak Transform (tf_transform_output).
    - Membangun dataset train dan eval dari TFRecord hasil Transform.
    - Mendefinisikan KerasTuner (Hyperband) yang membungkus model_builder.
    - Mengembalikan TunerFnResult yang berisi tuner dan argumen fit().
    """
    # Artefak Transform: berisi transform_graph & feature spec
    tf_transform_output = tft.TFTransformOutput(
        fn_args.transform_graph_path,
    )

    # Dataset training & evaluasi berbasis TFRecord hasil Transform
    train_set = input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_epochs=10,
    )
    val_set = input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        num_epochs=10,
    )

    # Callback untuk mencegah overfitting: hentikan lebih awal saat val_auc stagnan
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    # Callback untuk menurunkan learning rate ketika metrik tidak membaik
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    # KerasTuner Hyperband:
    # - hypermodel = model_builder (akan dipanggil berulang dengan kombinasi hparams berbeda)
    # - objective = val_auc (ingin dimaksimalkan)
    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=20,
        factor=3,
        directory=fn_args.working_dir,
        project_name="keras_tuner_diabetes",
    )

    # Argumen yang akan diteruskan ke tuner.search()/model.fit() oleh komponen TFX
    fit_kwargs: Dict[str, Any] = {
        "x": train_set,
        "validation_data": val_set,
        "steps_per_epoch": fn_args.train_steps,
        "validation_steps": fn_args.eval_steps,
        "epochs": 50,
        "callbacks": [stop_early, reduce_lr],
        "verbose": 2,
    }

    # Kembalikan konfigurasi untuk komponen TFX Tuner
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs=fit_kwargs,
    )

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

# Import fungsi dari trainer module
from trainer import input_fn, model_builder, transformed_name, LABEL_KEY


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Fungsi utama untuk menjalankan hyperparameter tuning.
    Menggunakan model_builder dan input_fn dari trainer.py.
    """
    # Muat hasil Transform
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Siapkan dataset training & evaluasi (pakai fungsi dari trainer.py)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set   = input_fn(fn_args.eval_files,  tf_transform_output, num_epochs=10)

    # Callback early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=10, restore_best_weights=True
    )
    
    # callback reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-5, verbose=1
    )

    # Definisikan strategi tuning dengan Hyperband
    tuner = kt.Hyperband(
        model_builder,  # panggil fungsi model_builder dari trainer
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=20,
        factor=3,
        directory=fn_args.working_dir,
        project_name="keras_tuner_linked"
    )

    # Kembalikan hasil untuk komponen TFX Tuner
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 50,
            "callbacks": [stop_early, reduce_lr],
            "verbose": 2
        }
    )

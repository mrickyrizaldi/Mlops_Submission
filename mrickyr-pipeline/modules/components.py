"""Init TFX pipeline components for diabetes prediction.

File ini berisi fungsi-fungsi helper untuk membangun seluruh komponen
pipeline TFX: mulai dari ingest data (CsvExampleGen), analisis data,
preprocessing (Transform), training + tuning model, evaluasi dengan TFMA,
hingga push model yang sudah lulus evaluasi ke direktori serving.
"""

import os
from dataclasses import dataclass

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


@dataclass
class PipelineConfig:
    """Config untuk inisialisasi pipeline TFX.

    Atribut:
        data_dir: Direktori berisi data mentah (CSV).
        transform_module: Path modul Transform (preprocessing.py).
        training_module: Path modul Trainer (trainer.py).
        tuner_module: Path modul Tuner (tuner.py).
        training_steps: Jumlah langkah training.
        eval_steps: Jumlah langkah evaluasi.
        serving_model_dir: Direktori output model untuk serving.
    """

    data_dir: str
    transform_module: str
    training_module: str
    tuner_module: str
    training_steps: int
    eval_steps: int
    serving_model_dir: str


def _create_example_components(data_dir: str):
    """Buat komponen ExampleGen + Statistik + Schema + Validator.

    Langkah:
    - CsvExampleGen: membaca data dari CSV dan membagi menjadi train/eval.
    - StatisticsGen: menghitung statistik dasar (mean, std, distribusi, dst.).
    - SchemaGen: menginfers schema (tipe data, domain nilai, dst.).
    - ExampleValidator: mengecek anomali/ketidaksesuaian terhadap schema.

    Args:
        data_dir: Direktori berisi file CSV.

    Returns:
        Tuple (example_gen, statistics_gen, schema_gen, example_validator).
    """
    # Atur pembagian data: 80% train, 20% eval dengan hash bucket
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train",
                    hash_buckets=8,
                ),
                example_gen_pb2.SplitConfig.Split(
                    name="eval",
                    hash_buckets=2,
                ),
            ],
        ),
    )

    # Ingest data CSV dan split menjadi train/eval
    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output_config,
    )

    # Hitung statistik dari data yang sudah di-split
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )

    # Infer schema (tipe, batas nilai, dsb.) dari statistik
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # Validasi data terhadap schema (mencari missing/ out-of-range / dsb.)
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    return example_gen, statistics_gen, schema_gen, example_validator


def _create_transform_component(
    example_gen: CsvExampleGen,
    schema_gen: SchemaGen,
    transform_module: str,
) -> Transform:
    """Buat komponen Transform.

    Komponen ini:
    - Mengaplikasikan preprocessing_fn dari modul Transform,
    - Menghasilkan transformed_examples dan transform_graph.

    Args:
        example_gen: Komponen ExampleGen.
        schema_gen: Komponen SchemaGen.
        transform_module: Path file modul Transform.

    Returns:
        Komponen Transform.
    """
    return Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        # os.path.abspath agar path bersifat absolut (aman untuk eksekusi dari mana pun)
        module_file=os.path.abspath(transform_module),
    )


def _create_tuner_and_trainer(
    transform: Transform,
    schema_gen: SchemaGen,
    config: PipelineConfig,
):
    """Buat komponen Tuner dan Trainer.

    Tuner:
    - Melakukan hyperparameter tuning menggunakan data yang sudah ditransform.
    Trainer:
    - Melatih model memakai grafik transform yang sama,
    - Menggunakan hyperparameter terbaik dari Tuner.

    Args:
        transform: Komponen Transform.
        schema_gen: Komponen SchemaGen.
        config: Konfigurasi pipeline (path modul & steps training/eval).

    Returns:
        Tuple (tuner, trainer).
    """
    tuner = Tuner(
        module_file=os.path.abspath(config.tuner_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"],
            num_steps=config.training_steps,
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"],
            num_steps=config.eval_steps,
        ),
    )

    trainer = Trainer(
        module_file=os.path.abspath(config.training_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        # Menghubungkan trainer dengan best_hyperparameters dari Tuner
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"],
            num_steps=config.training_steps,
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"],
            num_steps=config.eval_steps,
        ),
    )

    return tuner, trainer


def _create_model_resolver() -> Resolver:
    """Buat resolver untuk latest blessed model.

    Resolver ini akan mencari model terakhir yang sudah 'blessed'
    (lolos evaluasi sebelumnya) untuk dijadikan baseline saat evaluasi.
    """
    return Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")


def _create_eval_config() -> tfma.EvalConfig:
    """Buat konfigurasi evaluasi TFMA.

    Di sini didefinisikan:
    - label_key: kolom label (Outcome),
    - slicing_specs: evaluasi overall (tanpa slicing per grup),
    - metrics: metrik yang akan dihitung beserta threshold lulus (blessing).

    Threshold contoh:
    - AUC minimal 0.80,
    - Recall & Precision minimal 0.50,
    - Akurasi biner minimal 0.70,
    - Ditambah perbandingan terhadap model lama (change_threshold).
    """
    return tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key="Outcome",
            ),
        ],
        # Hanya evaluasi agregat (tidak ada slicing per feature tertentu)
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    # Hitung jumlah contoh yang dievaluasi
                    tfma.MetricConfig(
                        class_name="ExampleCount",
                    ),
                    # AUC dengan threshold performa minimum dan perubahan
                    tfma.MetricConfig(
                        class_name="AUC",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    "value": 0.80,
                                },
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=(
                                    tfma.MetricDirection.HIGHER_IS_BETTER
                                ),
                                absolute={
                                    "value": -0.01,
                                },
                            ),
                        ),
                    ),
                    # Recall
                    tfma.MetricConfig(
                        class_name="Recall",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    "value": 0.50,
                                },
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=(
                                    tfma.MetricDirection.HIGHER_IS_BETTER
                                ),
                                absolute={
                                    "value": -0.01,
                                },
                            ),
                        ),
                    ),
                    # Precision
                    tfma.MetricConfig(
                        class_name="Precision",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    "value": 0.50,
                                },
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=(
                                    tfma.MetricDirection.HIGHER_IS_BETTER
                                ),
                                absolute={
                                    "value": -0.01,
                                },
                            ),
                        ),
                    ),
                    # Akurasi biner
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={
                                    "value": 0.70,
                                },
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=(
                                    tfma.MetricDirection.HIGHER_IS_BETTER
                                ),
                                absolute={
                                    "value": -0.01,
                                },
                            ),
                        ),
                    ),
                    # Metrik confusion matrix
                    tfma.MetricConfig(
                        class_name="TruePositives",
                    ),
                    tfma.MetricConfig(
                        class_name="FalsePositives",
                    ),
                    tfma.MetricConfig(
                        class_name="TrueNegatives",
                    ),
                    tfma.MetricConfig(
                        class_name="FalseNegatives",
                    ),
                ],
            ),
        ],
    )


def _create_evaluator(
    example_gen: CsvExampleGen,
    trainer: Trainer,
    model_resolver: Resolver,
    eval_config: tfma.EvalConfig,
) -> Evaluator:
    """Buat komponen Evaluator.

    Evaluator:
    - Menggunakan model hasil training,
    - Membandingkan dengan baseline model (dari resolver, jika ada),
    - Menghitung metrik sesuai eval_config (TFMA),
    - Menghasilkan 'blessing' (lulus/tidak lulus threshold).
    """
    return Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )


def _create_pusher(
    trainer: Trainer,
    evaluator: Evaluator,
    config: PipelineConfig,
) -> Pusher:
    """Buat komponen Pusher.

    Pusher hanya akan mendorong (push) model ke direktori serving jika:
    - Model mendapat 'blessing' dari Evaluator (lulus threshold metrik).

    Args:
        trainer: Komponen Trainer yang menghasilkan model.
        evaluator: Komponen Evaluator yang menghasilkan blessing.
        config: Konfigurasi pipeline (lokasi serving_model_dir).

    Returns:
        Komponen Pusher.
    """
    return Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir,
            ),
        ),
    )


def init_components(config: PipelineConfig) -> tuple:
    """Init TFX pipeline components for diabetes prediction.

    Fungsi ini adalah entry point dari modul components:
    menggabungkan semua helper di atas menjadi satu tuple komponen
    yang siap diberikan ke objek tfx.orchestration.Pipeline.

    Args:
        config: Objek konfigurasi pipeline.

    Returns:
        Tuple komponen TFX dengan urutan:
        (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
        Transform, Tuner, Trainer, ModelResolver, Evaluator, Pusher).
    """
    # Data ingestion (example gen)
    (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
    ) = _create_example_components(config.data_dir)

    # Transform (preprocessing)
    transform = _create_transform_component(
        example_gen=example_gen,
        schema_gen=schema_gen,
        transform_module=config.transform_module,
    )

    # Hyperparameter tuning + training model
    tuner, trainer = _create_tuner_and_trainer(
        transform=transform,
        schema_gen=schema_gen,
        config=config,
    )

    # model baseline + konfigurasi evaluasi TFMA
    model_resolver = _create_model_resolver()
    eval_config = _create_eval_config()

    # Evaluasi model dan blessing
    evaluator = _create_evaluator(
        example_gen=example_gen,
        trainer=trainer,
        model_resolver=model_resolver,
        eval_config=eval_config,
    )

    # Push model ke direktori serving jika lulus evaluasi
    pusher = _create_pusher(
        trainer=trainer,
        evaluator=evaluator,
        config=config,
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )

    return components

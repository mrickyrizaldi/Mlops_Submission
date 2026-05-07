"""Main TFX pipeline definition for Diabetes prediction (mrickyr-pipeline).

File ini mengatur struktur direktori project, membuat konfigurasi pipeline,
menginisialisasi seluruh komponen TFX, dan menjalankan pipeline menggunakan
BeamDagRunner.
"""

import os
from typing import Iterable, Optional

from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from modules.components import PipelineConfig, init_components


# Lokasi dasar project
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)

# Struktur direktori input dan modul
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODULES_DIR = os.path.join(PIPELINE_DIR, "modules")

# Lokasi artefak pipeline
PIPELINES_ROOT = os.path.join(PIPELINE_DIR, "pipelines")
PIPELINE_NAME = "mrickyr-pipeline"
PIPELINE_ROOT = os.path.join(PIPELINES_ROOT, PIPELINE_NAME)

# Metadata SQLite untuk tracking eksekusi pipeline
METADATA_ROOT = os.path.join(PIPELINE_DIR, "metadata", PIPELINE_NAME)
METADATA_PATH = os.path.join(METADATA_ROOT, "metadata.db")

# Direktori model hasil serving
SERVING_ROOT = os.path.join(PIPELINE_DIR, "serving_model")
SERVING_MODEL_DIR = os.path.join(SERVING_ROOT, PIPELINE_NAME)

# Pastikan direktori output tersedia
for path in (PIPELINE_ROOT, METADATA_ROOT, SERVING_MODEL_DIR):
    os.makedirs(path, exist_ok=True)


# pylint: disable=too-many-arguments, too-many-positional-arguments
def create_pipeline(
    pipeline_name: str = PIPELINE_NAME,
    pipeline_root: str = PIPELINE_ROOT,
    data_root: str = DATA_ROOT,
    modules_dir: str = MODULES_DIR,
    serving_model_dir: str = SERVING_MODEL_DIR,
    metadata_path: str = METADATA_PATH,
    training_steps: int = 10,
    eval_steps: int = 3,
    beam_pipeline_args: Optional[Iterable[str]] = None,
) -> tfx_pipeline.Pipeline:
    """Menyusun dan mengembalikan objek TFX pipeline lengkap."""

    # Argumen default Apache Beam DirectRunner
    if beam_pipeline_args is None:
        beam_pipeline_args = [
            "--direct_running_mode=in_memory",
            "--direct_num_workers=1",
        ]

    # Path modul utama TFX
    transform_module = os.path.join(modules_dir, "preprocessing.py")
    trainer_module = os.path.join(modules_dir, "trainer.py")
    tuner_module = os.path.join(modules_dir, "tuner.py")

    # Konfigurasi pipeline sebagai dataclass
    config = PipelineConfig(
        data_dir=data_root,
        transform_module=transform_module,
        training_module=trainer_module,
        tuner_module=tuner_module,
        training_steps=training_steps,
        eval_steps=eval_steps,
        serving_model_dir=serving_model_dir,
    )

    # Inisialisasi seluruh komponen (ExampleGen → Pusher)
    components = init_components(config)

    # Bangun objek pipeline TFX
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=list(beam_pipeline_args),
    )


if __name__ == "__main__":
    pipeline_obj = create_pipeline()
    BeamDagRunner().run(pipeline_obj)
    print("Pipeline TFX (mrickyr-pipeline) selesai dijalankan.")

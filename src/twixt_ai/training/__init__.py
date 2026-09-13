"""Policy/value training workflows and commands."""

from .bootstrap import (
    BOOTSTRAP_FORMAT,
    BOOTSTRAP_VERSION,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mini_champion,
)
from .data import (
    DATASET_FORMAT,
    DATASET_VERSION,
    EXAMPLE_FORMAT,
    EXAMPLE_VERSION,
    DatasetConfig,
    DatasetSummary,
    Shard,
    build_dataset,
)
from .encoding_experiment import (
    MATCHED_ENCODING_TRAINING_FORMAT,
    MATCHED_ENCODING_TRAINING_VERSION,
    MatchedEncodingTrainingConfig,
    run_matched_encoding_training,
)
from .generations import (
    GENERATIONS_FORMAT,
    GENERATIONS_VERSION,
    MiniGenerationConfig,
    run_mini_training_generations,
)
from .inspection import (
    INSPECTION_FORMAT,
    INSPECTION_VERSION,
    PROBE_SET,
    build_mini_inspection_report,
    render_mini_inspection_report,
)
from .mini_experiment import (
    MiniTrainingExperimentConfig,
    run_mini_training_experiment,
)
from .selfplay_dataset import (
    SELFPLAY_DATASET_FORMAT,
    SELFPLAY_DATASET_VERSION,
    MiniSelfplayDatasetConfig,
    run_mini_selfplay_dataset,
)
from .trainer import (
    TRAINING_FORMAT,
    TRAINING_VERSION,
    EpochMetrics,
    TrainingConfig,
    TrainingSummary,
    train_model,
)
from .value_diagnostics import (
    VALUE_DIAGNOSTICS_FORMAT,
    VALUE_DIAGNOSTICS_VERSION,
    ValueDiagnosticsConfig,
    diagnose_value_model,
)

__all__ = [
    "BOOTSTRAP_FORMAT",
    "BOOTSTRAP_VERSION",
    "DATASET_FORMAT",
    "DATASET_VERSION",
    "EXAMPLE_FORMAT",
    "EXAMPLE_VERSION",
    "GENERATIONS_FORMAT",
    "GENERATIONS_VERSION",
    "INSPECTION_FORMAT",
    "INSPECTION_VERSION",
    "MATCHED_ENCODING_TRAINING_FORMAT",
    "MATCHED_ENCODING_TRAINING_VERSION",
    "PROBE_SET",
    "SELFPLAY_DATASET_FORMAT",
    "SELFPLAY_DATASET_VERSION",
    "TRAINING_FORMAT",
    "TRAINING_VERSION",
    "VALUE_DIAGNOSTICS_FORMAT",
    "VALUE_DIAGNOSTICS_VERSION",
    "DatasetConfig",
    "DatasetSummary",
    "DEFAULT_BOOTSTRAP_SEED",
    "EpochMetrics",
    "MatchedEncodingTrainingConfig",
    "MiniGenerationConfig",
    "MiniSelfplayDatasetConfig",
    "MiniTrainingExperimentConfig",
    "Shard",
    "TrainingConfig",
    "TrainingSummary",
    "ValueDiagnosticsConfig",
    "build_dataset",
    "bootstrap_mini_champion",
    "build_mini_inspection_report",
    "diagnose_value_model",
    "render_mini_inspection_report",
    "run_matched_encoding_training",
    "run_mini_training_experiment",
    "run_mini_training_generations",
    "run_mini_selfplay_dataset",
    "train_model",
]

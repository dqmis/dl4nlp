import sys
from pathlib import Path

from transformers import Seq2SeqTrainingArguments

from src.trainer import Trainer
from src.utils import load_config, populate_training_args
from src.utils.dataset import DATASETS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def build_run_name(config: dict) -> str:
    return f"{Path(config['checkpoint']).name}-{Path(config['dataset']).name}-{config['source_lang']}-{config['target_lang']}"  # noqa


def get_training_arguments(cfg: dict) -> Seq2SeqTrainingArguments:
    training_args = cfg["training_args"]
    training_args = populate_training_args(training_args)
    training_args["run_name"] = build_run_name(cfg)
    training_args["output_dir"] = (OUT_DIR / training_args["run_name"]).resolve()

    return Seq2SeqTrainingArguments(**training_args)


def get_dataset(cfg: dict, trainer) -> dict:
    return DATASETS["helsinki"](
        cfg["dataset"],
        cfg["source_lang"],
        cfg["target_lang"],
        cfg.get("prefix", ""),
        trainer.tokenizer,
    )


def evaluate_flores(trainer, cfg: dict) -> None:
    test_set_flores = DATASETS["flores"](
        cfg["source_lang_flores"], cfg["target_lang_flores"], trainer.tokenizer
    )
    trainer.evaluate("flores", test_set_flores["devtest"])


def evaluate_lithuanian(trainer) -> None:
    trainer.evaluate("lithuanian-english")


def evaluate_ntrex(trainer, cfg: dict) -> None:
    test_set_ntrex = DATASETS["ntrex"](
        cfg["source_lang_flores"], cfg["target_lang_flores"], trainer.tokenizer
    )
    trainer.evaluate(test_set_ntrex, "ntrex")


def main(config_name: str = "train_config") -> None:
    cfg = load_config((CONFIG_DIR / f"{config_name}.yaml").resolve())
    trainer = Trainer(
        cfg["checkpoint"], source_lang=cfg["source_lang"], target_lang=cfg["target_lang"]
    )

    trainer.download_and_save_dataset(
        cfg["dataset_path"],
        cfg["dataset_name"],
    )

    # Get training arguments and dataset
    training_args = get_training_arguments(cfg)
    # dataset = get_dataset(cfg, trainer)

    # Train or evaluate model
    trainer.train(training_args, only_eval=cfg["only_eval"])

    # Evaluate on FLORES dataset
    evaluate_ntrex(trainer, cfg)
    evaluate_flores(trainer, cfg)


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "train_config"
    main(config_name)

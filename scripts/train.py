import sys
from pathlib import Path
import wandb
from transformers import Seq2SeqTrainingArguments

from src.trainer import Trainer
from src.utils import load_config, populate_training_args
from src.utils.dataset import DATASETS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

wandb.init(project="dl4nlp")


def build_run_name(config: dict) -> str:
    return f"{Path(config['checkpoint']).name}-{Path(config['dataset']).name}-{config['source_lang']}-{config['target_lang']}"  # noqa


def get_training_arguments(cfg: dict) -> Seq2SeqTrainingArguments:
    training_args = cfg["training_args"]
    training_args = populate_training_args(training_args)
    training_args["run_name"] = build_run_name(cfg)
    training_args["output_dir"] = (OUT_DIR / training_args["run_name"]).resolve()

    return Seq2SeqTrainingArguments(**training_args)


def get_dataset(cfg: dict, trainer) -> dict:
    return DATASETS["nllb"](
        cfg["dataset"],
        cfg["source_lang"],
        cfg["target_lang"],
        # cfg.get("prefix", ""),
        trainer.tokenizer,
    )


def evaluate_flores(trainer, cfg: dict) -> None:
    test_set_flores = DATASETS["flores"](
        cfg["source_lang_flores"], cfg["target_lang_flores"], trainer.tokenizer
    )
    trainer.evaluate(test_set_flores["devtest"], "flores")


def evaluate_ntrex(trainer, cfg: dict) -> None:
    test_set_ntrex = DATASETS["ntrex"](
        cfg["source_lang_flores"], cfg["target_lang_flores"], trainer.tokenizer
    )
    trainer.evaluate(test_set_ntrex, "ntrex")


def main(config_name: str = "train_config") -> None:
    cfg = load_config((CONFIG_DIR / f"{config_name}").resolve())
    trainer = Trainer(
        cfg["checkpoint"],
        source_lang=cfg["source_lang"],
        target_lang=cfg["target_lang"],
    )
    print(f"Training model: {cfg['checkpoint']}")

    # Get training arguments and dataset
    training_args = get_training_arguments(cfg)
    dataset = get_dataset(cfg, trainer)

    # Train or evaluate model
    trainer.train(training_args, dataset, only_eval=cfg["only_eval"])

    # Evaluate on FLORES dataset
    evaluate_ntrex(trainer, cfg)
    evaluate_flores(trainer, cfg)

    print("Training done!")


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "train_config"
    print("Running ", config_name)
    main(config_name)

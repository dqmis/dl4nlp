import os

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DatasetType = type[DatasetDict | Dataset | IterableDatasetDict | IterableDataset]


os.environ["WANDB_PROJECT"] = "dl4nlp"


class Trainer:
    def __init__(self, checkpoint: str, source_lang: str, target_lang: str) -> None:
        """
        Initialize the Trainer class with the given checkpoint

        Args:
            checkpoint (str): The checkpoint (model) to initialize the Trainer with.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self._trainer: Seq2SeqTrainer = None
        self._fast_eval = True

        self._source_lang = source_lang
        self._target_lang = target_lang

        self._data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=checkpoint)

    def train(
        self,
        training_args: Seq2SeqTrainingArguments,
        dataset: DatasetType,
        use_wandb: bool = False,
        only_eval: bool = False,
    ) -> None:
        """
        Train the model with the given training dataset along with the training arguments.

        Args:
            training_args (Seq2SeqTrainingArguments): The training arguments.
            dataset (DatasetType): The dataset to train the model on.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging.


        Returns:
            None
        """

        if use_wandb:
            training_args.report_to = "wandb"
            training_args.logging_steps = 1

        self._fast_eval = True

        self._trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics,
        )

        if not only_eval:
            self._trainer.train()
        else:
            print("Skipping training as only_eval is set to True")

    def evaluate(self, dataset: DatasetType, dataset_prefix: str) -> None:
        """
        Evaluate the model with the given dataset.

        Args:
            dataset (DatasetType): The dataset to evaluate the model on.
        """

        print(f"Evaluating on {dataset_prefix} dataset")

        self._fast_eval = False
        self._trainer.evaluate(dataset, metric_key_prefix=f"test_{dataset_prefix}")
        self._fast_eval = True

    def _postprocess_text(self, preds: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
        """
        Postprocess the given predictions and labels.

        Args:
            preds (list[str]): The predictions.
            labels (list[str]): The labels.

        Returns:
            tuple[list[str], list[str]]: The postprocessed predictions and labels
        """
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def _compute_metrics(
        self, eval_preds: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> dict[str, float]:
        """
        Compute the metrics for the given evaluation predictions.

        Args:
            eval_preds (tuple[np.ndarray, np.ndarray]): The evaluation predictions.

        Returns:
            dict[str, float]: The computed metrics.
        """
        sacrebleu = evaluate.load("sacrebleu")
        if not self._fast_eval:
            chrf = evaluate.load("chrf")
            comet = evaluate.load("comet")
            bertscore = evaluate.load("bertscore")

        result = {}

        preds, labels, sources = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        if len(preds.shape) > 2:
            preds = np.argmax(preds, axis=-1)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # decode the predictions and labels in batches
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_sources = self.tokenizer.batch_decode(sources, skip_special_tokens=True)

        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)

        # compute the SacreBLEU score
        sacrebleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["sacrebleu"] = sacrebleu_result["score"]

        if not self._fast_eval:
            chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels)
            comet_score = comet.compute(
                predictions=decoded_preds, references=decoded_labels, sources=decoded_sources
            )
            bertscore_result = bertscore.compute(
                predictions=decoded_preds, references=decoded_labels, lang=self._target_lang
            )
            result["chrf"] = chrf_score["score"]
            result["comet"] = np.array(comet_score["scores"]).mean()
            result["bertscore"] = np.array(bertscore_result["f1"]).mean()

        # compute the average prediction length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

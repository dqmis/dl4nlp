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


class Trainer:
    def __init__(self, checkpoint: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        self._data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=checkpoint)

    def train(self, training_args: Seq2SeqTrainingArguments, dataset: DatasetType) -> None:
        trainer = Seq2SeqTrainer(
            model=self._model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=self._data_collator,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

    def _postprocess_text(
        self, preds: list[str], labels: list[str]
    ) -> tuple[list[str], list[str]]:
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]  # type: ignore

        return preds, labels

    def _compute_metrics(self, eval_preds: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        metric = evaluate.load("sacrebleu")

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

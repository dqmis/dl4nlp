from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments

from src.trainer.trainer import Trainer
from src.trainer.utils import preprocess_function


def main() -> None:
    trainer = Trainer("google-t5/t5-small")

    books = load_dataset("opus_books", "en-fr")
    books = books["train"].shuffle(seed=42).select(range(1000))
    books = books.train_test_split(test_size=0.2)

    tokenized_books = books.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "source_lang": "en",
            "tokenizer": trainer.tokenizer,
            "target_lang": "fr",
            "prefix": "translate English to French: ",
        },
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="out/my_awesome_opus_books_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
    )

    trainer.train(training_args, tokenized_books)


if __name__ == "__main__":
    main()

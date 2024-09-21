from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments

from src.trainer import Trainer
from src.utils import preprocess_function, load_config


def main() -> None:
    # Load the training configuration
    train_config = load_config("../config/train_config.yaml")

    trainer = Trainer(train_config["checkpoint"])

    books = load_dataset("opus_books", "en-fr")
    books = books["train"].shuffle(seed=42).select(range(1000))
    books = books.train_test_split(test_size=0.2)

    # Apply the preprocess function to dataset
    tokenized_books = books.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": trainer.tokenizer,
            "source_lang": train_config["source_lang"],
            "target_lang": train_config["target_lang"],
            "prefix": train_config["prefix"],
        },
    )

    # Define the training hyperparameters by calling Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(**train_config["training_args"])

    # Train the model using Seq2SeqTrainer
    trainer.train(training_args, tokenized_books)


if __name__ == "__main__":
    main()

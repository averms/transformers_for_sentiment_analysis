import datasets
import transformers

import common

imdb = datasets.load_dataset(
    "parquet",
    split="train",
    data_files="imdb.parquet",
).train_test_split(0.5)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased",
)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
)

imdb_tokenized = imdb.map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

for batch_size in (2, 4, 8):
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            "distilbert-imdb",
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            num_train_epochs=3,
        ),
        train_dataset=imdb_tokenized["train"],
        eval_dataset=imdb_tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=common.compute_metrics,
    )
    print(trainer.train())
    print(trainer.evaluate())

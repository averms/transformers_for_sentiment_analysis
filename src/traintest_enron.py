import datasets
import transformers

import common

enron_spam = datasets.load_dataset("SetFit/enron_spam", split="train+test")

# Remove useless columns and limit to 10,000 observations.
enron_clean = enron_spam.remove_columns(
    ["message_id", "label_text", "date", "subject", "message"],
).take(10_000)

# Load BERT-Tiny
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2",
)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2",
)

# Tokenize and truncate
enron_tokenized = enron_clean.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=512),
    batched=True,
)

# Set up padder
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

# Split into train and test sets
enron_split = enron_tokenized.train_test_split(0.5)

# Try different learning rates
for rate in (5e-4, 2e-5, 5e-5):
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            "bert-small-imdb",
            learning_rate=rate,
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            num_train_epochs=4,
        ),
        train_dataset=enron_split["train"],
        eval_dataset=enron_split["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=common.compute_metrics,
    )
    print(trainer.train())
    print(trainer.evaluate())

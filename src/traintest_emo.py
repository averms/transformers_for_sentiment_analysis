import datasets
import transformers

import common

emotion = datasets.load_dataset("dair-ai/emotion", "unsplit")

# Limit to 10,000 observations.
emotion_trimmed = emotion["train"].shuffle().take(10_000)

# Load BERT-Tiny
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2",
)

# There are 6 labels from sadness to surprise.
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2",
    num_labels=6,
)

# Tokenize and truncate
emotion_tokenized = emotion_trimmed.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=300),
    batched=True,
)

# Set up padder
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

# Split into train and test sets
emotion_split = emotion_tokenized.train_test_split(0.5)

# Try different learning rates
for decay in (0, 0.01, 0.10):
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            "bert-small-imdb",
            learning_rate=5e-4,
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            num_train_epochs=4,
            weight_decay=decay,
        ),
        train_dataset=emotion_split["train"],
        eval_dataset=emotion_split["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=common.compute_metrics,
    )
    print(trainer.train())
    print(trainer.evaluate())

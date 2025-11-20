import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os


# -------------------------
# Global seed for reproducibility
# -------------------------
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------
# Tokenization Function
# -------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# -------------------------
# Core Training Loop
# -------------------------
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

    print("Training completed...")
    print("Saving model...")
    model.save_pretrained(save_dir)


# -------------------------
# Evaluation Function
# -------------------------
def do_eval(eval_dataloader, model_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    out_f = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        metric.add_batch(
            predictions=predictions,
            references=batch["labels"]
        )

        # write predictions
        for pred, label in zip(predictions, batch["labels"]):
            out_f.write(f"{pred.item()}\n")
            out_f.write(f"{label.item()}\n")

    out_f.close()
    return metric.compute()


# -------------------------
# Q3: Create Augmented Dataloader (5000 transformed samples)
# -------------------------
def create_augmented_dataloader(args, dataset):

    # Raw train set ("text", "label")
    train_raw = dataset["train"]

    # Randomly pick 5000 samples
    num_aug = min(5000, len(train_raw))
    aug_subset = train_raw.shuffle(seed=42).select(range(num_aug))

    # Apply custom transformation (from utils.py)
    aug_transformed = aug_subset.map(
        custom_transform,
        load_from_cache_file=False
    )

    # Combine original + transformed
    combined_train = concatenate_datasets([train_raw, aug_transformed])

    # Tokenize combined dataset
    tokenized_combined = combined_train.map(
        tokenize_function,
        batched=True,
        load_from_cache_file=False
    )

    tokenized_combined = tokenized_combined.remove_columns(["text"])
    tokenized_combined = tokenized_combined.rename_column("label", "labels")
    tokenized_combined.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_combined,
        shuffle=True,
        batch_size=args.batch_size
    )

    return train_dataloader


# -------------------------
# Create transformed test dataloader (for Q2/Q3)
# -------------------------
def create_transformed_dataloader(args, dataset, debug_transformation):

    if debug_transformation:
        small = dataset["test"].shuffle(seed=42).select(range(5))
        transformed = small.map(custom_transform, load_from_cache_file=False)

        for i in range(5):
            print("Original Example", i)
            print(small[i])
            print("\n")
            print("Transformed Example", i)
            print(transformed[i])
            print("=" * 30)
        exit()

    transformed = dataset["test"].map(
        custom_transform,
        load_from_cache_file=False
    )

    tokenized = transformed.map(
        tokenize_function,
        batched=True,
        load_from_cache_file=False
    )

    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    eval_dataloader = DataLoader(tokenized, batch_size=args.batch_size)
    return eval_dataloader


# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_augmented", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_transformed", action="store_true")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true")
    parser.add_argument("--debug_transformation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    global device
    global tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Load raw dataset
    dataset = load_dataset("imdb")

    # Tokenize original dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # Debug mode: use small samples
    if args.debug_train:
        train_dataloader = DataLoader(
            tokenized_dataset["train"].shuffle(seed=42).select(range(4000)),
            shuffle=True,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["test"].shuffle(seed=42).select(range(1000)),
            batch_size=args.batch_size,
        )
        print("Debug training...")
        print("len(train_dataloader):", len(train_dataloader))
        print("len(eval_dataloader):", len(eval_dataloader))
    else:
        train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["test"],
            batch_size=args.batch_size,
        )
        print("Actual training...")
        print("len(train_dataloader):", len(train_dataloader))
        print("len(eval_dataloader):", len(eval_dataloader))

    # -------------------------
    # Train on original dataset
    # -------------------------
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out")
        args.model_dir = "./out"

    # -------------------------
    # Q3: Train augmented dataset
    # -------------------------
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2
        )
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        args.model_dir = "./out_augmented"

    # -------------------------
    # Evaluate on original test set
    # -------------------------
    if args.eval:
        outfile = os.path.basename(args.model_dir) + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, outfile)
        print("Score:", score)

    # -------------------------
    # Evaluate on transformed test set (Q2/Q3)
    # -------------------------
    if args.eval_transformed:
        outfile = os.path.basename(args.model_dir) + "_transformed.txt"
        eval_transformed = create_transformed_dataloader(
            args, dataset, args.debug_transformation
        )
        score = do_eval(eval_transformed, args.model_dir, outfile)
        print("Score:", score)

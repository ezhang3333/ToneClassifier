import torch
import numpy as np
import torch.utils.data
from transformers import DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification


# Dont modify 
MODEL_NAME = "gpt2"
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
YES_ID, NO_ID = TOKENIZER.encode(" yes")[0], TOKENIZER.encode(" no")[0]
MAX_LEN = 200  # maximum total length (prompt + answer + pads)


# Read TSV file in form: <label>\t<review text>\n
def load_classification_data(path="classification_train.txt"):
    records = []
    with open(path, encoding="utf‑8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad line {line_no} in {path!r}: {line!r}")
            label_str, text = parts
            records.append({"label": int(label_str), "content": text})
    return Dataset.from_list(records)


# Construct the prompt used for zero-shot sentiment prediction
def prompt_zero(text):
    return (
        f"Review: {text}\n"
        "Is the sentiment of the review positive?\n"
        "Answer yes or no: "
    )


def eval_seq2seq(ds, lm):
    lm.eval()
    dev = next(lm.parameters()).device
    total_nll, correct = 0.0, 0

    for ex in tqdm(ds, desc="seq2seq‑eval", leave=False):
        prompt = prompt_zero(ex["content"])
        inputs = TOKENIZER(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = lm(**inputs)
            logits = outputs.logits[0, -1]
        probs = torch.softmax(logits[[YES_ID, NO_ID]], dim=0).cpu().numpy()
        p_yes, p_no = probs

        true_label = "positive" if ex["label"] == 1 else "negative"
        predicted_label = "positive" if p_yes > p_no else "negative"
        p_true = p_yes if true_label == "positive" else p_no

        total_nll -= math.log(max(p_true, 1e-12))
        correct += int(predicted_label == true_label)

    avg_nll = total_nll / len(ds)
    accuracy = correct / len(ds)
    return avg_nll, accuracy


def eval_clf(ds, model, tokenize_clf):
    model.eval()
    dev = next(model.parameters()).device

    tok_ds = ds.map(
        tokenize_clf,
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
    )

    loader = torch.utils.data.DataLoader(
        tok_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(TOKENIZER),
    )

    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="clf‑eval", leave=False):
            labels = batch["labels"].to(dev)
            outputs = model(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev)
            )
            total_loss += ce(outputs.logits, labels).item()
            correct += (outputs.logits.argmax(-1) == labels).sum().item()

    avg_loss = total_loss / len(ds)
    accuracy = correct / len(ds)
    return avg_loss, accuracy


def eval_model(model, tokenize_clf=None):
    test_set = load_classification_data("classification_train.txt")

    if isinstance(model, GPT2LMHeadModel):
        return eval_seq2seq(test_set, model)

    if isinstance(model, GPT2ForSequenceClassification):
        assert tokenize_clf is not None, "tokenize_clf must be provided for classification head"
        return eval_clf(test_set, model, tokenize_clf)

    raise ValueError(
        f"Unsupported model type: {model.__class__.__name__}. "
        "Expected GPT2LMHeadModel or GPT2ForSequenceClassification."
    )
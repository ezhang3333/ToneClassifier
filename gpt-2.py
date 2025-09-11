import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from typing import Tuple, Dict
from tqdm import tqdm
from datasets import load_dataset, Dataset
import math
import random
from gpt_2_utils import (
    prompt_zero,
    load_classification_data,
    eval_model,
    TOKENIZER,
    MAX_LEN,
    YES_ID,
    NO_ID,
)


# Q-learning
def calculate_q_update(q_value, reward, next_max_q, alpha, gamma):
    # temporal‐difference error
    td_error = reward + gamma * next_max_q - q_value
    return q_value + alpha * td_error


# Greedy selection of highest Q-value
def select_best_action(q_values):
    return int(np.argmax(q_values))


# Trainig Q-table in 2x2 grid world
def simple_training_loop(episodes, alpha, gamma):
    q_table = np.zeros((4, 2))
    
    def step(state, action):
        if action == 0 and (state % 2) == 0:
            return state + 1
        if action == 1 and state < 2:
            return state + 2
        return state

    def reward_for(state):
        return 10 if state == 3 else -1

    for _ in range(episodes):
        state = 0
        while state != 3:
            action = select_best_action(q_table[state])
            next_state = step(state, action)
            r = reward_for(next_state)
            next_max = np.max(q_table[next_state])
            q_table[state, action] = calculate_q_update(
                q_table[state, action], r, next_max, alpha, gamma
            )
            state = next_state

    return q_table


SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED);  random.seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
# Do Not modify the following Constants
MODEL_NAME = "gpt2"
TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
YES_ID, NO_ID = TOKENIZER.encode(" yes")[0], TOKENIZER.encode(" no")[0]
MAX_LEN = 200   # maximum total length (prompt + answer + pads)


def build_prompt_with_answer(ex):
    prompt = prompt_zero(ex["content"])
    answer = " yes" if ex["label"] == 1 else " no"
    return {"text": prompt + answer}


# Tokenize prompt+answer for causal LM training.  Truncate from the front
# to MAX_LEN, pad to MAX_LEN, and mask all labels except the final token.
def tokenize_seq2seq(batch):
    """
    Tokenize prompt+answer for causal LM training.  Truncate from the front
    to MAX_LEN, pad to MAX_LEN, and mask all labels except the final token.
    """
    input_ids, attention_mask, labels = [], [], []
    for text, label in zip(batch["content"], batch["label"]):
        # build the full text
        full = prompt_zero(text) + (" yes" if label == 1 else " no")

        tok = TOKENIZER.encode(full, add_special_tokens=False)
        # truncate from front
        if len(tok) > MAX_LEN:
            tok = tok[-MAX_LEN:]
        # pad
        pad_len = MAX_LEN - len(tok)
        ids = tok + [TOKENIZER.pad_token_id] * pad_len
        mask = [1] * len(tok) + [0] * pad_len
        # labels: ignore all but final actual token
        lab = [-100] * MAX_LEN
        lab[len(tok) - 1] = tok[-1]
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# Fine-tune GPT2LMHeadModel in a causal seq2seq style
def train_seq2seq(train_ds, lm: GPT2LMHeadModel, seed=SEED):
    # Split into train/validation
    train_test = train_ds.train_test_split(test_size=0.1, seed=seed)
    train_ds, val_ds = train_test["train"], train_test["test"]

    tok_train = train_ds.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=[c for c in train_ds.column_names]
    )
    tok_val = val_ds.map(
        tokenize_seq2seq,
        batched=True,
        remove_columns=[c for c in val_ds.column_names]
    )

    collator = DataCollatorWithPadding(TOKENIZER)

    args = TrainingArguments(
        output_dir="./seq2seq",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,            # smaller LR
        weight_decay=0.01,
        num_train_epochs=3,            # train longer
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",              # disable wandb if not used
    )

    trainer = Trainer(
        model=lm,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=collator,
    )

    trainer.train()
    return trainer


# Tokenize for classification
def tokenize_clf(batch):
    enc = TOKENIZER(
        batch["content"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,           # we let the DataCollator pad
        return_attention_mask=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": batch["label"],
    }


#Fine-tune GPT2ForSequenceClassification
def train_clf(train_ds: Dataset, model: GPT2ForSequenceClassification, seed=SEED):
    # Split into train/validation
    train_test = train_ds.train_test_split(test_size=0.1, seed=seed)
    train_ds, val_ds = train_test["train"], train_test["test"]

    tok_train = train_ds.map(
        tokenize_clf,
        batched=True,
        remove_columns=[c for c in train_ds.column_names]
    )
    tok_val = val_ds.map(
        tokenize_clf,
        batched=True,
        remove_columns=[c for c in val_ds.column_names]
    )

    collator = DataCollatorWithPadding(TOKENIZER)

    args = TrainingArguments(
        output_dir="./clf",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        data_collator=collator,
    )

    trainer.train()
    return trainer


def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    # (..., seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    # (..., seq_len, d_k)
    return torch.matmul(weights, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, L, _ = q.size()

        q_proj = self.W_q(q)  
        k_proj = self.W_k(k)
        v_proj = self.W_v(v)

        q_heads = q_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k_heads = k_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v_heads = v_proj.view(B, L, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        out_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads)

        out_concat = (
            out_heads
            .permute(0, 2, 1, 3)       
            .contiguous()
            .view(B, L, self.d_model)  
        )

        return self.W_o(out_concat)
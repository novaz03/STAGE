# Auto-generated from notebook; edit as Python module if needed.
# You can refactor functions into the package (evacmob) and import them in scripts/.

# %% [code] In[1]
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
#import contextily as ctxv
import osmnx as ox
from osmnx import graph_to_gdfs
from shapely.ops import unary_union
from shapely.geometry import MultiPoint
import logging 
import pickle
ox.settings.log_console = True

# %% [code] In[2]
df = pd.read_csv("../US_POI.csv")
df["SUB_CATEGORY"] = df["SUB_CATEGORY"].fillna("<null_val>")
df["TOP_CATEGORY"] = df["TOP_CATEGORY"].fillna("<null_val>")
df["LOCATION_NAME"] = df["LOCATION_NAME"].fillna("<null_val>")

# %% [code] In[3]
df.columns

# %% [code] In[4]
df["SUB_CATEGORY"][:5]

# %% [code] In[5]
def concat_columns(row):
    a = row["TOP_CATEGORY"]
    b = row["SUB_CATEGORY"]
    c = row["LOCATION_NAME"]
    # You can put any separator you like; here we just use a space or a special token:
    return a + "[sep]" + b + "[sep]" + c

# 2) Apply row-wise
df["concatenated"] = df.apply(concat_columns, axis=1)

# %% [code] In[6]
cols_to_keep = [
    "PLACEKEY",
    "LONGITUDE",      # “longtitude” in your prompt → correct column name is LONGITUDE
    "LATITUDE",       # “lattitude” → correct column name is LATITUDE
    "concatenated",
    "REGION",
    "LOCATION_NAME"
]

new_df = df[cols_to_keep]

# (4) Save to CSV (without the index column):
new_df.to_csv("poi_subset.csv", index=False)

# %% [code] In[7]
import pandas as pd
new_df = pd.read_csv("poi_subset.csv")

# %% [code] In[8]
mask = (
    (new_df["LONGITUDE"] >= -88.57)
    & (new_df["LONGITUDE"] <= 79.95)
    & (new_df["LATITUDE"] >= 24.45)
    & (new_df["LATITUDE"] <= 32.35)
)
FL_new_df = new_df[mask]
FL_new_df.to_csv("Hex_bound_POI.csv")

# %% [code] In[9]
from datasets import Dataset

# 2) Take just the “concatenated” column as a Python list of strings
texts = new_df["concatenated"].tolist()

# 3) Build a simple HF Dataset with one field "text"
tokenizer_ds = Dataset.from_dict({ "text": texts })

# %% [code] In[10]
from transformers import AutoTokenizer

# %% [code] In[11]

# 4) Load a pretrained tokenizer (e.g. BERT’s)
#base_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

# 5) Train-in-place on your “text” field

#new_tok = base_tok.train_new_from_iterator(
#    tokenizer_ds["text"],                # an iterator over all concatenated strings
#    vocab_size=base_tok.vocab_size + 5000,  # e.g. original BERT vocab + 5k new tokens
#    show_progress=True
#)

# 6) Save the extended tokenizer
#new_tok.save_pretrained("bert_extended_two_col_tok")

# %% [code] In[12]
tok = AutoTokenizer.from_pretrained("bert_extended_two_col_tok")
num_added = tok.add_special_tokens({
    "additional_special_tokens": ["<null_val>"]
})
print(f"Added {num_added} new special token(s).")
print(tok.tokenize("Hair, Nail, and Skin Care Services <null_val>"))
#print("ID for <null_val>:", tok.convert_tokens_to_ids("Hair, Nail, and Skin Care Services"))

# %% [code] In[13]
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    LlamaTokenizer
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% [code] In[14]
from huggingface_hub import login

# Replace "hf_…" with your actual Hugging Face API token
login("hf_")

# %% [code] In[15]
tokenizer = tok
OUTPUT_DIR = "./models/lora-3.2-1b-lm-finetuned"
print("Tokenizing test string:",
      tok.tokenize("Hair, Nail, and Skin Care Services <null_val>"))

# 2) Load the causal-LM model (Gemma-3-1b-it) and resize its embeddings
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="eager"
).to(device)
model.resize_token_embeddings(len(tok))
raw_ds = tokenizer_ds
print(f"raw_ds (from concatenated column): {len(raw_ds)} examples")
print("sample[0][:200]:", raw_ds[0]["text"], "\n")

# %% [code] In[16]
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    attn_implementation="eager"
).to(device)

# (If you previously resized embeddings, make sure the tokenizer & model match:)
model.resize_token_embeddings(len(tok))

# %% [code] In[17]

concatenated_texts = [
    "The two texts <sep> are somewhat <sep> similar",
    "The two texts <sep> are somewhat <sep> not anyhow similar"
]

for txt in concatenated_texts:
    enc = tok(txt, return_tensors="pt", truncation=True, padding="longest")
    print(txt)
    print(" input_ids:", enc.input_ids[0].tolist())
    print(" tokens:   ", tok.convert_ids_to_tokens(enc.input_ids[0]))
    print()

# %% [code] In[18]
import torch

batch = tok(
    concatenated_texts,
    return_tensors="pt",
    truncation=True,
    max_length=256,      # Adjust as needed; 32 was very short.
    padding="longest",   # Pad to the longest sequence in the batch
    return_attention_mask=True # Ensure attention_mask is returned
).to(device)

# Forward‐pass
with torch.no_grad(): # Disable gradient calculations for inference
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        output_hidden_states=True,  # You need this to access all hidden layers
        return_dict=True
    )

# Get the hidden states from the last layer
final_hidden = outputs.hidden_states[-1]  # shape: (batch_size, seq_len, hidden_size)

# --- Implement Mean Pooling (Recommended for Gemma sequence representation) ---
attention_mask = batch.attention_mask # shape: (batch_size, seq_len)

# Expand attention_mask to match the shape of token_embeddings for broadcasting
mask_expanded = attention_mask.unsqueeze(-1).expand(final_hidden.size()).float()
# shape: (batch_size, seq_len, hidden_size)

# Zero out the embeddings of padding tokens
sum_embeddings = torch.sum(final_hidden * mask_expanded, dim=1)
# shape: (batch_size, hidden_size)

# Count the number of non-padding tokens in each sequence
# Clamp to avoid division by zero for sequences that might be all padding (shouldn't happen with proper input)
sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
# shape: (batch_size, hidden_size)

# Compute the mean
mean_pooled_embeddings = sum_embeddings / sum_mask
# mean_pooled_embeddings shape: (batch_size, hidden_size)

# Now mean_pooled_embeddings[i] is the single fixed-length vector for input sequence i.

# Example usage for comparison (if you have at least two items in your batch)
if mean_pooled_embeddings.shape[0] >= 2:
    embedding_0 = mean_pooled_embeddings[0].cpu().numpy()
    embedding_1 = mean_pooled_embeddings[1].cpu().numpy()

    print("Length of each Mean Pooled embedding:", len(embedding_0), len(embedding_1))
    print("Mean Pooled embedding for first example (first 10 dims):\n", embedding_0[:10], "…\n")
    print("Mean Pooled embedding for second example (first 10 dims):\n", embedding_1[:10], "…\n")

    cos_sim_mean = torch.nn.functional.cosine_similarity(
        mean_pooled_embeddings[0], mean_pooled_embeddings[1], dim=0
    ).item()
    print(f"Cosine similarity (Mean Pooled embeddings) = {cos_sim_mean:.4f}")

elif mean_pooled_embeddings.shape[0] == 1:
    embedding_0 = mean_pooled_embeddings[0].cpu().numpy()
    print("Length of the single Mean Pooled embedding:", len(embedding_0))
    print("Mean Pooled embedding (first 10 dims):\n", embedding_0[:10], "…\n")
else:
    print("Batch size is 0, no embeddings to display.")

# The 'mean_pooled_embeddings' tensor contains one fixed-length vector per input string.

# %% [code] In[19]
def tokenize_fn(examples):
    return tok(examples["text"], return_attention_mask=True)

tok_ds = raw_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)
print(f"⮑ tok_ds: {len(tok_ds)} examples")
print("  sample[0]:", {k: tok_ds[0][k][:10] for k in ("input_ids", "attention_mask")}, "\n")

# 9) Group tokenized outputs into fixed‐length blocks (for causal LM)
block_size = 64

def group_texts(examples):
    # Flatten all input_ids and then split in chunks of size block_size
    all_ids = sum(examples["input_ids"], [])
    total_length = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_length]

    chunks = [
        all_ids[i : i + block_size]
        for i in range(0, total_length, block_size)
    ]
    masks = [[1] * block_size for _ in chunks]
    return {
        "input_ids": chunks,
        "attention_mask": masks,
    }

# %% [code] In[20]
lm_ds = tok_ds.map(
    group_texts,
    batched=True,
    batch_size=1024,
    num_proc=None,
    remove_columns=tok_ds.column_names   # ← important!
)
print(f"⮑ lm_ds: {len(lm_ds)} blocks\n")

if len(lm_ds) == 0:
    raise RuntimeError("No LM blocks were created—check block_size or your input data!")
else:
    first_block = lm_ds[0]["input_ids"]
    print("  lm_ds[0]['input_ids'][:10]:", first_block[:10], "(len)", len(first_block))

# %% [code] In[21]
with open("./llm_ds.pickle", "wb") as f:
    pickle.dump(lm_ds,f)

# %% [code] In[22]
with open("./llm_ds.pickle", "rb") as f:
    lm_ds = pickle.load(f)

# %% [code] In[23]
# 11) Set up LoRA for causal-LM fine-tuning
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_cfg)
#print(model)

# %% [code] In[24]
import os
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tok,
    mlm=False
)

# 13) Training arguments
OUTPUT_DIR = "./models/lora-3.2-1b-lm-finetuned-with-null"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#print(model)

# %% [code] In[25]
print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)

# %% [code] In[26]
import torch
print(torch.cuda.is_available())
def add_labels(examples):
    examples["labels"] = examples["input_ids"]
    return examples

# Apply this function to your dataset
lm_ds = lm_ds.map(
    add_labels,
    batched=True,  # Process examples in batches
    num_proc=None  # Set to your desired number of CPU cores for parallel processing
)

# Optional: Verify the dataset structure
print(lm_ds)

# %% [code] In[27]
print("Model device:", next(model.parameters()).device)

# %% [code] In[28]
from peft import LoraConfig, get_peft_model
#model = get_peft_model(model, lora_cfg)

# 4. Apply torch.compile to the PEFT-wrapped model
# This is the crucial step to enable compilation for speed, and it should be *after* PEFT

training_args = TrainingArguments(
    output_dir="checkpoint-dir",
    per_device_train_batch_size=128*2,
    gradient_accumulation_steps=64,
    num_train_epochs=1,
    learning_rate=3e-4,
    dataloader_num_workers=16,   # or 8 if you have spare CPU cores
    fp16=True,
    logging_steps=10,
    save_steps=30,
    save_total_limit=3,
    optim="adamw_torch_fused",
    label_names=["labels"],
)
#model_compiled = torch.compile(model) # Use a temporary variable first

# 2. Then apply PEFT
#model = get_peft_model(model_compiled, lora_cfg)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_ds,             # lm_ds still has columns "input_ids", "attention_mask"
    data_collator=data_collator,
    tokenizer=tok,             # or processing_class=tokenizer
)
trainer.train()

# %% [code] In[29]
lm_ds


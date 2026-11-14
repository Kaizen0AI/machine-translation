import os
from functools import partial
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import torch
from torch.utils.data import DataLoader

TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "custom_bpe_tokenizer")  # Default directory for saving/loading the tokenizer

def load_dataset_from_hf():
    # Load the dataset from the Hugging Face Hub
    return load_dataset("bentrevett/multi30k")

def get_or_train_tokenizer(dataset, save_dir=TOKENIZER_DIR):
    # load from disk if already trained
    if os.path.exists(save_dir):
        print(f"Loading existing tokenizer from {save_dir}...")
        return PreTrainedTokenizerFast.from_pretrained(save_dir)

    print("Training new BPE tokenizer...")
    # Define the special tokens dictionary configuration 
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    # Initialize a tokenizer with a BPE model and whitespace pre-tokenizer
    # (We pass the special tokens directly to the BPE object to enforce strict ID ordering)
    raw_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    raw_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=12000
    )

    # Train the tokenizer on the dataset using a batch iterator
    def batch_iterator():
        for example in dataset["train"]:
            yield example["en"]
            yield example["de"]

    raw_tokenizer.train_from_iterator(batch_iterator(), trainer)

    # Wrap tokenizer in a PreTrainedTokenizerFast 
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>"
    )

    # save vocabulary and config for future runs and inference
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    return tokenizer


def encode_sentences(sentences, tokenizer, max_length=128):
    encoded = tokenizer(
        sentences,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 2
    )

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    sequences = [
        [bos_id] + ids + [eos_id] for ids in encoded["input_ids"]
    ]

    attention_masks = [[1] * len(seq) for seq in sequences]

    return sequences, attention_masks

# Preprocess dataset
def preprocess_function(examples, tokenizer):
    inputs, input_masks = encode_sentences(examples["en"], tokenizer)
    targets, _ = encode_sentences(examples["de"], tokenizer)

    return {
        "input_ids": inputs,
        "attention_mask": input_masks,
        "labels": targets
    }


def seq2seq_collator(batch, pad_val):
    # Extract lists from the batch dictionaries
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    
    # Track original lengths for pack_padded_sequence 
    input_lengths = torch.tensor([len(ids) for ids in input_ids])
    target_lengths = torch.tensor([len(lbls) for lbls in labels])

    # Use tokenizer.pad_token_id dynamically instead of hardcoded 0
    #pad_val = tokenizer.pad_token_id
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_val)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_val)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0) # Mask padding is always 0

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "input_lengths": input_lengths,
        "labels": padded_labels,
        "target_lengths": target_lengths
    }

def get_dataloaders():
    # Load dataset and tokenizer once
    dataset = load_dataset_from_hf()
    tokenizer = get_or_train_tokenizer(dataset)

    # pre-fill the tokenizer argument using partial
    # Hugging Face's .map() and PyTorch's DataLoader expect functions that only take one argument (examples or batch). 
    # We can use Python's partial utility to pre-fill the tokenizer argument before passing those functions in.
    preprocess_with_tok = partial(preprocess_function, tokenizer=tokenizer)
    collator_with_pad = partial(seq2seq_collator, pad_val=tokenizer.pad_token_id) # We pass tokenizer.pad_token_id into it so that the filler number matches the exact ID your vocabulary reserved for the <PAD> token

    # map across splits
    tokenized_datasets = dataset.map(
        preprocess_with_tok,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collator_with_pad
    )

    val_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=32,
        shuffle=False,
        collate_fn=collator_with_pad
    )

    # Return tokenizer as well so train.py knows the vocab_size and pad_token_id
    return train_dataloader, val_dataloader, tokenizer

'''
# Test a single batch execution to verify shapes
for batch in train_dataloader:
    print("\n--- Batch Verification Successful ---")
    print("English Matrix (input_ids):", batch["input_ids"].shape)          
    print("English Mask (attention_mask):", batch["attention_mask"].shape)  
    print("German Matrix (labels):", batch["labels"].shape)                 
    print("English Original Lengths:", batch["input_lengths"][:5])             
    break
    
Inference later: When you write your predict.py script later, you will simply run 
tokenizer = PreTrainedTokenizerFast.from_pretrained("./my_bpe_tokenizer")
 to decode the model's output IDs back into German words.
'''
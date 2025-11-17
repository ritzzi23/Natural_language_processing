#!/usr/bin/env python3
"""
Compute Q4 Statistics: Before and After Preprocessing
"""
from transformers import T5TokenizerFast
import os
from collections import Counter

def load_lines(path):
    """Load lines from a file"""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_statistics_before_preprocessing(data_folder='data'):
    """
    Compute Table 1: Statistics BEFORE preprocessing
    Uses raw data with T5 tokenizer (no prefix, no other preprocessing)
    """
    print("=" * 60)
    print("TABLE 1: Statistics BEFORE Preprocessing")
    print("=" * 60)
    
    # Initialize T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load raw data
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Tokenize without any preprocessing (no prefix, just raw tokenization)
    def tokenize_raw(text):
        """Tokenize raw text without any preprocessing"""
        encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
        return encoded['input_ids']
    
    # Compute statistics for train set
    train_nl_tokens = [tokenize_raw(nl) for nl in train_nl]
    train_sql_tokens = [tokenize_raw(sql) for sql in train_sql]
    
    train_nl_lengths = [len(tokens) for tokens in train_nl_tokens]
    train_sql_lengths = [len(tokens) for tokens in train_sql_tokens]
    
    # Collect all unique tokens for vocabulary
    train_nl_vocab = set()
    for tokens in train_nl_tokens:
        train_nl_vocab.update(tokens)
    
    train_sql_vocab = set()
    for tokens in train_sql_tokens:
        train_sql_vocab.update(tokens)
    
    # Compute statistics for dev set
    dev_nl_tokens = [tokenize_raw(nl) for nl in dev_nl]
    dev_sql_tokens = [tokenize_raw(sql) for sql in dev_sql]
    
    dev_nl_lengths = [len(tokens) for tokens in dev_nl_tokens]
    dev_sql_lengths = [len(tokens) for tokens in dev_sql_tokens]
    
    # Collect all unique tokens for vocabulary
    dev_nl_vocab = set()
    for tokens in dev_nl_tokens:
        dev_nl_vocab.update(tokens)
    
    dev_sql_vocab = set()
    for tokens in dev_sql_tokens:
        dev_sql_vocab.update(tokens)
    
    # Print results
    print(f"\n{'Statistics Name':<35} {'Train':<15} {'Dev':<15}")
    print("-" * 65)
    print(f"{'Number of examples':<35} {len(train_nl):<15} {len(dev_nl):<15}")
    print(f"{'Mean sentence length (NL)':<35} {sum(train_nl_lengths)/len(train_nl_lengths):<15.2f} {sum(dev_nl_lengths)/len(dev_nl_lengths):<15.2f}")
    print(f"{'Mean SQL query length':<35} {sum(train_sql_lengths)/len(train_sql_lengths):<15.2f} {sum(dev_sql_lengths)/len(dev_sql_lengths):<15.2f}")
    print(f"{'Vocabulary size (natural language)':<35} {len(train_nl_vocab):<15} {len(dev_nl_vocab):<15}")
    print(f"{'Vocabulary size (SQL)':<35} {len(train_sql_vocab):<15} {len(dev_sql_vocab):<15}")
    
    # Additional statistics
    print(f"\n{'Additional Statistics':<35} {'Train':<15} {'Dev':<15}")
    print("-" * 65)
    print(f"{'Max sentence length (NL)':<35} {max(train_nl_lengths):<15} {max(dev_nl_lengths):<15}")
    print(f"{'Max SQL query length':<35} {max(train_sql_lengths):<15} {max(dev_sql_lengths):<15}")
    print(f"{'Min sentence length (NL)':<35} {min(train_nl_lengths):<15} {min(dev_nl_lengths):<15}")
    print(f"{'Min SQL query length':<35} {min(train_sql_lengths):<15} {min(dev_sql_lengths):<15}")
    
    return {
        'train': {
            'num_examples': len(train_nl),
            'mean_nl_length': sum(train_nl_lengths)/len(train_nl_lengths),
            'mean_sql_length': sum(train_sql_lengths)/len(train_sql_lengths),
            'nl_vocab_size': len(train_nl_vocab),
            'sql_vocab_size': len(train_sql_vocab),
        },
        'dev': {
            'num_examples': len(dev_nl),
            'mean_nl_length': sum(dev_nl_lengths)/len(dev_nl_lengths),
            'mean_sql_length': sum(dev_sql_lengths)/len(dev_sql_lengths),
            'nl_vocab_size': len(dev_nl_vocab),
            'sql_vocab_size': len(dev_sql_vocab),
        }
    }

def compute_statistics_after_preprocessing(data_folder='data'):
    """
    Compute Table 2: Statistics AFTER preprocessing
    Uses processed data (with prefix and preprocessing from load_data.py)
    """
    print("\n" + "=" * 60)
    print("TABLE 2: Statistics AFTER Preprocessing")
    print("=" * 60)
    
    # Initialize T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load raw data
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Apply preprocessing (same as in load_data.py)
    prefix = "translate English to SQL: "
    
    def tokenize_processed(nl_text, sql_text):
        """Tokenize with preprocessing (prefix for NL, truncation)"""
        # NL with prefix
        nl_encoded = tokenizer(
            prefix + nl_text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        # SQL with truncation
        sql_encoded = tokenizer(
            sql_text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        return nl_encoded['input_ids'], sql_encoded['input_ids']
    
    # Compute statistics for train set
    train_nl_tokens = []
    train_sql_tokens = []
    for nl, sql in zip(train_nl, train_sql):
        nl_tokens, sql_tokens = tokenize_processed(nl, sql)
        train_nl_tokens.append(nl_tokens)
        train_sql_tokens.append(sql_tokens)
    
    train_nl_lengths = [len(tokens) for tokens in train_nl_tokens]
    train_sql_lengths = [len(tokens) for tokens in train_sql_tokens]
    
    # Collect all unique tokens for vocabulary
    train_nl_vocab = set()
    for tokens in train_nl_tokens:
        train_nl_vocab.update(tokens)
    
    train_sql_vocab = set()
    for tokens in train_sql_tokens:
        train_sql_vocab.update(tokens)
    
    # Compute statistics for dev set
    dev_nl_tokens = []
    dev_sql_tokens = []
    for nl, sql in zip(dev_nl, dev_sql):
        nl_tokens, sql_tokens = tokenize_processed(nl, sql)
        dev_nl_tokens.append(nl_tokens)
        dev_sql_tokens.append(sql_tokens)
    
    dev_nl_lengths = [len(tokens) for tokens in dev_nl_tokens]
    dev_sql_lengths = [len(tokens) for tokens in dev_sql_tokens]
    
    # Collect all unique tokens for vocabulary
    dev_nl_vocab = set()
    for tokens in dev_nl_tokens:
        dev_nl_vocab.update(tokens)
    
    dev_sql_vocab = set()
    for tokens in dev_sql_tokens:
        dev_sql_vocab.update(tokens)
    
    # Print results
    print(f"\n{'Statistics Name':<35} {'Train':<15} {'Dev':<15}")
    print("-" * 65)
    print(f"{'Mean sentence length (NL)':<35} {sum(train_nl_lengths)/len(train_nl_lengths):<15.2f} {sum(dev_nl_lengths)/len(dev_nl_lengths):<15.2f}")
    print(f"{'Mean SQL query length':<35} {sum(train_sql_lengths)/len(train_sql_lengths):<15.2f} {sum(dev_sql_lengths)/len(dev_sql_lengths):<15.2f}")
    print(f"{'Vocabulary size (natural language)':<35} {len(train_nl_vocab):<15} {len(dev_nl_vocab):<15}")
    print(f"{'Vocabulary size (SQL)':<35} {len(train_sql_vocab):<15} {len(dev_sql_vocab):<15}")
    
    # Additional statistics
    print(f"\n{'Additional Statistics':<35} {'Train':<15} {'Dev':<15}")
    print("-" * 65)
    print(f"{'Max sentence length (NL)':<35} {max(train_nl_lengths):<15} {max(dev_nl_lengths):<15}")
    print(f"{'Max SQL query length':<35} {max(train_sql_lengths):<15} {max(dev_sql_lengths):<15}")
    print(f"{'Min sentence length (NL)':<35} {min(train_nl_lengths):<15} {min(dev_nl_lengths):<15}")
    print(f"{'Min SQL query length':<35} {min(train_sql_lengths):<15} {min(dev_sql_lengths):<15}")
    
    return {
        'train': {
            'mean_nl_length': sum(train_nl_lengths)/len(train_nl_lengths),
            'mean_sql_length': sum(train_sql_lengths)/len(train_sql_lengths),
            'nl_vocab_size': len(train_nl_vocab),
            'sql_vocab_size': len(train_sql_vocab),
        },
        'dev': {
            'mean_nl_length': sum(dev_nl_lengths)/len(dev_nl_lengths),
            'mean_sql_length': sum(dev_sql_lengths)/len(dev_sql_lengths),
            'nl_vocab_size': len(dev_nl_vocab),
            'sql_vocab_size': len(dev_sql_vocab),
        }
    }

if __name__ == "__main__":
    print("\nComputing Q4 Statistics for Report\n")
    
    # Compute Table 1: Before preprocessing
    stats_before = compute_statistics_before_preprocessing()
    
    # Compute Table 2: After preprocessing
    stats_after = compute_statistics_after_preprocessing()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nTable 1 (Before) and Table 2 (After) statistics computed.")
    print("Copy the values above into your report tables.")
    print("\nNote: Number of examples is the same in both tables.")
    print("Other statistics may differ due to preprocessing (prefix, truncation).")


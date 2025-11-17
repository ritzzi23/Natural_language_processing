#!/bin/bash
# Test script to run on Lightning AI
# Usage: bash run_tests.sh

cd ~/content

echo "=========================================="
echo "TEST 1: Data Loading"
echo "=========================================="
python3 << 'EOF'
from load_data import load_t5_data
train_loader, dev_loader, test_loader = load_t5_data(batch_size=2, test_batch_size=2)
print(f"✓ Train: {len(train_loader)} batches")
print(f"✓ Dev: {len(dev_loader)} batches")
print(f"✓ Test: {len(test_loader)} batches")

batch = next(iter(train_loader))
encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
print(f"✓ Batch shapes:")
print(f"  encoder_ids: {encoder_ids.shape}")
print(f"  encoder_mask: {encoder_mask.shape}")
print(f"  decoder_inputs: {decoder_inputs.shape}")
print(f"  decoder_targets: {decoder_targets.shape}")
EOF

echo ""
echo "=========================================="
echo "TEST 2: Model Initialization"
echo "=========================================="
python3 << 'EOF'
from t5_utils import initialize_model
import argparse
import torch

args = argparse.Namespace()
args.finetune = True
model = initialize_model(args)
print(f"✓ Model initialized: {type(model).__name__}")
print(f"✓ Device: {next(model.parameters()).device}")

# Test forward pass
from load_data import load_t5_data
train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
batch = next(iter(train_loader))
device = next(model.parameters()).device
encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
encoder_ids = encoder_ids.to(device)
encoder_mask = encoder_mask.to(device)
decoder_inputs = decoder_inputs.to(device)

with torch.no_grad():
    outputs = model(input_ids=encoder_ids, attention_mask=encoder_mask, decoder_input_ids=decoder_inputs)
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {outputs.logits.shape}")
EOF

echo ""
echo "=========================================="
echo "TEST 3: Training Step"
echo "=========================================="
python3 << 'EOF'
from t5_utils import initialize_model, initialize_optimizer_and_scheduler
from load_data import load_t5_data
import argparse
import torch
import torch.nn as nn

args = argparse.Namespace()
args.finetune = True
args.learning_rate = 1e-3
args.weight_decay = 0
args.optimizer_type = 'AdamW'
args.scheduler_type = 'none'
args.max_n_epochs = 1
args.num_warmup_epochs = 0

model = initialize_model(args)
train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

model.train()
batch = next(iter(train_loader))
device = next(model.parameters()).device
encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
encoder_ids = encoder_ids.to(device)
encoder_mask = encoder_mask.to(device)
decoder_inputs = decoder_inputs.to(device)
decoder_targets = decoder_targets.to(device)

optimizer.zero_grad()
outputs = model(input_ids=encoder_ids, attention_mask=encoder_mask, decoder_input_ids=decoder_inputs)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
targets_flat = decoder_targets.view(-1)
loss = criterion(logits_flat, targets_flat)
loss.backward()
optimizer.step()

print(f"✓ Training step successful")
print(f"  Loss: {loss.item():.4f}")
EOF

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="


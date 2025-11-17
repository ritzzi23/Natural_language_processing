# Testing Guide for T5 Pipeline on Lightning AI

## Step 1: Connect to Lightning AI
```bash
ssh s_01ka47crpgzbah2cgt4hyrhtd8@ssh.lightning.ai
cd ~/content
```

## Step 2: Test Data Loading
```bash
python3 -c "
from load_data import load_t5_data
train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=4)
print(f'Train: {len(train_loader)} batches')
print(f'Dev: {len(dev_loader)} batches')
print(f'Test: {len(test_loader)} batches')

# Test a batch
batch = next(iter(train_loader))
encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
print(f'Batch shapes:')
print(f'  encoder_ids: {encoder_ids.shape}')
print(f'  encoder_mask: {encoder_mask.shape}')
print(f'  decoder_inputs: {decoder_inputs.shape}')
print(f'  decoder_targets: {decoder_targets.shape}')
print('✓ Data loading works!')
"
```

## Step 3: Test Model Initialization
```bash
python3 -c "
from t5_utils import initialize_model
import argparse

args = argparse.Namespace()
args.finetune = True
model = initialize_model(args)
print(f'✓ Model initialized: {type(model).__name__}')
print(f'  Device: {next(model.parameters()).device}')

# Test forward pass
from load_data import load_t5_data
train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
batch = next(iter(train_loader))
encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
device = next(model.parameters()).device
encoder_ids = encoder_ids.to(device)
encoder_mask = encoder_mask.to(device)
decoder_inputs = decoder_inputs.to(device)

import torch
with torch.no_grad():
    outputs = model(input_ids=encoder_ids, attention_mask=encoder_mask, decoder_input_ids=decoder_inputs)
    print(f'✓ Forward pass successful')
    print(f'  Logits shape: {outputs.logits.shape}')
"
```

## Step 4: Test Training Step
```bash
python3 -c "
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

print(f'✓ Training step successful')
print(f'  Loss: {loss.item():.4f}')
"
```

## Step 5: Run Full Test Script
```bash
python3 test_pipeline.py
```

## Step 6: Test Small Training Run (1-2 epochs)
```bash
python3 train_t5.py \
  --finetune \
  --learning_rate 1e-3 \
  --batch_size 4 \
  --test_batch_size 4 \
  --max_n_epochs 2 \
  --patience_epochs 10 \
  --experiment_name test_run
```

## Expected Results:
- Data loading should work without errors
- Model should initialize and forward pass should work
- Training step should complete without errors
- Small training run should complete 1-2 epochs

## Common Issues:
1. **Out of memory**: Reduce batch_size
2. **Import errors**: Make sure all files are synced
3. **File not found**: Check that data files exist in ~/content/data/
4. **CUDA errors**: Check GPU availability with `nvidia-smi`


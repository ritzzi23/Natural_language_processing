#!/usr/bin/env python3
"""
Test script for T5 pipeline
"""
import torch
from load_data import load_t5_data
from t5_utils import initialize_model
import argparse

def test_data_loading():
    print("=" * 50)
    print("TEST 1: Data Loading")
    print("=" * 50)
    try:
        train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=4)
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Dev loader: {len(dev_loader)} batches")
        print(f"✓ Test loader: {len(test_loader)} batches")
        
        # Test a batch from train loader
        batch = next(iter(train_loader))
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch
        print(f"✓ Train batch shapes:")
        print(f"  encoder_ids: {encoder_ids.shape}")
        print(f"  encoder_mask: {encoder_mask.shape}")
        print(f"  decoder_inputs: {decoder_inputs.shape}")
        print(f"  decoder_targets: {decoder_targets.shape}")
        
        # Test a batch from test loader
        test_batch = next(iter(test_loader))
        test_encoder_ids, test_encoder_mask, test_initial_decoder_inputs = test_batch
        print(f"✓ Test batch shapes:")
        print(f"  encoder_ids: {test_encoder_ids.shape}")
        print(f"  encoder_mask: {test_encoder_mask.shape}")
        
        print("✓ Data loading test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    print("=" * 50)
    print("TEST 2: Model Initialization")
    print("=" * 50)
    try:
        # Create minimal args
        args = argparse.Namespace()
        args.finetune = True
        
        model = initialize_model(args)
        print(f"✓ Model initialized")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Device: {next(model.parameters()).device}")
        
        # Test forward pass
        train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
        batch = next(iter(train_loader))
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
        encoder_ids = encoder_ids.to(next(model.parameters()).device)
        encoder_mask = encoder_mask.to(next(model.parameters()).device)
        decoder_inputs = decoder_inputs.to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_inputs,
            )
            print(f"✓ Forward pass successful")
            print(f"  Logits shape: {outputs.logits.shape}")
        
        print("✓ Model initialization test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Model initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    print("=" * 50)
    print("TEST 3: Training Step")
    print("=" * 50)
    try:
        # Create minimal args
        args = argparse.Namespace()
        args.finetune = True
        args.learning_rate = 1e-3
        args.weight_decay = 0
        args.optimizer_type = "AdamW"
        args.scheduler_type = "none"
        args.max_n_epochs = 1
        args.num_warmup_epochs = 0
        
        model = initialize_model(args)
        train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
        
        from t5_utils import initialize_optimizer_and_scheduler
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        
        # Test one training step
        model.train()
        batch = next(iter(train_loader))
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, _ = batch
        encoder_ids = encoder_ids.to(next(model.parameters()).device)
        encoder_mask = encoder_mask.to(next(model.parameters()).device)
        decoder_inputs = decoder_inputs.to(next(model.parameters()).device)
        decoder_targets = decoder_targets.to(next(model.parameters()).device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=encoder_ids,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_inputs,
        )
        
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
        targets_flat = decoder_targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print("✓ Training step test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Training step test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("T5 PIPELINE TEST SUITE")
    print("=" * 50 + "\n")
    
    results = []
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("Training Step", test_training_step()))
    
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests PASSED! Pipeline is ready.")
    else:
        print("\n✗ Some tests FAILED. Please fix issues before training.")
    
    exit(0 if all_passed else 1)


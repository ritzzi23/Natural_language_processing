# Q7: Model Checkpoint Location

## Where is the model checkpoint used to generate outputs?

The model checkpoint used to generate test outputs is located at:

**Path**: `checkpoints/ft_experiments/baseline/best_model.pt`

### Checkpoint Details:

1. **Checkpoint Directory Structure**:
   ```
   checkpoints/
   └── ft_experiments/          # ft = fine-tuned, scr = from-scratch
       └── baseline/             # experiment_name (from --experiment_name argument)
           ├── best_model.pt    # Best model (highest Record F1 on dev set)
           └── checkpoint.pt     # Latest checkpoint (saved every epoch)
   ```

2. **How it's saved**:
   - During training, the best model (highest Record F1 on development set) is saved to `checkpoints/ft_experiments/baseline/best_model.pt`
   - This happens in `train_t5.py` line 96: `save_model(checkpoint_dir, model, best=True)`
   - The checkpoint contains:
     - `model_state_dict`: The trained model weights
     - `model_config`: The model configuration

3. **How it's loaded for test inference**:
   - In `train_t5.py` line 271: `model = load_model_from_checkpoint(args, best=True)`
   - This loads the best model checkpoint before generating test set outputs
   - The loaded model is then used in `test_inference()` function (line 290) to generate:
     - `results/t5_ft_ft_experiment_test.sql`
     - `records/t5_ft_ft_experiment_test.pkl`

4. **Checkpoint Location on Lightning AI**:
   - The checkpoint was saved during training on Lightning AI platform
   - Path: `~/content/checkpoints/ft_experiments/baseline/best_model.pt`
   - The checkpoint corresponds to **Epoch 16**, which achieved the best Record F1 of **83.8%** on the development set

5. **For Submission**:
   - The checkpoint itself is **NOT** required for submission
   - Only the generated output files are needed:
     - `results/t5_ft_ft_experiment_test.sql` (or renamed to `t5_ft_experiment_test.sql`)
     - `records/t5_ft_ft_experiment_test.pkl` (or renamed to `t5_ft_experiment_test.pkl`)
   - These files were generated using the best model checkpoint and are already available locally

### Summary:

- **Checkpoint path**: `checkpoints/ft_experiments/baseline/best_model.pt`
- **Best epoch**: Epoch 16 (Record F1: 83.8% on dev set)
- **Used to generate**: Test set SQL queries and records
- **Submission files**: `t5_ft_experiment_test.sql` and `t5_ft_experiment_test.pkl` (already generated)


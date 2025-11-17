# Q5: T5 Fine-tuning Documentation

## Table 3: Details of the best-performing T5 model configurations (fine-tuned)

### Data Processing

**Description:**

The data processing pipeline consists of the following steps:

1. **Prefix addition**: Each natural language query is prepended with the task-specific prefix "translate English to SQL: " before tokenization. This prefix explicitly informs the encoder about the translation task, helping the model understand that it should convert English text to SQL queries. The prefix adds approximately 6 tokens to each input sequence.

2. **Truncation**: Both encoder inputs (prefixed natural language queries) and decoder targets (SQL queries) are truncated to a maximum length of 512 tokens using the T5 tokenizer's truncation functionality. This ensures all sequences fit within the model's maximum sequence length constraint while preserving as much information as possible.

3. **EOS token handling**: End-of-sequence (EOS) tokens (`</s>`, token ID 1) are explicitly appended to decoder targets during preprocessing if not already present. This ensures consistent sequence termination for the language modeling objective.

4. **Dynamic padding**: During batch collation, sequences are dynamically padded to the maximum length within each batch. Encoder and decoder input sequences are padded using `pad_token_id` (0), while decoder target sequences are padded with `-100` to indicate positions that should be ignored during loss computation.

5. **Decoder input preparation**: For training, decoder inputs are created by shifting decoder targets by one position and prepending `pad_token_id` (0) as the initial decoder input token. This implements the standard teacher-forcing training approach for autoregressive models.

6. **No additional preprocessing**: Raw text from data files is used directly without lowercasing, special formatting, or other text transformations, preserving the original case and structure of both natural language queries and SQL statements.

### Tokenization

**Description:**

Tokenization is performed using the **T5TokenizerFast** from HuggingFace Transformers, loaded from the `google-t5/t5-small` checkpoint. This tokenizer uses SentencePiece-based subword tokenization, which is well-suited for both natural language and code-like structures such as SQL.

**Encoder tokenization**: Natural language queries are tokenized after prefix addition. The tokenizer is called with `padding=False`, `truncation=True`, and `max_length=512`. The tokenizer automatically handles subword segmentation and special token insertion.

**Decoder tokenization**: SQL queries are tokenized directly without any prefix. The same tokenizer settings are applied (`padding=False`, `truncation=True`, `max_length=512`). The SentencePiece tokenizer naturally handles SQL syntax, keywords, and identifiers through its learned subword vocabulary.

**Special tokens**: The tokenizer manages the following special tokens automatically:
- `<pad>` (token ID: 0): Used for padding sequences to equal lengths within batches
- `</s>` (token ID: 1): End-of-sequence token, used to mark the end of generated sequences
- `<unk>` (token ID: 2): Unknown token for out-of-vocabulary subwords (rarely encountered)

**Rationale for using default T5 tokenizer**: The standard T5 tokenizer was chosen because (1) it was pretrained on a diverse corpus that likely included code-like structures, (2) SentencePiece tokenization handles both natural language and SQL syntax effectively without requiring custom vocabulary, and (3) using the pretrained tokenizer maintains consistency with the pretrained model weights. No custom tokenization or vocabulary modifications were necessary.

### Architecture

**Description:**

**Model**: T5ForConditionalGeneration (small variant) from the `google-t5/t5-small` checkpoint. The small variant consists of 6 encoder layers, 6 decoder layers, 8 attention heads, and a hidden dimension of 512.

**Fine-tuning approach**: **Full model fine-tuning** - all parameters in both the encoder and decoder are trainable. No layers or components are frozen during training.

**Model components fine-tuned**:
- **Encoder**: All 6 transformer layers including self-attention mechanisms, feed-forward networks, layer normalization, and positional embeddings. The encoder processes the prefixed natural language queries.
- **Decoder**: All 6 transformer layers including self-attention, cross-attention (to encoder outputs), feed-forward networks, layer normalization, and positional embeddings. The decoder generates SQL queries autoregressively.
- **Language modeling head**: The output projection layer that maps decoder hidden states to the vocabulary space (32,128 tokens for T5-small).
- **Shared token embeddings**: The embedding layer shared between encoder and decoder inputs, which is updated during fine-tuning.

**Rationale for full fine-tuning**: Full fine-tuning was chosen over partial freezing (e.g., freezing encoder layers) because (1) the text-to-SQL task requires understanding both natural language semantics and SQL syntax, which benefits from adapting all model representations, (2) the domain shift from general text to SQL queries warrants updating all layers, and (3) the small model size (60M parameters) allows full fine-tuning without excessive computational cost. This approach achieved 83.6% Record F1 on the development set, indicating that full fine-tuning is effective for this task.

### Hyperparameters

**Key hyperparameters used in the best-performing model:**

- **Learning rate**: **1e-3 (0.001)**
  - This value was chosen after experimentation. Initial attempts with 1e-1 (0.1) caused training instability and divergence. The value 1e-3 provides stable convergence while allowing sufficient model updates.

- **Batch size**: **16**
  - Selected based on GPU memory constraints and training efficiency. Smaller batch sizes (e.g., 8) were tested but did not show significant differences in final performance.

- **Optimizer**: **AdamW**
  - Beta parameters: (0.9, 0.999) - standard values for AdamW
  - Epsilon: 1e-8 - numerical stability parameter
  - Weight decay: **0** - no weight decay regularization was applied in the best model

- **Learning rate scheduler**: **Cosine annealing with warmup**
  - Scheduler type: cosine decay
  - Warmup epochs: **0** - no warmup period was used
  - The learning rate decays following a cosine schedule from the initial learning rate (1e-3) to near zero over the training period

- **Maximum training epochs**: **20**
  - Training was allowed to run for up to 20 epochs, though early stopping typically terminated training earlier

- **Early stopping**: **Patience-based**
  - Patience: **3-5 epochs** (exact value depends on configuration)
  - Metric: Record F1 on development set
  - Training stops if the development set Record F1 does not improve for the specified number of consecutive epochs
  - The best model checkpoint (highest Record F1) is saved and restored for final evaluation

- **Loss function**: **CrossEntropyLoss with ignore_index=-100**
  - Standard cross-entropy loss for sequence-to-sequence language modeling
  - Padded positions in decoder targets are set to -100 and automatically ignored in loss computation
  - Loss is computed only over non-padded tokens

- **Generation parameters** (for evaluation and inference):
  - Decoding strategy: **Greedy decoding** (num_beams=1)
  - Maximum generation length: **512 tokens**
  - Early stopping: True (generation stops when EOS token is produced)
  - No temperature scaling or top-k/top-p sampling was used

**Stopping criteria:**
- **Primary**: Early stopping based on Record F1 on development set. Training stops if no improvement is observed for the patience period (typically 3-5 epochs).
- **Secondary**: Maximum number of epochs (20) reached if early stopping does not trigger.
- **Best model selection**: The model checkpoint with the highest Record F1 on the development set is saved as `best_model.pt` and used for final evaluation and test set inference.

**Training results**: The best model achieved **83.6% Record F1** and **82.2% Record EM** on the development set at epoch 16, with training stopping at epoch 19 due to early stopping criteria.


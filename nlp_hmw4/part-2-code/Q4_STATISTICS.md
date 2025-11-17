# Q4: Data Statistics and Processing

## Table 1: Data Statistics Before Pre-processing

| Statistics Name | Train | Dev |
|----------------|-------|-----|
| Number of examples | 4225 | 466 |
| Mean sentence length | 17.10 | 17.07 |
| Mean SQL query length | 216.37 | 210.05 |
| Vocabulary size (natural language) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |

**Additional Statistics:**
- Max sentence length (NL): Train=59, Dev=43
- Max SQL query length: Train=510, Dev=502
- Min sentence length (NL): Train=2, Dev=3
- Min SQL query length: Train=25, Dev=30

## Table 2: Data Statistics After Pre-processing

**Model name:** T5 fine-tuned (google-t5/t5-small)

| Statistics Name | Train | Dev |
|----------------|-------|-----|
| Mean sentence length | 23.10 | 23.07 |
| Mean SQL query length | 217.37 | 211.05 |
| Vocabulary size (natural language) | 796 | 470 |
| Vocabulary size (SQL) | 556 | 396 |

**Additional Statistics:**
- Max sentence length (NL): Train=65, Dev=49
- Max SQL query length: Train=511, Dev=503
- Min sentence length (NL): Train=8, Dev=9
- Min SQL query length: Train=26, Dev=31

## Notes:

1. **Number of examples** remains the same (4225 train, 466 dev) before and after preprocessing.

2. **Mean sentence length (NL)** increases from ~17 to ~23 tokens because:
   - We add the prefix "translate English to SQL: " to each natural language query
   - This adds approximately 6 tokens to each sentence

3. **Mean SQL query length** increases slightly (~1 token) because:
   - We add an EOS token to each SQL query during preprocessing
   - Truncation may affect some longer queries, but most are within the 512 token limit

4. **Vocabulary sizes** increase slightly after preprocessing:
   - The prefix introduces a few additional tokens
   - EOS tokens are explicitly added to SQL queries

5. **Preprocessing steps applied:**
   - Added prefix "translate English to SQL: " to NL queries
   - Tokenized with T5TokenizerFast
   - Applied truncation (max_length=512) to both NL and SQL
   - Added EOS token to SQL queries if not present


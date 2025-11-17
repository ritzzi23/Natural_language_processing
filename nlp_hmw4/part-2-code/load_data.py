import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Load and tokenize data for the specified split.
        '''
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        self.nl_queries = load_lines(nl_path)
        
        # For train and dev, also load SQL queries
        if split in ['train', 'dev']:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            self.sql_queries = load_lines(sql_path)
        else:
            # Test set has no ground truth SQL
            self.sql_queries = None
        
        # Tokenize encoder inputs (NL queries)
        # Add a prefix for better task specification
        prefix = "translate English to SQL: "
        self.encoder_inputs = []
        for nl_query in self.nl_queries:
            # Tokenize with prefix
            encoded = tokenizer(
                prefix + nl_query,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            self.encoder_inputs.append(encoded['input_ids'])
        
        # Tokenize decoder targets (SQL queries) for train/dev
        if split in ['train', 'dev']:
            self.decoder_targets = []
            for sql_query in self.sql_queries:
                encoded = tokenizer(
                    sql_query,
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_tensors=None
                )
                # Add EOS token if not present
                if encoded['input_ids'][-1] != tokenizer.eos_token_id:
                    encoded['input_ids'].append(tokenizer.eos_token_id)
                self.decoder_targets.append(encoded['input_ids'])
        else:
            self.decoder_targets = None
    
    def __len__(self):
        return len(self.nl_queries)

    def __getitem__(self, idx):
        '''
        Returns:
            For train/dev: (encoder_input_ids, decoder_input_ids, decoder_target_ids)
            For test: (encoder_input_ids,)
        '''
        encoder_input_ids = torch.tensor(self.encoder_inputs[idx], dtype=torch.long)
        
        if self.split == 'test':
            # Test set: only return encoder input
            return (encoder_input_ids,)
        
        # Train/dev: return encoder input, decoder input, and decoder targets
        decoder_target_ids = torch.tensor(self.decoder_targets[idx], dtype=torch.long)
        
        # Decoder input is decoder target shifted by one position
        # Start with pad_token_id (0) which serves as decoder start token
        decoder_input_ids = torch.cat([
            torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
            decoder_target_ids[:-1]  # Remove last token (EOS) from input
        ])
        
        return (encoder_input_ids, decoder_input_ids, decoder_target_ids)

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Unpack batch
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create encoder attention mask (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder inputs
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Pad decoder targets with -100 (ignored by CrossEntropyLoss)
    # This is critical: padded positions must be -100, not PAD_IDX (0)
    # Otherwise the model will learn to predict padding tokens
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=-100)
    
    # Initial decoder inputs: first token for each sequence in the batch
    # This is the pad_token_id (0) which serves as decoder start token
    initial_decoder_inputs = torch.full(
        (len(batch), 1), 
        PAD_IDX, 
        dtype=torch.long
    )
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Unpack batch (test set only has encoder inputs)
    encoder_inputs = [item[0] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create encoder attention mask (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs: first token for each sequence in the batch
    # This is the pad_token_id (0) which serves as decoder start token
    initial_decoder_inputs = torch.full(
        (len(batch), 1), 
        PAD_IDX, 
        dtype=torch.long
    )
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    '''
    Load natural language queries and SQL queries from data files.
    
    Returns:
        train_x: List of natural language queries (training set)
        train_y: List of SQL queries (training set)
        dev_x: List of natural language queries (dev set)
        dev_y: List of SQL queries (dev set)
        test_x: List of natural language queries (test set, no ground truth SQL)
    '''
    # Load training data
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    
    # Load dev data
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Load test data (only NL queries, no SQL ground truth)
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x
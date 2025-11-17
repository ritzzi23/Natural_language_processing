import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Combined Transformation: Synonym Replacement + Typos
    # Higher probabilities to create more challenging OOD scenarios (>4% accuracy drop)
    # This is reasonable as people naturally use synonyms and make typos
    
    # QWERTY keyboard layout for typo simulation
    qwerty_neighbors = {
        'a': ['s', 'w', 'q'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v', 'd'],
        'd': ['s', 'f', 'e'], 'e': ['w', 'r', 'd'], 'f': ['d', 'g', 'r'],
        'g': ['f', 'h', 't'], 'h': ['g', 'j', 'y'], 'i': ['u', 'o', 'k'],
        'j': ['h', 'k', 'u'], 'k': ['j', 'l', 'i'], 'l': ['k', 'o'],
        'm': ['n', 'j'], 'n': ['b', 'm', 'h'], 'o': ['i', 'p', 'l'],
        'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 't', 'f'],
        's': ['a', 'd', 'w'], 't': ['r', 'y', 'g'], 'u': ['y', 'i', 'j'],
        'v': ['c', 'b', 'f'], 'w': ['q', 'e', 's'], 'x': ['z', 'c', 's'],
        'y': ['t', 'u', 'h'], 'z': ['x', 'a']
    }
    
    def introduce_typo(word):
        """Introduce a typo by replacing a random letter with a QWERTY neighbor"""
        if len(word) < 3:
            return word
        
        word_lower = word.lower()
        # Focus on vowels and common letters for more realistic typos
        vowels = ['a', 'e', 'i', 'o', 'u']
        candidates = [i for i, char in enumerate(word_lower) if char in vowels or char in qwerty_neighbors]
        
        if not candidates:
            candidates = [i for i, char in enumerate(word_lower) if char.isalpha()]
        
        if not candidates:
            return word
        
        pos = random.choice(candidates)
        char = word_lower[pos]
        
        if char in qwerty_neighbors and qwerty_neighbors[char]:
            # Replace with a neighbor key
            replacement_char = random.choice(qwerty_neighbors[char])
            # Preserve case
            if word[pos].isupper():
                replacement_char = replacement_char.upper()
            result = word[:pos] + replacement_char + word[pos+1:]
            return result
        
        return word
    
    text = example["text"]
    words = word_tokenize(text)
    transformed_words = []
    
    for word in words:
        # Keep punctuation and non-alphabetic tokens as-is
        if not word.isalpha() or len(word) < 3:
            transformed_words.append(word)
            continue
        
        # 50% probability for synonym replacement, 30% probability for typos
        # This creates more challenging transformations to exceed 4% accuracy drop
        rand_val = random.random()
        
        if rand_val < 0.5:
            # Try synonym replacement (increased from 30% to 50%)
            synsets = wordnet.synsets(word.lower())
            
            if synsets:
                # Get all lemmas (synonyms) from multiple synsets for more variety
                synonyms = []
                for syn in synsets[:3]:  # Check first 3 synsets for more variety
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        # Exclude the original word and ensure it's a single word
                        if synonym.lower() != word.lower() and ' ' not in synonym:
                            synonyms.append(synonym)
                
                # If synonyms found, randomly select one
                if synonyms:
                    # Prefer synonyms that match the original case
                    if word[0].isupper():
                        replacement = random.choice(synonyms).capitalize()
                    else:
                        replacement = random.choice(synonyms).lower()
                    transformed_words.append(replacement)
                    continue
        
        elif rand_val < 0.8:
            # Introduce typo (30% probability, when synonym replacement fails)
            transformed_word = introduce_typo(word)
            transformed_words.append(transformed_word)
            continue
        
        # Keep original word if no transformation applied
        transformed_words.append(word)
    
    # Reconstruct text using TreebankWordDetokenizer
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example

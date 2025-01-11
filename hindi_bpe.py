import re
from collections import Counter
from typing import Dict, List, Tuple, Set
import unicodedata
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import json
from matplotlib import pyplot as plt
from pathlib import Path
from byte_pair_encoder import BytePairEncoder, TokenizerInternal

class HindiBPE:
    def __init__(self, vocab_size: int = 5000):
        print(f"\nInitializing HindiBPE with max vocab size: {vocab_size}")
        self.vocab_size = vocab_size
        self.encoder = None
        
    def train(self, text: str) -> None:
        """Train BPE on Hindi text."""
        print("\nInitializing BytePairEncoder...")
        self.encoder = BytePairEncoder(text)
        
        print("\nTraining BPE...")
        self.encoder.encode_to_vocab_size(
            target_vocab_size=self.vocab_size,
            plot_interval=1000,
            print_interval=100
        )
        
        # Plot final statistics
        self.encoder.plot_statistics()
        
        # Save the trained model
        self.save_tokenizer()
    
    def encode(self, text: str) -> List[str]:
        """Encode Hindi text using trained tokenizer."""
        if self.encoder is None:
            raise ValueError("Tokenizer not trained yet!")
            
        print("\nTokenizing text...")
        tokenizer = TokenizerInternal(self.encoder)
        tokens = list(tokenizer.tokenize(text))
        
        compression = self.calculate_compression_ratio(text, tokens)
        print(f"\nEncoding completed:")
        print(f"Token count: {len(tokens)}")
        print(f"Unique tokens: {len(set(tokens))}")
        print(f"Compression ratio: {compression:.2f}")
        
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        """Decode tokens back to text."""
        if self.encoder is None:
            raise ValueError("Tokenizer not trained yet!")
            
        print("\nDecoding tokens...")
        decoded = "".join(tokens)
        print(f"Decoded length: {len(decoded)} characters")
        return decoded
    
    def save_tokenizer(self, path: str = "tokenizer") -> None:
        """Save the tokenizer to disk."""
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        
        # Save the encoder
        self.encoder.save_to_file(save_dir / "encoder.json")
        
        # Save vocabulary stats
        stats = self.get_token_statistics()
        with open(save_dir / "vocab_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        print(f"Tokenizer saved to {save_dir}")
    
    @classmethod
    def load_tokenizer(cls, path: str = "tokenizer") -> "HindiBPE":
        """Load a trained tokenizer from disk."""
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {load_dir}")
            
        # Create instance
        instance = cls()
        
        # Load encoder
        instance.encoder = BytePairEncoder.load_from_file(load_dir / "encoder.json")
        
        print(f"Loaded tokenizer from {load_dir}")
        print(f"Vocabulary size: {len(instance.encoder.itos)}")
        return instance
    
    def get_token_statistics(self) -> Dict:
        """Get statistics about the learned tokens."""
        if self.encoder is None:
            raise ValueError("Tokenizer not trained yet!")
            
        token_lengths = [len(token) for token in self.encoder.itos.values()]
        return {
            'vocab_size': len(self.encoder.itos),
            'avg_token_length': sum(token_lengths) / len(token_lengths),
            'min_token_length': min(token_lengths),
            'max_token_length': max(token_lengths),
            'length_distribution': Counter(token_lengths),
            'training_stats': self.encoder.stats
        }
    
    def calculate_compression_ratio(self, text: str, tokens: List[str]) -> float:
        """Calculate compression ratio."""
        original_size = len(text)
        encoded_size = sum(len(token) for token in tokens)
        return original_size / encoded_size

def preprocess_hindi_text(text: str) -> str:
    """Preprocess Hindi text for better BPE training."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove unnecessary punctuation (keep essential ones)
    text = re.sub(r'[^\u0900-\u097F\s।]', '', text)
    
    return text 
import unittest
from pathlib import Path
from hindi_bpe import HindiBPE, preprocess_hindi_text
from data_loader import load_hindi_dataset
import pandas as pd
import plotly.express as px
from typing import List, Dict
import time

class HindiBPETest:
    def __init__(self, vocab_size: int = 4500, num_articles: int = 1000):
        self.vocab_size = vocab_size
        self.num_articles = num_articles
        self.bpe = None
        self.text = None
        self.encoded = None
        self.stats = {}
        
    def load_data(self) -> str:
        """Load and preprocess the dataset."""
        print("\nStep 1: Loading dataset...")
        start_time = time.time()
        
        # Load train split
        self.text = load_hindi_dataset(
            split="train",
            num_files=self.num_articles
        )
        self.text = preprocess_hindi_text(self.text)
        
        # Get validation text for testing
        self.valid_text = load_hindi_dataset(
            split="valid",
            num_files=min(self.num_articles // 5, 100)  # 20% of train size or max 100
        )
        
        self.stats['load_time'] = time.time() - start_time
        self.stats['original_length'] = len(self.text)
        self.stats['valid_length'] = len(self.valid_text)
        print(f"Loading completed in {self.stats['load_time']:.2f} seconds")
        return self.text
    
    def train_tokenizer(self) -> HindiBPE:
        """Train the BPE tokenizer."""
        print("\nStep 2: Training BPE tokenizer...")
        start_time = time.time()
        
        self.bpe = HindiBPE(vocab_size=self.vocab_size)
        self.bpe.train(self.text)
        
        self.stats['train_time'] = time.time() - start_time
        self.stats['vocab_size'] = len(self.bpe.vocab)
        print(f"Training completed in {self.stats['train_time']:.2f} seconds")
        return self.bpe
    
    def encode_text(self) -> List[str]:
        """Encode the text using trained tokenizer."""
        print("\nStep 3: Encoding text...")
        start_time = time.time()
        
        self.encoded = self.bpe.encode(self.text)
        
        self.stats['encode_time'] = time.time() - start_time
        self.stats['encoded_length'] = sum(len(token) for token in self.encoded)
        self.stats['compression_ratio'] = self.stats['original_length'] / self.stats['encoded_length']
        print(f"Encoding completed in {self.stats['encode_time']:.2f} seconds")
        return self.encoded
    
    def save_visualizations(self, output_dir: str = "output"):
        """Generate and save visualizations."""
        print("\nStep 4: Generating visualizations...")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Token length distribution
        token_lengths = [len(token) for token in self.bpe.vocab]
        df = pd.DataFrame({'Length': token_lengths})
        fig = px.histogram(df, x='Length', 
                          title='Token Length Distribution',
                          labels={'Length': 'Token Length', 'count': 'Frequency'})
        fig.write_html(output_dir / "token_distribution.html")
        
        # Compression visualization
        comp_df = pd.DataFrame({
            'Stage': ['Original', 'Encoded'],
            'Size': [self.stats['original_length'], self.stats['encoded_length']]
        })
        fig = px.bar(comp_df, x='Stage', y='Size',
                    title='Text Compression Comparison')
        fig.write_html(output_dir / "compression.html")
        
        # Save statistics to CSV
        pd.DataFrame([self.stats]).to_csv(output_dir / "stats.csv")
        print(f"Visualizations saved to {output_dir}")
    
    def print_summary(self):
        """Print summary of the tokenization process."""
        print("\nTokenization Summary:")
        print("-" * 50)
        print(f"Dataset size: {self.stats['original_length']:,} characters")
        print(f"Vocabulary size: {self.stats['vocab_size']:,} tokens")
        print(f"Compression ratio: {self.stats['compression_ratio']:.2f}")
        print(f"\nProcessing times:")
        print(f"Loading: {self.stats['load_time']:.2f} seconds")
        print(f"Training: {self.stats['train_time']:.2f} seconds")
        print(f"Encoding: {self.stats['encode_time']:.2f} seconds")
        
    def run_full_pipeline(self) -> Dict:
        """Run the complete tokenization pipeline."""
        self.load_data()
        self.train_tokenizer()
        self.encode_text()
        self.save_visualizations()
        self.print_summary()
        return self.stats

def main():
    # Example usage
    test = HindiBPETest(vocab_size=4500, num_articles=1000)
    stats = test.run_full_pipeline()
    
    # Test tokenization on a sample text
    sample_text = """
    भारत एक विशाल देश है। यहाँ की संस्कृति बहुत पुरानी है।
    हिंदी भारत की प्रमुख भाषाओं में से एक है।
    """
    
    print("\nTesting tokenization on sample text:")
    tokens = test.bpe.encode(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Tokens: {tokens}")
    decoded = test.bpe.decode(tokens)
    print(f"Decoded text: {decoded}")
    
    # Verify compression ratio requirement
    if stats['compression_ratio'] >= 3.2:
        print("\nSuccess: Achieved required compression ratio ≥ 3.2")
    else:
        print("\nWarning: Compression ratio below target 3.2")
    
    # Verify vocabulary size requirement
    if stats['vocab_size'] < 5000:
        print("Success: Vocabulary size within limit < 5000")
    else:
        print("Warning: Vocabulary size exceeds limit")

if __name__ == "__main__":
    main() 
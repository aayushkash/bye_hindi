from typing import List, Dict, Optional
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import json
from pathlib import Path

class TrieNode:
    """Node in the prefix tree (trie) for fast token matching"""
    def __init__(self):
        self.children = {}
        self.is_token = False
        self.token = None

class BytePairEncoder:
    def __init__(self, text: str):
        # Initialize vocabulary from characters
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        # Initial encoding of text
        self.data = [self.stoi[c] for c in text]
        
        # Statistics tracking
        self.stats = {
            "vocab_sizes": [len(self.chars)],
            "data_sizes": [len(self.data)],
            "compression_ratios": [1.0],
            "merge_counts": [],
            "tokens_created": [],
            "max_token_lengths": [1],
        }
        
        # Store original length for compression ratio
        self.original_length = len(self.data)
        self.max_token_length = 1

    def get_digram_stats(self) -> Counter:
        """Get digram counts"""
        counts = Counter()
        for pair in zip(self.data, self.data[1:]):
            pair = (int(pair[0]), int(pair[1]))
            counts[pair] += 1
        return counts

    def encode_to_vocab_size(self, target_vocab_size: int, plot_interval: Optional[int] = None, 
                           print_interval: int = 100) -> None:
        """Train until reaching target vocabulary size"""
        pbar = tqdm(total=target_vocab_size, desc="Training BPE", initial=len(self.chars))
        
        iteration = 0
        while len(self.itos) < target_vocab_size:
            result = self._merge_step()
            if result is None:
                break
                
            iteration += 1
            pbar.update(1)
            
            if print_interval and iteration % print_interval == 0:
                self._print_progress(iteration)
                
            if plot_interval and iteration % plot_interval == 0:
                self.plot_statistics(iteration=iteration)
        
        pbar.close()

    def _merge_step(self):
        """Perform one merge operation"""
        stats = self.get_digram_stats()
        if not stats:
            return None
            
        top_pair, count = max(stats.items(), key=lambda x: x[1])
        new_token = self._add_token(top_pair)
        self.data = self._replace_pairs(top_pair, new_token)
        self._update_stats(count)
        
        return new_token, count

    def _add_token(self, pair: tuple) -> int:
        """Add new token to vocabulary"""
        token_str = self.itos[pair[0]] + self.itos[pair[1]]
        token_id = len(self.itos)
        self.stoi[token_str] = token_id
        self.itos[token_id] = token_str
        self.max_token_length = max(self.max_token_length, len(token_str))
        return token_id

    def _replace_pairs(self, pair: tuple, new_token: int) -> List[int]:
        """Replace all occurrences of pair with new token"""
        result = []
        i = 0
        while i < len(self.data):
            if i < len(self.data) - 1 and self.data[i] == pair[0] and self.data[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(self.data[i])
                i += 1
        return result

    def _update_stats(self, merge_count: int):
        """Update training statistics"""
        self.stats["vocab_sizes"].append(len(self.itos))
        self.stats["data_sizes"].append(len(self.data))
        compression = self.original_length / len(self.data)
        self.stats["compression_ratios"].append(compression)
        self.stats["merge_counts"].append(merge_count)
        self.stats["tokens_created"].append(self.itos[len(self.itos)-1])
        self.stats["max_token_lengths"].append(self.max_token_length)

    def plot_statistics(self, iteration: Optional[int] = None):
        """Plot training statistics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training metrics
        ax1.plot(self.stats["vocab_sizes"], self.stats["data_sizes"])
        ax1.set_title("Vocabulary vs Dataset Size")
        
        ax2.plot(self.stats["vocab_sizes"], self.stats["compression_ratios"])
        ax2.set_title("Compression Ratio Progress")
        
        if self.stats["merge_counts"]:
            ax3.hist(self.stats["merge_counts"], bins=30)
            ax3.set_title("Merge Counts Distribution")
        
        if self.stats["tokens_created"]:
            lengths = [len(t) for t in self.stats["tokens_created"]]
            ax4.plot(range(len(lengths)), lengths)
            ax4.set_title("Token Length Evolution")
        
        plt.tight_layout()
        plt.show()

    def save_to_file(self, filepath: Path):
        """Save encoder state"""
        state = {
            "chars": self.chars,
            "stoi": self.stoi,
            "max_token_length": self.max_token_length,
            "stats": self.stats
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path):
        """Load encoder state"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        instance = cls("")  # Create empty instance
        instance.chars = state["chars"]
        instance.stoi = state["stoi"]
        instance.itos = {int(i): s for s, i in state["stoi"].items()}
        instance.max_token_length = state["max_token_length"]
        instance.stats = state["stats"]
        
        return instance

    def _print_progress(self, iteration: int):
        """Print training progress"""
        print(f"\nIteration {iteration}:")
        print(f"Vocabulary size: {len(self.itos):,}")
        print(f"Data size: {len(self.data):,}")
        print(f"Compression ratio: {self.stats['compression_ratios'][-1]:.2f}")
        
        if self.stats["merge_counts"]:
            last_merge = self.stats["merge_counts"][-1]
            last_token = self.stats["tokens_created"][-1]
            print(f"Last merge count: {last_merge:,}")
            print(f"Last token created: '{last_token}'")
        
        print(f"Max token length: {self.max_token_length}")

class TokenizerInternal:
    """Tokenizer using trained BPE model"""
    def __init__(self, encoder: BytePairEncoder):
        self.stoi = encoder.stoi
        self.max_token_length = encoder.max_token_length
        self._trie = self._build_trie()

    def _build_trie(self) -> TrieNode:
        """Build trie for efficient tokenization"""
        root = TrieNode()
        for token in self.stoi:
            node = root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_token = True
            node.token = token
        return root

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using trie-based matching"""
        tokens = []
        pos = 0
        while pos < len(text):
            token = self._find_longest_token(text[pos:])
            tokens.append(token)
            pos += len(token)
        return tokens

    def _find_longest_token(self, text: str) -> str:
        """Find longest matching token starting at current position"""
        node = self._trie
        longest = text[0]
        current = ""
        
        for char in text[:self.max_token_length]:
            if char not in node.children:
                break
            current += char
            node = node.children[char]
            if node.is_token:
                longest = node.token
                
        return longest 
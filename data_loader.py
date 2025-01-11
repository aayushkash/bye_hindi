import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import random

def load_hindi_dataset(base_path: str = "data", split: str = "train", num_files: int = None) -> str:
    """
    Load Hindi text from dataset with train/validation split structure.
    
    Args:
        base_path: Base directory containing train and validation folders
        split: Either 'train' or 'valid'
        num_files: Number of files to load (None for all files)
    """
    base_dir = Path(base_path)
    split_dir = base_dir / split / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    print(f"\nLoading Hindi dataset from {split_dir}")
    
    # Get all txt files in the directory
    txt_files = list(split_dir.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {split_dir}")
    
    # Sort files by word count (assuming filenames contain word counts)
    txt_files.sort(key=lambda x: int(x.stem))
    
    # Sample files if num_files is specified
    if num_files is not None:
        if num_files < len(txt_files):
            txt_files = random.sample(txt_files, num_files)
    
    print(f"Found {len(txt_files)} files")
    
    # Load and combine text from files
    texts = []
    total_chars = 0
    total_words = 0
    
    for idx, file_path in enumerate(txt_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                word_count = int(file_path.stem)  # Filename is word count
                texts.append(text)
                total_chars += len(text)
                total_words += word_count
                
                if idx % 10 == 0:
                    print(f"Processed {idx}/{len(txt_files)} files. "
                          f"Total characters: {total_chars:,}, "
                          f"Total words: {total_words:,}")
                
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    combined_text = "\n\n".join(texts)
    
    print(f"\nDataset loading completed:")
    print(f"Total files: {len(texts)}")
    print(f"Total characters: {len(combined_text):,}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per file: {total_words/len(texts):,.1f}")
    
    return combined_text

def get_dataset_stats(base_path: str = "data") -> Dict:
    """Get statistics about the dataset."""
    stats = {}
    for split in ['train', 'valid']:
        split_dir = Path(base_path) / split
        if split_dir.exists():
            txt_files = list(split_dir.glob("*.txt"))
            word_counts = [int(f.stem) for f in txt_files]
            stats[split] = {
                'num_files': len(txt_files),
                'total_words': sum(word_counts),
                'min_words': min(word_counts) if word_counts else 0,
                'max_words': max(word_counts) if word_counts else 0,
                'avg_words': sum(word_counts)/len(word_counts) if word_counts else 0
            }
    return stats

def load_train_valid_split(base_path: str = "data", 
                          train_files: int = None,
                          valid_files: int = None) -> Tuple[str, str]:
    """Load both train and validation splits."""
    train_text = load_hindi_dataset(base_path, "train", train_files)
    valid_text = load_hindi_dataset(base_path, "valid", valid_files)
    return train_text, valid_text

if __name__ == "__main__":
    # Print dataset statistics
    stats = get_dataset_stats()
    print("\nDataset Statistics:")
    print("-" * 50)
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        for key, value in split_stats.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")
    
    # Load sample data
    print("\nLoading sample data...")
    train_text, valid_text = load_train_valid_split(train_files=5, valid_files=2)
    print(f"\nSample train text (first 200 chars):\n{train_text[:200]}")
    print(f"\nSample valid text (first 200 chars):\n{valid_text[:200]}") 
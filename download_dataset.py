import kagglehub
from pathlib import Path
import shutil
import pandas as pd
import re
import nltk
from typing import List, Dict
from tqdm import tqdm

def count_hindi_words(text: str) -> int:
    """Count words in Hindi text."""
    words = text.strip().split()
    hindi_words = [w for w in words if re.search(r'[\u0900-\u097F]', w)]
    return len(hindi_words)

def create_dataframe_from_files(downloaded_paths: List[str]) -> pd.DataFrame:
    """Create a DataFrame from downloaded text files."""
    print("\nCreating DataFrame from text files...")
    
    data = []
    for file_path in tqdm(downloaded_paths):
        if file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Split into title and text (assuming first line is title)
                lines = content.split('\n', 1)
                title = lines[0].strip()
                text = lines[1].strip() if len(lines) > 1 else ""
                
                data.append({
                    'title': title,
                    'text': text,
                    'word_count': count_hindi_words(content)
                })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} articles")
    return df

def process_and_split_articles(df: pd.DataFrame, 
                             output_dir: Path,
                             train_ratio: float = 0.8,
                             min_words: int = 100,
                             max_words: int = 5000) -> Dict[str, int]:
    """Process articles and split them into files based on word count."""
    
    # Create output directories
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    
    stats = {'train': 0, 'valid': 0, 'skipped': 0}
    
    print("\nProcessing articles...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Skip if too short or too long
            if row['word_count'] < min_words or row['word_count'] > max_words:
                stats['skipped'] += 1
                continue
            
            # Combine title and text
            full_text = f"{row['title']}\n\n{row['text']}"
            
            # Decide split (train or valid)
            is_train = pd.np.random.random() < train_ratio
            output_dir = train_dir if is_train else valid_dir
            
            # Save to file named by word count
            file_path = output_dir / f"{row['word_count']}.txt"
            suffix = 1
            while file_path.exists():
                file_path = output_dir / f"{row['word_count']}_{suffix}.txt"
                suffix += 1
                
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            if is_train:
                stats['train'] += 1
            else:
                stats['valid'] += 1
                
        except Exception as e:
            print(f"Error processing article: {e}")
            stats['skipped'] += 1
            continue
    
    return stats

def download_hindi_wikipedia_dataset():
    """Download and process Hindi Wikipedia dataset."""
    print("Starting dataset download...")
    
    try:
        # Download the dataset using kagglehub
        downloaded_paths = kagglehub.dataset_download(
            "disisbig/hindi-wikipedia-articles-172k"
        )
        
        print("Dataset downloaded successfully!")
        print("Downloaded files:", downloaded_paths)
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Create DataFrame from downloaded files
        df = create_dataframe_from_files(downloaded_paths)
        
        # Save DataFrame for future use
        df.to_parquet(data_dir / "articles.parquet")
        print(f"Saved DataFrame to {data_dir / 'articles.parquet'}")
        
        # Process and split the articles
        stats = process_and_split_articles(df, data_dir)
        
        # Print statistics
        print("\nProcessing completed:")
        print(f"Train files: {stats['train']}")
        print(f"Validation files: {stats['valid']}")
        print(f"Skipped articles: {stats['skipped']}")
        
        # Get file sizes
        train_size = sum(f.stat().st_size for f in (data_dir / "train").glob("*.txt"))
        valid_size = sum(f.stat().st_size for f in (data_dir / "valid").glob("*.txt"))
        
        print(f"\nTotal size:")
        print(f"Train: {train_size / (1024*1024):.2f} MB")
        print(f"Validation: {valid_size / (1024*1024):.2f} MB")
        
        return True
            
    except Exception as e:
        print(f"Error downloading/processing dataset: {e}")
        return False

def verify_dataset_structure():
    """Verify the dataset directory structure and files."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("Error: Data directory not found!")
        return False
    
    # Check if we have the processed DataFrame
    parquet_file = data_dir / "articles.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        print(f"\nArticles DataFrame:")
        print(f"Total articles: {len(df)}")
        # print(f"Word count range: {df['word_count'].min()} - {df['word_count'].max()}")
    
    for split in ['train', 'valid']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Error: {split} directory not found!")
            return False
        
        txt_files = list(split_dir.glob("*.txt"))
        if not txt_files:
            print(f"Error: No text files found in {split} directory!")
            return False
        
        print(f"\n{split.upper()} split:")
        print(f"Number of files: {len(txt_files)}")
        word_counts = [int(f.stem.split('_')[0]) for f in txt_files]
        print(f"Word count range: {min(word_counts)} - {max(word_counts)}")
    
    return True

if __name__ == "__main__":
    # Download and process the dataset
    success = download_hindi_wikipedia_dataset()
    
    if success:
        print("\nVerifying dataset structure...")
        verify_dataset_structure() 
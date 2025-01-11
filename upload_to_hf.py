from huggingface_hub import HfApi, create_repo, Repository
import json
from pathlib import Path
import shutil
import os
import subprocess

def setup_git_lfs():
    """Setup git-lfs if not already configured"""
    try:
        # Check if git-lfs is installed
        subprocess.run(['git', 'lfs', 'version'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("git-lfs is not installed. Please install it from https://git-lfs.github.com/")
    
    # Initialize git-lfs
    subprocess.run(['git', 'lfs', 'install'], check=True)

def upload_tokenizer_to_hf(
    repo_name: str,
    token: str,
    local_dir: str = ".",
    organization: str = None,
    username: str = None
) -> str:
    """Upload tokenizer to Hugging Face Hub"""
    # Setup git-lfs first
    setup_git_lfs()
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Get username if not provided
    if not username:
        try:
            username = api.whoami(token=token)["name"]
        except Exception as e:
            raise ValueError(f"Could not get username. Please provide it explicitly: {e}")
    
    # Setup repository name
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        repo_id = f"{username}/{repo_name}"
        
    print(f"\nPreparing to upload to {repo_id}...")
    
    try:
        # Create repository
        repo_url = create_repo(
            repo_id,
            token=token,
            private=False,
            exist_ok=True
        )
        
        # Prepare local directory
        repo_local_path = Path("hf_repo")
        if repo_local_path.exists():
            shutil.rmtree(repo_local_path)
        repo_local_path.mkdir(parents=True)
            
        # Initialize git repository
        subprocess.run(['git', 'init'], cwd=repo_local_path, check=True)
        subprocess.run(['git', 'lfs', 'track', "*.json"], cwd=repo_local_path, check=True)
        
        # Configure git
        subprocess.run(['git', 'config', 'user.name', username], cwd=repo_local_path, check=True)
        subprocess.run(['git', 'config', 'user.email', f"{username}@users.noreply.huggingface.co"], 
                      cwd=repo_local_path, check=True)
        
        # Files to upload
        files_to_copy = [
            "README.md",
            "hindi_bpe.py",
            "byte_pair_encoder.py",
            "data_loader.py",
            "tokenizer/encoder.json",
            "tokenizer/vocab_stats.json",
            "requirements.txt"
        ]
        
        # Copy files
        for file in files_to_copy:
            src = Path(local_dir) / file
            if src.exists():
                dst = repo_local_path / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"Copied {file}")
        
        # Create .gitignore
        with open(repo_local_path / ".gitignore", "w") as f:
            f.write("__pycache__/\n*.pyc\n.DS_Store\n")
        
        # Create .gitattributes for LFS
        with open(repo_local_path / ".gitattributes", "w") as f:
            f.write("*.json filter=lfs diff=lfs merge=lfs -text\n")
        
        # Add metadata to README
        update_readme_with_metadata(repo_local_path, repo_id)
        
        # Git operations
        subprocess.run(['git', 'add', '.'], cwd=repo_local_path, check=True)
        subprocess.run(['git', 'commit', '-m', "Initial commit"], cwd=repo_local_path, check=True)
        
        # Set up remote and push
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=repo_local_path, check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'main'], 
                      cwd=repo_local_path, 
                      check=True, 
                      env={**os.environ, 'HUGGING_FACE_HUB_TOKEN': token})
        
        print(f"\nSuccessfully uploaded to {repo_url}")
        print(f"View your tokenizer at: https://huggingface.co/{repo_id}")
        
        return repo_url
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        raise

def update_readme_with_metadata(repo_path: Path, repo_id: str):
    """Add tokenizer metadata to README"""
    try:
        # Load tokenizer stats
        with open(repo_path / "tokenizer/vocab_stats.json", "r") as f:
            stats = json.load(f)
        
        # Read existing README
        readme_path = repo_path / "README.md"
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add Hugging Face specific metadata
        hf_metadata = f"""
            ## Hugging Face Integration 
            python
            from huggingface_hub import snapshot_download
            from hindi_bpe import HindiBPE
            Download tokenizer files
            snapshot_download(repo_id="REPO_ID", local_dir="tokenizer")
            Load tokenizer
            tokenizer = HindiBPE.load_tokenizer("tokenizer")
            Example usage
            text = "हिंदी भाषा बहुत सुंदर है।"
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)

            ## Model Statistics

                - Vocabulary Size: {stats['vocab_size']}
                - Average Token Length: {stats['avg_token_length']:.2f}
                - Compression Ratio: {stats['training_stats']['compression_ratios'][-1]:.2f}
                - Max Token Length: {stats['max_token_length']}
            """


         # Update README
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content + hf_metadata)
            
    except Exception as e:
        print(f"Warning: Could not update README with metadata: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload tokenizer to Hugging Face Hub")
    parser.add_argument("--repo-name", required=True, help="Repository name")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--org", help="Optional organization name")
    
    args = parser.parse_args()

    upload_tokenizer_to_hf(
        args.repo_name,
        args.token,
        args.org
    )


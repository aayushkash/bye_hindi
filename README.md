# Hindi Byte Pair Encoding (BPE) Tokenizer

A specialized BPE tokenizer for Hindi text that achieves efficient compression while maintaining linguistic coherence.

## Project Overview

This project implements a Byte Pair Encoding (BPE) tokenizer specifically designed for Hindi text. It features:
- Efficient trie-based tokenization
- Visualization of training progress
- Compression ratio optimization
- Support for large Hindi text datasets
- Hugging Face compatibility

## Project Structure 
hindi-bpe/
├── data/ # Dataset directory
│ ├── train/ # Training data
│ └── valid/ # Validation data
├── tokenizer/ # Saved tokenizer files
│ ├── encoder.json # Encoder state
│ └── vocab_stats.json # Vocabulary statistics
├── output/ # Visualization outputs
├── byte_pair_encoder.py # Core BPE implementation
├── hindi_bpe.py # Hindi-specific wrapper
├── app.py # Streamlit web interface
├── test_hindi_bpe.py # Test suite
└── requirements.txt # Dependencies


## File Descriptions

1. **byte_pair_encoder.py**
   - Core BPE implementation
   - Trie-based tokenization
   - Training statistics tracking
   - Visualization utilities

2. **hindi_bpe.py**
   - Hindi-specific tokenizer wrapper
   - Text preprocessing
   - Model saving/loading
   - Compression ratio calculation

3. **app.py**
   - Interactive web interface
   - Real-time tokenization
   - Training visualization
   - Model parameter tuning

4. **test_hindi_bpe.py**
   - Test suite for tokenizer
   - Performance benchmarks
   - Example usage

## Installation
    - bash
    - Clone repository
    - git clone https://github.com/yourusername/hindi-bpe.git
    - cd hindi-bpe
    - pip install -r requirements.txt

## Download and prepare dataset
    - python download_dataset.py
  
### Web Interface
    - streamlit run app.py

### Test-
    - python test_hindi_bpe.py
    - The test suite includes:
    - Training pipeline verification
    - Compression ratio validation
    - Token count requirements
    - Encoding/decoding accuracy

## Performance Metrics

    The tokenizer aims to achieve:
    - Vocabulary size < 5000 tokens
    - Compression ratio ≥ 3.2
    - Fast encoding/decoding
    - Memory-efficient operation

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
from hindi_bpe import HindiBPE, preprocess_hindi_text
from data_loader import load_hindi_dataset
from typing import List

class BPEVisualizer:
    def __init__(self):
        self.bpe = None
        
    def train_bpe(self, text: str, vocab_size: int):
        """Train BPE model and store it."""
        self.bpe = HindiBPE(vocab_size=vocab_size)
        self.bpe.train(text)
        
    def visualize_token_distribution(self) -> go.Figure:
        """Create token length distribution plot."""
        token_lengths = [len(token) for token in self.bpe.vocab]
        length_counts = Counter(token_lengths)
        
        df = pd.DataFrame({
            'Length': list(length_counts.keys()),
            'Count': list(length_counts.values())
        })
        
        fig = px.bar(df, x='Length', y='Count',
                    title='Token Length Distribution')
        return fig
    
    def visualize_compression_progress(self, text: str) -> go.Figure:
        """Visualize compression ratio progress during encoding."""
        encoded = self.bpe.encode(text)
        original_chars = len(text)
        encoded_chars = sum(len(token) for token in encoded)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Original', 'Encoded'],
            y=[original_chars, encoded_chars],
            name='Character Count'
        ))
        
        fig.update_layout(title='Text Compression Comparison')
        return fig
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize input text and return tokens."""
        if self.bpe is None:
            raise ValueError("BPE model not trained yet!")
        return self.bpe.encode(text)

def main():
    st.title("Hindi BPE Tokenizer Visualization")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    vocab_size = st.sidebar.slider("Vocabulary Size", 1000, 5000, 4500)
    num_articles = st.sidebar.slider("Number of Articles", 100, 5000, 1000)
    
    # Initialize visualizer
    visualizer = BPEVisualizer()
    
    # Load and train
    if st.button("Train New Model"):
        with st.spinner("Loading dataset..."):
            try:
                text = load_hindi_dataset(
                    split="train",
                    num_files=num_articles
                )
                text = preprocess_hindi_text(text)
                
                # Load some validation data
                valid_text = load_hindi_dataset(
                    split="valid",
                    num_files=min(num_articles // 5, 100)
                )
                
            except FileNotFoundError:
                st.error("Dataset not found! Please check the data directory structure")
                return
        
        with st.spinner("Training BPE..."):
            visualizer.train_bpe(text, vocab_size)
            
        st.success("Training completed!")
        
        # Show visualizations
        st.subheader("Token Length Distribution")
        st.plotly_chart(visualizer.visualize_token_distribution())
        
        st.subheader("Compression Results")
        st.plotly_chart(visualizer.visualize_compression_progress(text))
    
    # Interactive tokenization
    st.header("Try Tokenization")
    input_text = st.text_area("Enter Hindi text to tokenize:", height=150)
    
    if st.button("Tokenize") and visualizer.bpe is not None:
        if input_text:
            tokens = visualizer.tokenize_text(input_text)
            
            # Display tokens
            st.subheader("Tokenization Result")
            token_df = pd.DataFrame({
                'Token': tokens,
                'Length': [len(token) for token in tokens]
            })
            st.dataframe(token_df)
            
            # Show token visualization
            fig = px.scatter(token_df, x=range(len(tokens)), y='Length',
                           hover_data=['Token'],
                           title='Token Lengths in Sequence')
            st.plotly_chart(fig)
        else:
            st.warning("Please enter some text to tokenize")
    elif visualizer.bpe is None:
        st.warning("Please train the model first")

if __name__ == "__main__":
    main() 
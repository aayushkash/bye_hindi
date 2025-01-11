import gradio as gr
from huggingface_hub import snapshot_download
from hindi_bpe import HindiBPE, preprocess_hindi_text
import pandas as pd
import plotly.express as px
import os

# Download tokenizer if not exists
if not os.path.exists("tokenizer"):
    snapshot_download(
        repo_id="aayushraina/bpe-hindi",
        local_dir="tokenizer",
        allow_patterns=["*.json"]
    )

class TokenizerDemo:
    def __init__(self):
        self.tokenizer = HindiBPE.load_tokenizer("tokenizer")
        
    def tokenize_text(self, text: str) -> tuple:
        """Tokenize text and return visualization"""
        if not text:
            return "", None, "Please enter some text"
            
        # Preprocess
        text = preprocess_hindi_text(text)
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Create visualization
        token_df = pd.DataFrame({
            'Token': tokens,
            'Length': [len(token) for token in tokens]
        })
        
        fig = px.scatter(token_df, 
                        x=range(len(tokens)), 
                        y='Length',
                        hover_data=['Token'],
                        title='Token Lengths in Sequence')
        
        # Calculate statistics
        stats = {
            'Total Tokens': len(tokens),
            'Unique Tokens': len(set(tokens)),
            'Average Token Length': sum(len(t) for t in tokens) / len(tokens),
            'Compression Ratio': len(text) / sum(len(t) for t in tokens)
        }
        
        stats_str = "\n".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                             for k, v in stats.items())
        
        return (
            " ".join(tokens),  # Tokenized text
            fig,              # Visualization
            stats_str        # Statistics
        )
    
    def decode_tokens(self, tokens_text: str) -> str:
        """Decode space-separated tokens back to text"""
        if not tokens_text:
            return "Please tokenize some text first"
        tokens = tokens_text.split()
        return self.tokenizer.decode(tokens)

# Create Gradio interface
demo = TokenizerDemo()

interface = gr.Blocks(title="Hindi BPE Tokenizer")

with interface:
    gr.Markdown("""
    # Hindi BPE Tokenizer Demo
    
    This demo showcases a Byte Pair Encoding (BPE) tokenizer specifically trained for Hindi text.
    Enter Hindi text to see how it gets tokenized and analyze the token distribution.
    
    [View model on Hugging Face](https://huggingface.co/aayushraina/bpe-hindi)
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Hindi Text",
                placeholder="हिंदी में टेक्स्ट दर्ज करें...",
                lines=5
            )
            tokenize_btn = gr.Button("Tokenize")
        
        with gr.Column():
            tokens_output = gr.Textbox(
                label="Tokenized Output",
                lines=5
            )
            decode_btn = gr.Button("Decode")
            
    original_output = gr.Textbox(
        label="Decoded Text",
        lines=5
    )
    
    stats_output = gr.Textbox(
        label="Tokenization Statistics",
        lines=4
    )
    
    plot_output = gr.Plot(
        label="Token Length Distribution"
    )
    
    # Set up event handlers
    tokenize_btn.click(
        fn=demo.tokenize_text,
        inputs=input_text,
        outputs=[tokens_output, plot_output, stats_output]
    )
    
    decode_btn.click(
        fn=demo.decode_tokens,
        inputs=tokens_output,
        outputs=original_output
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["हिंदी भाषा बहुत सुंदर है।"],
            ["भारत एक विशाल देश है। यहाँ की संस्कृति बहुत पुरानी है।"],
            ["मैं हिंदी में प्रोग्रामिंग सीख रहा हूं।"]
        ],
        inputs=input_text
    )

# Launch the interface
interface.launch() 
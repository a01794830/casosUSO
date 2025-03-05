from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Hugging Face token
token = os.getenv("HUGGINGFACE_TOKEN")

def summarize_text(text):
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6", use_auth_token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6", use_auth_token=token)
    
    # Tokenize and generate summary
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
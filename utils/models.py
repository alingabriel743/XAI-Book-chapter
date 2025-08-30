import streamlit as st
import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    MarianMTModel, MarianTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer
)

warnings.filterwarnings('ignore')

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model and tokenizer"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Eroare la incarcarea modelului de sentiment: {str(e)}")
        return None, None

@st.cache_resource
def load_translation_model():
    """Load translation model and tokenizer"""
    try:
        model_name = "Helsinki-NLP/opus-mt-en-ro"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Eroare la incarcarea modelului de traducere: {str(e)}")
        return None, None

@st.cache_resource
def load_generation_model():
    """Load text generation model and tokenizer"""
    try:
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        st.error(f"Eroare la incarcarea modelului de generare: {str(e)}")
        return None, None

def predict_sentiment(text, tokenizer, model):
    """Make sentiment prediction"""
    if tokenizer is None or model is None:
        return None, None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return predictions, predicted_class

def translate_text(text, tokenizer, model):
    """Translate text from English to Romanian"""
    if tokenizer is None or model is None:
        return None, None
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        # Generate translation
        outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get attention outputs separately with proper decoder inputs
        try:
            # For MarianMT, we need to use the actual generated output as decoder input for attention analysis
            # Use the first few tokens of the generated translation
            generated_tokens = outputs[0][:min(10, len(outputs[0]))]  # First 10 tokens or less
            
            attention_outputs = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=generated_tokens.unsqueeze(0),
                output_attentions=True,
                return_dict=True
            )
        except Exception as e:
            print(f"Could not get attention outputs: {e}")
            # Try alternative approach with just start token
            try:
                decoder_start_token_id = model.config.decoder_start_token_id or model.config.bos_token_id
                if decoder_start_token_id is not None:
                    decoder_input_ids = torch.tensor([[decoder_start_token_id]])
                    attention_outputs = model(
                        input_ids=inputs['input_ids'],
                        decoder_input_ids=decoder_input_ids,
                        output_attentions=True,
                        return_dict=True
                    )
                else:
                    attention_outputs = None
            except Exception as e2:
                print(f"Alternative attention extraction also failed: {e2}")
                attention_outputs = None
    
    return translated_text, attention_outputs

def generate_text(prompt, tokenizer, model, max_length=50, temperature=0.7):
    """Generate text continuation"""
    if tokenizer is None or model is None:
        return None, None
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=len(inputs['input_ids'][0]) + max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs['attention_mask'],
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return generated_text, outputs

# Model status cache
@st.cache_data
def get_model_status():
    """Get status of all models"""
    return {
        "sentiment": {"loaded": False, "error": None},
        "translation": {"loaded": False, "error": None},
        "generation": {"loaded": False, "error": None}
    }
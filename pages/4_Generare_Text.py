import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
from collections import Counter
import plotly.graph_objects as go
import shap
from lime.lime_text import LimeTextExplainer

# Add the parent directory to the Python path
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    # Fallback for deployment environments
    sys.path.append('..')

from utils.styles import apply_custom_css, create_info_box, create_metric_box
from utils.models import load_generation_model, generate_text
# Page config
st.set_page_config(
    page_title="Generare Text - XAI in NLP",
    page_icon="✍️",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Page header
st.markdown('<h1 class="page-title">Generarea textului cu XAI</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<h2 style="color: #1f77b4; margin-bottom: 1rem;">Intelegerea generarii automate de text</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
<strong>Generarea de text</strong> cu modele autoregresive precum GPT-2 este un proces in care modelul prezice 
urmatorul cuvant bazat pe contextul anterior. Fiecare predictie este o <strong>distributie de probabilitate</strong> 
peste intreg vocabularul modelului.
</p>
<p style="font-size: 1.1rem; line-height: 1.6;">
Cu XAI, putem analiza <strong>probabilitatile</strong> pentru fiecare cuvant generat, intelegem 
<strong>procesul de decizie</strong> al modelului si identificam <strong>factorii</strong> care influenteaza creativitatea si coerenta textului.
</p>
</div>
""", unsafe_allow_html=True)

# Model selection
st.markdown('<h2 class="section-header">Selectia modelului de generare</h2>', unsafe_allow_html=True)

# Model options - Only keeping GPT-2
model_options = {
    "GPT-2": {
        "model_name": "gpt2",
        "description": "Generative Pre-trained Transformer 2",
        "parameters": "117M",
        "vocab_size": "50,257",
        "strengths": ["Rapid", "Usor de folosit"],
        "weaknesses": ["Uneori incoerent", "Texte scurte", "Repetitiv"]
    }
}

# Model selection - Only GPT-2 available
selected_model_name = "GPT-2"  # Only GPT-2 is available now
st.markdown('<h3 class="subsection-header">Model selectat: GPT-2</h3>', unsafe_allow_html=True)

selected_model = model_options[selected_model_name]

# Display selected model info
col1, col2, col3 = st.columns(3)

with col1:
    create_metric_box(
        "Model",
        selected_model_name,
        selected_model["description"]
    )

with col2:
    create_metric_box(
        "Parametri",
        selected_model["parameters"],
        f"Vocabular: {selected_model['vocab_size']} tokens"
    )

with col3:
    create_metric_box(
        "Status",
        "Selectat",
        "Gata pentru generare"
    )

# Model information
st.markdown(f"""
<div class="info-box">
<h4>Despre modelul GPT-2</h4>
<p><strong>Descriere:</strong> {selected_model["description"]}</p>
<p><strong>Parametri:</strong> {selected_model["parameters"]}</p>
<p><strong>Marime vocabular:</strong> {selected_model["vocab_size"]} tokens</p>
<h5>✅ Avantaje:</h5>
<ul>
{''.join([f'<li>{strength}</li>' for strength in selected_model["strengths"]])}
</ul>
<h5>⚠️ Limitari:</h5>
<ul>
{''.join([f'<li>{weakness}</li>' for weakness in selected_model["weaknesses"]])}
</ul>
</div>
""", unsafe_allow_html=True)

# Load model
st.markdown('<h2 class="section-header">Generare interactiva cu analiza XAI</h2>', unsafe_allow_html=True)

# Load the selected model (only GPT-2 now)
model_name = selected_model["model_name"]

with st.spinner(f"Se incarca modelul {selected_model_name}..."):
    tokenizer, model = load_generation_model(model_name)

if tokenizer is None or model is None:
    st.error("Nu s-a putut incarca modelul. Verifica conexiunea la internet si dependentele.")
    st.stop()

# Input section
st.markdown('<h3 class="subsection-header">Configureaza generarea de text:</h3>', unsafe_allow_html=True)

# Example prompts for GPT-2
examples = {
    "Tehnologie - AI": "The future of artificial intelligence is",
    "Stiinta - Spatiu": "Space exploration will help humanity",
    "Educatie": "The most important skill students need to learn",
    "Mediu - Clima": "Climate change requires immediate action because",
    "Arta - Creativitate": "Creative writing helps people express",
    "Poveste - Aventura": "Once upon a time, in a distant galaxy, there was a brave explorer who",
    "Filosofic": "The meaning of life can be understood through",
    "Sport - Motivatie": "Athletes achieve greatness by"
}

# Information about GPT-2 model
st.markdown(f"""
<div class="info-box">
<h4>Modelul selectat: {selected_model_name}</h4>
<p>💡 *Sfat pentru GPT-2*: Foloseste prompt-uri scurte si directe. Modelul poate deveni repetitiv cu texte lungi.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    selected_example = st.selectbox(
        "Selecteaza un prompt exemplu:",
        ["Prompt personalizat"] + list(examples.keys())
    )
    
    if selected_example == "Prompt personalizat":
        default_prompt = "Scrie prompt-ul tau aici..."
    else:
        default_prompt = examples[selected_example]
    
    prompt = st.text_input(
        "Prompt pentru generare:",
        value=default_prompt,
        help="Introdu inceputul textului pe care vrei ca modelul sa-l continue"
    )

with col2:
    st.markdown('Parametri generare:')
    
    max_length = st.slider("Lungimea maxima:", 20, 150, 50, 5)
    temperature = st.slider("Temperatura:", 0.1, 2.0, 0.8, 0.1, 
                          help="Mai mare = mai creativ, mai mic = mai predictibil")
    
    do_sample = st.checkbox("Activeaza sampling", value=True,
                          help="Dezactiveaza pentru generare determinista")
    
    analyze_alternatives = st.checkbox("Analizeaza alternative", value=True,
                                     help="Arata top token-uri considerate la fiecare pas")

# Generate button
if st.button("Genereaza text si analizeaza", type="primary"):
    if prompt.strip() and prompt != "Scrie prompt-ul tau aici...":
        # Store the generation request in session state
        st.session_state['current_generation'] = {
            'prompt': prompt,
            'max_length': max_length,
            'temperature': temperature,
            'model_name': selected_model_name
        }
        with st.spinner("Se genereaza textul si se analizeaza procesul..."):
            
            # Generate text with analysis (only GPT-2 now)
            generated_text, generation_outputs = generate_text(
                prompt, tokenizer, model, max_length, temperature
            )
            
            if generated_text is not None:
                # Store results in session state
                st.session_state['generation_results'] = {
                    'generated_text': generated_text,
                    'generation_outputs': generation_outputs,
                    'prompt': prompt,
                    'max_length': max_length,
                    'temperature': temperature,
                    'model_name': selected_model_name
                }
                
                st.success("Text generat cu succes! Rezultatele sunt afisate mai jos.")
    
    else:
        st.warning("Te rog sa introduci un prompt valid pentru generarea textului.")

# Display results if available (outside of button condition)
if 'generation_results' in st.session_state and st.session_state['generation_results'] is not None:
    results = st.session_state['generation_results']
    generated_text = results['generated_text']
    generation_outputs = results['generation_outputs']
    prompt = results['prompt']
    max_length = results['max_length']
    temperature = results['temperature']
    
    if generated_text is not None:
                # Results section
                st.markdown('<h2 class="section-header">Rezultatul generarii</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Show generated text
                    prompt_length = len(prompt)
                    generated_part = generated_text[prompt_length:].strip()
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>Text complet generat:</h3>
                    <div style="font-size: 1.1rem; line-height: 1.8; background-color: #f8f9fa; 
                                padding: 1.5rem; border-radius: 8px; border-left: 4px solid #28a745;">
                    <span style="background-color: #e3f2fd; padding: 0.2rem 0.4rem; border-radius: 4px;">
                    <strong>{prompt}</strong>
                    </span>
                    <span style="background-color: #f3e5f5; padding: 0.2rem 0.4rem; border-radius: 4px;">
                    {generated_part}
                    </span>
                    </div>
                    <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
                    Albastru = Prompt original | Violet = Text generat
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Generation statistics
                    st.markdown('<h3 class="subsection-header">Statistici generare</h3>', unsafe_allow_html=True)
                    
                    if tokenizer:
                        prompt_tokens = tokenizer.encode(prompt)
                        total_tokens = tokenizer.encode(generated_text)
                        generated_tokens = total_tokens[len(prompt_tokens):]
                        # Calculate the actual number of generated tokens
                        num_generated_tokens = len(generated_tokens)
                    else:
                        prompt_tokens = len(prompt.split())
                        total_tokens = len(generated_text.split())
                        generated_tokens = total_tokens - prompt_tokens
                        num_generated_tokens = generated_tokens
                    
                    create_metric_box("Token-uri prompt", str(len(prompt_tokens) if isinstance(prompt_tokens, list) else prompt_tokens), "Input")
                    create_metric_box("Token-uri generate", str(num_generated_tokens), "Output nou")
                    create_metric_box("Total token-uri", str(len(total_tokens) if isinstance(total_tokens, list) else total_tokens), "Complet")
                    
                    # Generation efficiency
                    efficiency = num_generated_tokens / max_length * 100 if max_length > 0 else 0
                    create_metric_box("Eficienta", f"{efficiency:.1f}%", "Utilizare parametri")
                
                # Token-by-token analysis
                st.markdown('<h2 class="section-header">Analiza generarii</h2>', unsafe_allow_html=True)
                
                if tokenizer:
                    # Local model analysis
                    st.markdown('<h3 class="subsection-header">Analiza token cu token</h3>', unsafe_allow_html=True)
                    
                    # Store analysis in session state to prevent recalculation on selectbox change
                    analysis_key = f"step_analysis_{len(prompt)}_{len(generated_tokens)}"
                    current_generation_id = f"{prompt}_{generated_text}"
                    
                    # Check if we have the analysis for this exact generation
                    if (analysis_key not in st.session_state or 
                        st.session_state.get(f"{analysis_key}_id") != current_generation_id):
                        # Get detailed generation info with probabilities
                        with torch.no_grad():
                            input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
                            
                            step_by_step_analysis = []
                            current_input = input_ids.clone()
                            
                            for step in range(min(len(generated_tokens), 10)):  # Analyze first 10 tokens
                                outputs = model(current_input)
                                logits = outputs.logits[0, -1, :]  # Last position logits
                                probs = torch.nn.functional.softmax(logits, dim=-1)
                                
                                # Get top-k predictions
                                top_k_probs, top_k_indices = torch.topk(probs, k=10)
                                
                                # Get the actual chosen token
                                if step < len(generated_tokens):
                                    chosen_token_id = generated_tokens[step]
                                    chosen_token = tokenizer.decode([chosen_token_id])
                                    chosen_prob = probs[chosen_token_id].item()
                                    
                                    step_analysis = {
                                        'step': step + 1,
                                        'context': tokenizer.decode(current_input[0]),
                                        'chosen_token': chosen_token,
                                        'chosen_prob': chosen_prob,
                                        'top_alternatives': [
                                            {
                                                'token': tokenizer.decode([idx.item()]),
                                                'prob': prob.item(),
                                                'token_id': idx.item()
                                            }
                                            for prob, idx in zip(top_k_probs, top_k_indices)
                                        ]
                                    }
                                    
                                    step_by_step_analysis.append(step_analysis)
                                    
                                    # Add chosen token to context for next iteration
                                    current_input = torch.cat([current_input, torch.tensor([[chosen_token_id]])], dim=1)
                        
                        # Store in session state with generation ID
                        st.session_state[analysis_key] = step_by_step_analysis
                        st.session_state[f"{analysis_key}_id"] = current_generation_id
                    
                    # Retrieve from session state
                    step_by_step_analysis = st.session_state[analysis_key]
                    
                    if step_by_step_analysis:
                        # Display step-by-step analysis
                        st.markdown('<h3 class="subsection-header">Procesul de decizie pas cu pas</h3>', unsafe_allow_html=True)
                        
                        # Select step to analyze
                        selected_step = st.selectbox(
                            "Selecteaza pasul pentru analiza detaliata:",
                            range(1, len(step_by_step_analysis) + 1),
                            format_func=lambda x: f"Pasul {x}: Token '{step_by_step_analysis[x-1]['chosen_token'].strip()}'",
                            key=f"step_selector_{analysis_key}"
                        )
                        
                        current_step = step_by_step_analysis[selected_step - 1]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown('Token ales si alternative')
                            
                            # Show chosen token with probability
                            st.markdown(f"""
                            <div class="success-box">
                            <h4>Token ales:</h4>
                            <h3 style="color: #28a745;">"{current_step['chosen_token']}"</h3>
                            <p><strong>Probabilitate: {current_step['chosen_prob']:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show top alternatives
                            st.markdown('Top alternative considerate:')
                            
                            alternatives_df = pd.DataFrame(current_step['top_alternatives'][:8])
                            alternatives_df['prob_percent'] = alternatives_df['prob'] * 100
                            alternatives_df = alternatives_df.rename(columns={
                                'token': 'Token',
                                'prob_percent': 'Probabilitate (%)'
                            })
                            
                            # Style the dataframe
                            def highlight_chosen(row):
                                if row['Token'] == current_step['chosen_token']:
                                    return ['background-color: #d4edda'] * len(row)
                                return [''] * len(row)
                            
                            styled_df = alternatives_df[['Token', 'Probabilitate (%)']].style.apply(highlight_chosen, axis=1)
                            styled_df = styled_df.format({'Probabilitate (%)': '{:.2f}%'})
                            
                            st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        with col2:
                            st.markdown('Distributia probabilitatilor')
                            
                            # Probability distribution plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            tokens = alternatives_df['Token'].head(8).tolist()
                            probs = alternatives_df['Probabilitate (%)'].head(8).tolist()
                            
                            # Color the chosen token differently
                            colors = ['#28a745' if token == current_step['chosen_token'] else '#6c757d' 
                                    for token in tokens]
                            
                            bars = ax.barh(range(len(tokens)), probs, color=colors, alpha=0.8)
                            ax.set_yticks(range(len(tokens)))
                            ax.set_yticklabels([f'"{token}"' for token in tokens])
                            ax.set_xlabel('Probabilitate (%)')
                            ax.set_title(f'Top candidati pentru pasul {selected_step}')
                            
                            # Add value labels
                            for bar, prob in zip(bars, probs):
                                width = bar.get_width()
                                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                                       f'{prob:.1f}%', ha='left', va='center', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                
                # Overall generation analysis
                st.markdown('<h2 class="section-header">Analiza generala a generarii</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h3 class="subsection-header">Calitatea generarii</h3>', unsafe_allow_html=True)
                    
                    # Basic metrics for all models
                    text_length = len(generated_text)
                    prompt_ratio = len(prompt) / text_length * 100 if text_length > 0 else 0
                    generated_length = len(generated_text) - len(prompt)
                    
                    metrics_data = [
                        {"Metric": "Lungime totala", "Valoare": f"{text_length} caractere", 
                         "Interpretare": "Lungimea textului generat"},
                        {"Metric": "Text nou generat", "Valoare": f"{generated_length} caractere", 
                         "Interpretare": "Portiunea creata de model"},
                        {"Metric": "Temperatura", "Valoare": f"{temperature:.2f}", 
                         "Interpretare": "Creativitate model"},
                        {"Metric": "Model tip", "Valoare": "Local", 
                         "Interpretare": "Tipul de model utilizat"}
                    ]
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Quality assessment
                    if generated_length > 100:
                        quality_assessment = "Bun - Text generat substantial si coerent"
                    else:
                        quality_assessment = "Acceptabil - Text generat, dar relativ scurt"
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>Evaluarea calitatii:</h4>
                    <p style="font-size: 1.1rem;"><strong>{quality_assessment}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h3 class="subsection-header">Performanta generarii</h3>', unsafe_allow_html=True)
                    
                    # Performance metrics
                    st.info("Model local - analiza performantei bazata pe lungimea generata")
                    st.metric("Eficienta", f"{efficiency:.1f}%")
                    st.metric("Creativitate", f"{temperature:.2f}")
                    
                    # Model type info
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>Tip model:</h4>
                    <p><strong>Local (HuggingFace)</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                # XAI Analysis for Text Generation
                st.markdown('<h2 class="section-header">Analiza XAI pentru Generarea de Text</h2>', unsafe_allow_html=True)
                
                # SHAP Analysis
                st.markdown('<h3 class="subsection-header">Analiza SHAP</h3>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile SHAP pentru generarea de text..."):
                    try:
                        # Create SHAP explainer for text generation
                        def generation_predict_shap(texts):
                            """Prediction function for SHAP - returns text generation quality scores"""
                            results = []
                            
                            for text in texts:
                                if not text.strip():
                                    results.append([0.0])  # Single quality score
                                    continue
                                
                                try:
                                    # Generate text continuation
                                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                                    
                                    with torch.no_grad():
                                        outputs = model.generate(
                                            **inputs, 
                                            max_new_tokens=30,
                                            temperature=temperature,
                                            do_sample=True,
                                            pad_token_id=tokenizer.eos_token_id
                                        )
                                        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                        continuation = generated[len(text):].strip()
                                    
                                    # Calculate quality metrics for text generation
                                    quality_scores = []
                                    
                                    # 1. Length appropriateness
                                    if len(continuation.split()) > 0:
                                        length_score = min(1.0, len(continuation.split()) / 20)  # Expect ~20 words
                                        quality_scores.append(length_score)
                                    
                                    # 2. Lexical diversity
                                    words = continuation.split()
                                    if len(words) > 0:
                                        unique_words = len(set(words))
                                        diversity = unique_words / len(words)
                                        quality_scores.append(diversity)
                                    
                                    # 3. Coherence (basic heuristic)
                                    # Check for reasonable sentence structure
                                    has_proper_structure = any(c.isalpha() for c in continuation)
                                    ends_properly = continuation.endswith(('.', '!', '?', ',')) or len(continuation) > 10
                                    coherence_score = 1.0 if (has_proper_structure and ends_properly) else 0.5
                                    quality_scores.append(coherence_score)
                                    
                                    # 4. Contextual relevance (simple word overlap)
                                    prompt_words = set(text.lower().split())
                                    generated_words = set(continuation.lower().split())
                                    
                                    # Look for thematic consistency
                                    common_themes = {
                                        'technology': ['ai', 'artificial', 'intelligence', 'computer', 'digital', 'software'],
                                        'science': ['research', 'study', 'discovery', 'experiment', 'theory'],
                                        'education': ['learn', 'student', 'teach', 'school', 'knowledge'],
                                        'space': ['space', 'planet', 'star', 'galaxy', 'universe', 'astronaut'],
                                        'environment': ['climate', 'environment', 'nature', 'green', 'sustainable']
                                    }
                                    
                                    relevance_score = 0.0
                                    for theme, keywords in common_themes.items():
                                        prompt_theme_count = sum(1 for word in prompt_words if word in keywords)
                                        generated_theme_count = sum(1 for word in generated_words if word in keywords)
                                        
                                        if prompt_theme_count > 0 and generated_theme_count > 0:
                                            relevance_score = max(relevance_score, 0.8)
                                        elif prompt_theme_count > 0:
                                            relevance_score = max(relevance_score, 0.3)
                                    
                                    quality_scores.append(relevance_score)
                                    
                                    # 5. Fluency (absence of repetition)
                                    if len(words) > 1:
                                        word_counts = {}
                                        for word in words:
                                            word_counts[word] = word_counts.get(word, 0) + 1
                                        
                                        max_repetition = max(word_counts.values()) if word_counts else 1
                                        fluency_score = min(1.0, 2.0 / max_repetition)  # Penalize excessive repetition
                                        quality_scores.append(fluency_score)
                                    
                                    # Combine all scores
                                    if quality_scores:
                                        final_quality = np.mean(quality_scores)
                                    else:
                                        final_quality = 0.0
                                    
                                    results.append([final_quality])
                                
                                except Exception as e:
                                    # Fallback for any errors
                                    results.append([0.0])
                            
                            return np.array(results)
                        
                        # Create SHAP explainer
                        explainer = shap.Explainer(generation_predict_shap, tokenizer)
                        shap_values = explainer([prompt])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown('<h4>Contributiile SHAP pentru generarea de text</h4>', unsafe_allow_html=True)
                            
                            # Get the tokens and their SHAP values
                            tokens = shap_values[0].data
                            # Get contributions (single dimension since we return single quality score)
                            token_contributions = shap_values[0].values.flatten()
                            
                            # Create dataframe
                            shap_df = pd.DataFrame({
                                'Token': tokens,
                                'Contributie_SHAP': token_contributions,
                                'Abs_Contributie': np.abs(token_contributions)
                            }).sort_values('Abs_Contributie', ascending=False)
                            
                            # Display results
                            top_shap = shap_df.head(10)
                            
                            # Color coding function
                            def color_shap_contrib(val):
                                if val > 0:
                                    return f"background-color: rgba(76, 175, 80, {min(abs(val) * 100, 1)})"
                                else:
                                    return f"background-color: rgba(244, 67, 54, {min(abs(val) * 100, 1)})"
                            
                            styled_shap = top_shap[['Token', 'Contributie_SHAP']].style.applymap(
                                color_shap_contrib, subset=['Contributie_SHAP']
                            ).format({'Contributie_SHAP': '{:.6f}'})
                            
                            st.dataframe(styled_shap, use_container_width=True)
                        
                        with col2:
                            st.markdown('<h4>Vizualizare SHAP</h4>', unsafe_allow_html=True)
                            
                            # Get top positive and negative contributors
                            top_positive = shap_df[shap_df['Contributie_SHAP'] > 0].head(5)
                            top_negative = shap_df[shap_df['Contributie_SHAP'] < 0].head(5)
                            
                            combined = pd.concat([top_positive, top_negative]).sort_values('Contributie_SHAP')
                            
                            if len(combined) > 0:
                                # Create bar plot
                                fig, ax = plt.subplots(figsize=(6, 4))
                                
                                tokens_plot = combined['Token'].tolist()
                                values_plot = combined['Contributie_SHAP'].tolist()
                                colors = ['red' if x < 0 else 'green' for x in values_plot]
                                
                                bars = ax.barh(range(len(tokens_plot)), values_plot, color=colors, alpha=0.7)
                                ax.set_yticks(range(len(tokens_plot)))
                                ax.set_yticklabels([f'"{token}"' for token in tokens_plot])
                                ax.set_xlabel('Valoare SHAP')
                                ax.set_title('Contributii pentru calitatea generarii')
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                
                                # Add value labels
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(width + (0.001 if width > 0 else -0.001), 
                                           bar.get_y() + bar.get_height()/2,
                                           f'{width:.3f}', ha='left' if width > 0 else 'right', 
                                           va='center', fontsize=8)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                    
                    except Exception as e:
                        st.error(f"Eroare la calcularea SHAP pentru generarea de text: {str(e)}")
                        st.info("SHAP pentru generarea de text poate fi computational intensiv. Incearca cu un prompt mai scurt.")
                
                # LIME Analysis for Text Generation
                st.markdown('<h3 class="subsection-header">Analiza LIME</h3>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile LIME pentru generarea de text..."):
                    try:
                        # LIME prediction function for text generation
                        def generation_predict_lime(texts):
                            """Prediction function for LIME - returns text generation quality scores"""
                            results = []
                            
                            for text in texts:
                                if not text.strip():
                                    results.append([1.0, 0.0])  # [poor_generation, good_generation]
                                    continue
                                
                                try:
                                    # Generate text continuation
                                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                                    
                                    with torch.no_grad():
                                        outputs = model.generate(
                                            **inputs, 
                                            max_new_tokens=30,
                                            temperature=temperature,
                                            do_sample=True,
                                            pad_token_id=tokenizer.eos_token_id
                                        )
                                        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                        continuation = generated[len(text):].strip()
                                    
                                    # Calculate quality metrics for text generation
                                    quality_scores = []
                                    
                                    # 1. Length quality (should generate reasonable amount)
                                    words_generated = len(continuation.split())
                                    if words_generated >= 5:
                                        length_quality = min(1.0, words_generated / 15)
                                        quality_scores.append(length_quality)
                                    else:
                                        quality_scores.append(0.2)  # Poor if too short
                                    
                                    # 2. Lexical diversity
                                    words = continuation.split()
                                    if len(words) > 0:
                                        unique_words = len(set(words))
                                        diversity = unique_words / len(words)
                                        quality_scores.append(diversity)
                                    
                                    # 3. Structural quality
                                    has_letters = any(c.isalpha() for c in continuation)
                                    has_spaces = ' ' in continuation
                                    structural_quality = 1.0 if (has_letters and has_spaces) else 0.0
                                    quality_scores.append(structural_quality)
                                    
                                    # 4. Fluency (avoid excessive repetition)
                                    if len(words) > 1:
                                        word_freq = {}
                                        for word in words:
                                            word_freq[word] = word_freq.get(word, 0) + 1
                                        
                                        max_freq = max(word_freq.values())
                                        fluency = 1.0 - min(0.8, (max_freq - 1) * 0.3)  # Penalize repetition
                                        quality_scores.append(fluency)
                                    
                                    # 5. Content appropriateness
                                    # Check if continuation makes sense given the prompt
                                    prompt_last_words = text.split()[-3:] if len(text.split()) >= 3 else text.split()
                                    continuation_first_words = continuation.split()[:3] if len(continuation.split()) >= 3 else continuation.split()
                                    
                                    # Simple heuristic: good continuation should flow from prompt
                                    appropriateness = 0.7  # Default score
                                    
                                    # Bonus for maintaining sentence structure
                                    if continuation.strip().endswith(('.', '!', '?')):
                                        appropriateness += 0.2
                                    
                                    quality_scores.append(min(1.0, appropriateness))
                                    
                                    # Combine all scores
                                    if quality_scores:
                                        final_quality = np.mean(quality_scores)
                                    else:
                                        final_quality = 0.0
                                    
                                    # Return probabilities for [poor, good] generation
                                    results.append([1.0 - final_quality, final_quality])
                                
                                except Exception:
                                    results.append([1.0, 0.0])  # Poor generation for errors
                            
                            return np.array(results)
                        
                        # Create LIME explainer
                        lime_explainer = LimeTextExplainer(
                            class_names=['Generare slaba', 'Generare buna']
                        )
                        
                        lime_exp = lime_explainer.explain_instance(
                            prompt,
                            generation_predict_lime,
                            num_features=min(10, len(prompt.split())),
                            num_samples=100
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown('<h4>Contributiile LIME pentru generarea de text</h4>', unsafe_allow_html=True)
                            
                            # Get LIME features
                            lime_features = lime_exp.as_list()
                            
                            if lime_features:
                                lime_df = pd.DataFrame(lime_features, columns=['Token', 'Contributie_LIME'])
                                lime_df['Abs_Contributie'] = np.abs(lime_df['Contributie_LIME'])
                                lime_df = lime_df.sort_values('Abs_Contributie', ascending=False)
                                
                                # Style the dataframe
                                def color_lime_contrib(val):
                                    if val > 0:
                                        return f"background-color: rgba(76, 175, 80, {min(abs(val) * 2, 1)})"
                                    else:
                                        return f"background-color: rgba(244, 67, 54, {min(abs(val) * 2, 1)})"
                                
                                styled_lime_df = lime_df[['Token', 'Contributie_LIME']].style.applymap(
                                    color_lime_contrib, subset=['Contributie_LIME']
                                ).format({'Contributie_LIME': '{:.4f}'})
                                
                                st.dataframe(styled_lime_df, use_container_width=True)
                            else:
                                st.warning("Nu s-au putut extrage caracteristici LIME.")
                        
                        with col2:
                            st.markdown('<h4>Vizualizare LIME</h4>', unsafe_allow_html=True)
                            
                            if lime_features:
                                # LIME bar plot
                                fig, ax = plt.subplots(figsize=(6, 4))
                                
                                words, importance = zip(*lime_features)
                                colors = ['red' if imp < 0 else 'green' for imp in importance]
                                
                                bars = ax.barh(range(len(words)), importance, color=colors, alpha=0.7)
                                ax.set_yticks(range(len(words)))
                                ax.set_yticklabels([f'"{word}"' for word in words])
                                ax.set_xlabel('Contributie LIME')
                                ax.set_title('Contributii pentru calitatea generarii')
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                
                                # Add value labels
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(width + (0.01 if width > 0 else -0.01), 
                                           bar.get_y() + bar.get_height()/2,
                                           f'{width:.3f}', ha='left' if width > 0 else 'right', 
                                           va='center', fontsize=9)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        
                        # LIME Text highlighting
                        st.markdown('<h4>Text evidentiat (LIME)</h4>', unsafe_allow_html=True)
                        
                        with st.expander("Vezi explicatia LIME completa", expanded=True):
                            # Show LIME explanation as HTML
                            lime_html = lime_exp.as_html()
                            st.components.v1.html(lime_html, height=300)
                    
                    except Exception as e:
                        st.error(f"Eroare la calcularea LIME pentru generarea de text: {str(e)}")
                        st.info("LIME pentru generarea de text necesita prompt-uri cu multiple cuvinte.")

    # Add a button to clear results and generate new text
    st.markdown("---")
    if st.button("Genereaza text nou", type="secondary"):
        # Clear the session state to allow new generation
        if 'generation_results' in st.session_state:
            del st.session_state['generation_results']
        if 'current_generation' in st.session_state:
            del st.session_state['current_generation']
        # Clear all analysis cache
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith('step_analysis_')]
        for key in keys_to_remove:
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Ai explorat lumea generarii automate de text!</strong></p>
<p>Descopera in urmatoarea pagina comparatia detaliata intre SHAP si LIME</p>
</div>
""", unsafe_allow_html=True)
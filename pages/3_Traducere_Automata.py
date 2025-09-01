import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import shap
from lime.lime_text import LimeTextExplainer

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styles import apply_custom_css, create_info_box, create_metric_box
from utils.models import load_translation_model, translate_text

# Page config
st.set_page_config(
    page_title="Traducere Automata - XAI in NLP",
    page_icon="🌐",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Page header
st.markdown('<h1 class="page-title">Traducere automata cu XAI</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<h2 style="color: #1f77b4; margin-bottom: 1rem;">Intelegerea traducerii automate</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
<strong>Traducerea automata neurala</strong> foloseste retele neuronale complexe pentru a traduce text dintr-o limba in alta. 
Modelele moderne, precum Transformer, folosesc mecanisme de <strong>atentie</strong> pentru a decide care cuvinte 
din limba sursa sunt importante pentru traducerea fiecarui cuvant din limba tinta.
</p>
<p style="font-size: 1.1rem; line-height: 1.6;">
Cu ajutorul XAI, putem <strong>vizualiza</strong> aceste mecanisme de atentie si <strong>intelege</strong> 
cum modelul "gandeste" in timpul procesului de traducere.
</p>
</div>
""", unsafe_allow_html=True)

# Model information
st.markdown('<h2 class="section-header">Modelul de traducere</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    create_metric_box(
        "Model",
        "MarianMT",
        "Helsinki-NLP/opus-mt-en-ro"
    )

with col2:
    create_metric_box(
        "Traducere",
        "EN -> RO",
        "Engleza catre Romana"
    )

with col3:
    create_metric_box(
        "Arhitectura",
        "Transformer",
        "Cu mecanisme de atentie"
    )

# How attention works
st.markdown('<h2 class="section-header">Cum functioneaza atentia in traducere</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="success-box">
    <h3>Mecanismul de atentie</h3>
    <p><strong>Atentia</strong> permite modelului sa se "concentreze" pe diferite parti ale textului sursa 
    cand genereaza fiecare cuvant din traducere.</p>
    
    <h4>Pasii procesului:</h4>
    <ol>
    <li><strong>Encoding:</strong> Textul sursa este transformat in reprezentari vectoriale</li>
    <li><strong>Attention Weights:</strong> Modelul calculeaza cat de important este fiecare cuvant sursa</li>
    <li><strong>Context Vector:</strong> Se creaza o combinatie ponderata a cuvintelor sursa</li>
    <li><strong>Decoding:</strong> Se genereaza cuvantul din traducere bazat pe context</li>
    </ol>
    
    <h4>Tipuri de atentie:</h4>
    <ul>
    <li><strong>Self-Attention:</strong> Cuvintele se "uita" unele la altele in aceeasi secventa</li>
    <li><strong>Cross-Attention:</strong> Cuvintele din traducere se "uita" la cuvintele sursa</li>
    <li><strong>Multi-Head:</strong> Mai multe mecanisme de atentie in paralel</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Attention visualization diagram
    st.markdown('<h3 class="subsection-header">Schema atentiei</h3>', unsafe_allow_html=True)
    
    # Create a simple attention heatmap example
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Example attention pattern
    source_words = ["The", "cat", "is", "sleeping"]
    target_words = ["Pisica", "doarme"]
    
    # Simulate attention weights
    attention_matrix = np.array([
        [0.1, 0.7, 0.1, 0.1],  # "Pisica" attends to
        [0.05, 0.1, 0.15, 0.7]   # "doarme" attends to
    ])
    
    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(source_words)))
    ax.set_yticks(range(len(target_words)))
    ax.set_xticklabels(source_words)
    ax.set_yticklabels(target_words)
    ax.set_xlabel('Cuvinte sursa (EN)')
    ax.set_ylabel('Cuvinte tinta (RO)')
    ax.set_title('Exemplu pattern atentie')
    
    # Add text annotations
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                         ha="center", va="center", color="white" if attention_matrix[i, j] > 0.5 else "black")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Load model
st.markdown('<h2 class="section-header">Traducere interactiva cu analiza XAI</h2>', unsafe_allow_html=True)

with st.spinner("Se incarca modelul de traducere..."):
    tokenizer, model = load_translation_model()

if tokenizer is None or model is None:
    st.error("Nu s-a putut incarca modelul. Verifica conexiunea la internet si dependentele.")
    st.stop()

# Text input section
st.markdown('<h3 class="subsection-header">Introdu textul pentru traducere:</h3>', unsafe_allow_html=True)

# Example texts
examples = {
    "Simplu - Salut": "Hello, how are you today?",
    "Profesional - Afaceri": "We would like to schedule a meeting to discuss the project details and timeline.",
    "Tehnic - IT": "The artificial intelligence model uses neural networks to process natural language.",
    "Literar - Poveste": "Once upon a time, in a faraway kingdom, there lived a wise old wizard who could speak to animals.",
    "Conversational": "I love spending time with my family during weekends, especially when we go hiking in the mountains.",
    "Stiinta": "The research demonstrates that climate change significantly affects biodiversity in tropical regions."
}

selected_example = st.selectbox(
    "Selecteaza un exemplu sau scrie propriul text:",
    ["Text personalizat"] + list(examples.keys())
)

if selected_example == "Text personalizat":
    default_text = "Write your English text here..."
else:
    default_text = examples[selected_example]

text_input = st.text_area(
    "Text in engleza:",
    value=default_text,
    height=100,
    help="Introdu un text in limba engleza pentru traducere in romana"
)

# Parameters
with st.expander("Parametri avansati", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Lungimea maxima traducere:", 50, 200, 100)
        num_layers = st.slider("Layere de analizat:", 1, 6, 6)
    
    with col2:
        num_heads = st.slider("Capete de atentie:", 1, 8, 8)
        temperature = st.slider("Temperatura generare:", 0.1, 2.0, 1.0, 0.1)

# Translation button
if st.button("Tradu si analizeaza", type="primary"):
    if text_input.strip() and text_input != "Write your English text here...":
        with st.spinner("Se traduce textul si se analizeaza atentia..."):
            
            # Perform translation
            translated_text, attention_outputs = translate_text(text_input, tokenizer, model)
            
            if translated_text is not None:
                # Results section
                st.markdown('<h2 class="section-header">Rezultatul traducerii</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>Text original (EN):</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6; background-color: #f8f9fa; 
                              padding: 1rem; border-radius: 8px; border-left: 4px solid #007bff;">
                    "{text_input}"
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h3>Traducere (RO):</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6; background-color: #f8f9fa; 
                              padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745;">
                    "{translated_text}"
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Text statistics
                st.markdown('<h3 class="subsection-header">Statistici text</h3>', unsafe_allow_html=True)
                
                source_words = text_input.split()
                target_words = translated_text.split()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    create_metric_box("Cuvinte EN", str(len(source_words)), "Text original")
                
                with col2:
                    create_metric_box("Cuvinte RO", str(len(target_words)), "Text tradus")
                
                with col3:
                    compression_ratio = len(target_words) / len(source_words) if source_words else 0
                    create_metric_box("Raport lungime", f"{compression_ratio:.2f}", "RO/EN ratio")
                
                with col4:
                    chars_ratio = len(translated_text) / len(text_input) if text_input else 0
                    create_metric_box("Raport caractere", f"{chars_ratio:.2f}", "Caractere RO/EN")
                
                # XAI Analysis for Translation
                st.markdown('<h2 class="section-header">Analiza XAI pentru Traducere</h2>', unsafe_allow_html=True)
                
                # SHAP Analysis
                st.markdown('<h3 class="subsection-header">Analiza SHAP</h3>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile SHAP pentru traducere..."):
                    try:
                        # Create SHAP explainer for translation
                        def translation_predict(texts):
                            """Prediction function for SHAP - returns translation probabilities"""
                            results = []
                            for text in texts:
                                if not text.strip():  # Handle empty text
                                    results.append(np.zeros(tokenizer.vocab_size))
                                    continue
                                
                                try:
                                    # Tokenize input
                                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                                    
                                    with torch.no_grad():
                                        # Get model outputs
                                        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, 
                                                                return_dict_in_generate=True, output_scores=True)
                                        
                                        # Get the probability of the first generated token
                                        if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                                            # Use softmax of first token scores as proxy for translation quality
                                            first_token_probs = torch.nn.functional.softmax(outputs.scores[0][0], dim=-1)
                                            results.append(first_token_probs.numpy())
                                        else:
                                            # Fallback: uniform distribution
                                            results.append(np.ones(tokenizer.vocab_size) / tokenizer.vocab_size)
                                
                                except Exception as e:
                                    # Fallback for any errors
                                    results.append(np.zeros(tokenizer.vocab_size))
                            
                            return np.array(results)
                        
                        # Create SHAP explainer
                        explainer = shap.Explainer(translation_predict, tokenizer)
                        shap_values = explainer([text_input])
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown('<h4>Contributiile SHAP pentru traducere</h4>', unsafe_allow_html=True)
                            
                            # Get the tokens and their SHAP values
                            tokens = shap_values[0].data
                            # Sum across all vocab dimensions to get overall contribution
                            token_contributions = np.sum(shap_values[0].values, axis=1)
                            
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
                            st.markdown('<h3 class="subsection-header">Vizualizare SHAP</h3>', unsafe_allow_html=True)
                            
                            # Get top positive and negative contributors
                            top_positive = shap_df[shap_df['Contributie_SHAP'] > 0].head(5)
                            top_negative = shap_df[shap_df['Contributie_SHAP'] < 0].head(5)
                            
                            combined = pd.concat([top_positive, top_negative]).sort_values('Contributie_SHAP')
                            
                            if len(combined) > 0:
                                # Use Plotly instead of matplotlib to avoid pixel limits
                                import plotly.graph_objects as go
                                
                                colors = ['red' if x < 0 else 'green' for x in combined['Contributie_SHAP']]
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    y=combined['Token'],
                                    x=combined['Contributie_SHAP'],
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f'{val:.3f}' for val in combined['Contributie_SHAP']],
                                    textposition='outside',
                                    name='Contributii SHAP'
                                ))
                                
                                fig.update_layout(
                                    title='Top contributori pentru traducere',
                                    xaxis_title='Valoare SHAP',
                                    yaxis_title='Token',
                                    height=400,
                                    showlegend=False,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                
                                # Add vertical line at x=0
                                fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Eroare la calcularea SHAP pentru traducere: {str(e)}")
                        st.info("SHAP pentru traducere poate fi computational intensiv. Incearca cu un text mai scurt.")
                
                # LIME Analysis for Translation
                st.markdown('<h3 class="subsection-header">Analiza LIME</h3>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile LIME pentru traducere..."):
                    try:
                        # LIME prediction function for translation
                        def lime_translation_predict(texts):
                            """Prediction function for LIME - returns translation quality scores"""
                            results = []
                            
                            for text in texts:
                                if not text.strip():
                                    results.append([0.0, 1.0])  # [poor_translation, good_translation]
                                    continue
                                
                                try:
                                    # Translate text
                                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                                    
                                    with torch.no_grad():
                                        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
                                        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                                        
                                        # Simple heuristic for translation quality
                                        # Based on length ratio and token diversity
                                        input_length = len(text.split())
                                        output_length = len(translation.split())
                                        
                                        # Quality score based on reasonable length ratio
                                        if input_length > 0:
                                            length_ratio = output_length / input_length
                                            quality_score = min(1.0, max(0.0, 1.0 - abs(length_ratio - 0.8)))  # Expect ~80% length ratio
                                        else:
                                            quality_score = 0.0
                                        
                                        results.append([1.0 - quality_score, quality_score])
                                
                                except Exception:
                                    results.append([1.0, 0.0])  # Poor translation for errors
                            
                            return np.array(results)
                        
                        # Create LIME explainer
                        lime_explainer = LimeTextExplainer(
                            class_names=['Traducere slaba', 'Traducere buna']
                        )
                        
                        lime_exp = lime_explainer.explain_instance(
                            text_input,
                            lime_translation_predict,
                            num_features=min(10, len(text_input.split())),
                            num_samples=100
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown('<h4>Contributiile LIME pentru traducere</h4>', unsafe_allow_html=True)
                            
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
                                # LIME bar plot - reduced size to avoid pixel limit
                                fig, ax = plt.subplots(figsize=(6, 4))
                                
                                words, importance = zip(*lime_features)
                                colors = ['red' if imp < 0 else 'green' for imp in importance]
                                
                                bars = ax.barh(range(len(words)), importance, color=colors, alpha=0.7)
                                ax.set_yticks(range(len(words)))
                                ax.set_yticklabels(words)
                                ax.set_xlabel('Contributie LIME')
                                ax.set_title('Contributii LIME pentru calitatea traducerii')
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
                        st.error(f"Eroare la calcularea LIME pentru traducere: {str(e)}")
                        st.info("LIME pentru traducere necesita texte cu multiple cuvinte. Incearca cu o propozitie mai complexa.")
    
    else:
        st.warning("Te rog sa introduci un text valid in engleza pentru traducere.")

# Educational section
st.markdown('<h2 class="section-header">Importanta XAI in traducerea automata</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
    <h3>Beneficiile analizei XAI</h3>
    <h4>Pentru dezvoltatori:</h4>
    <ul>
    <li><strong>Debugging:</strong> Identificarea unde modelul face alegeri proaste</li>
    <li><strong>Optimizare:</strong> Ajustarea parametrilor pentru rezultate mai bune</li>
    <li><strong>Validare:</strong> Verificarea ca modelul "gandeste" logic</li>
    <li><strong>Bias Detection:</strong> Identificarea prejudecatilor in traducere</li>
    </ul>
    
    <h4>Pentru utilizatori:</h4>
    <ul>
    <li><strong>Transparenta:</strong> Intelegerea procesului de traducere</li>
    <li><strong>Incredere:</strong> Validarea calitatii output-ului</li>
    <li><strong>Control:</strong> Ajustarea parametrilor pentru stiluri diferite</li>
    <li><strong>Educatie:</strong> Invatarea despre AI de traducere</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h3>Limitari si precautii</h3>
    <h4>Limitari:</h4>
    <ul>
    <li><strong>Complexitate:</strong> Multe capete de atentie sunt greu de interpretat</li>
    <li><strong>Indirecta:</strong> Atentia nu inseamna intotdeauna cauzalitate</li>
    <li><strong>Variabilitate:</strong> Diferite layers pot avea comportamente diferite</li>
    <li><strong>Context:</strong> Atentia poate fi influentata de factori neevident</li>
    </ul>
    
    <h4>Bune practici:</h4>
    <ul>
    <li>Analizeaza mai multe layers si capete</li>
    <li>Compara cu rezultate pe texte similare</li>
    <li>Valideaza interpretarile cu experti</li>
    <li>Foloseste atentia ca ghid, nu ca dovada</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Ai explorat mecanismele de atentie in traducerea automata!</strong></p>
<p>Continua cu urmatoarea pagina pentru a vedea cum functioneaza generarea de text</p>
</div>
""", unsafe_allow_html=True)
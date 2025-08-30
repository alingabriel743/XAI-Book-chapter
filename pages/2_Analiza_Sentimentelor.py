import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_text import LimeTextExplainer
import torch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styles import apply_custom_css, create_info_box, create_metric_box
from utils.models import load_sentiment_model, predict_sentiment

# Page config
st.set_page_config(
    page_title="Analiza Sentimentelor - XAI in NLP",
    page_icon="📊",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Page header
st.markdown('<h1 class="page-title">Analiza sentimentelor cu XAI</h1>', unsafe_allow_html=True)

# Introduction to sentiment analysis
st.markdown("""
<div class="info-box">
<h2 style="color: #1f77b4; margin-bottom: 1rem;">Ce este analiza sentimentelor?</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
<strong>Analiza sentimentelor</strong> este procesul de identificare si clasificare a opiniilor exprimate intr-un text 
pentru a determina daca atitudinea autorului fata de un subiect este pozitiva, negativa sau neutra.
</p>
<p style="font-size: 1.1rem; line-height: 1.6;">
Cu ajutorul XAI, putem intelege nu doar <em>care</em> este sentimentul, ci si <em>de ce</em> modelul 
a luat aceasta decizie, identificand cuvintele si frazele cheie care influenteaza clasificarea.
</p>
</div>
""", unsafe_allow_html=True)

# Model information
st.markdown('<h2 class="section-header">Modelul folosit</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    create_metric_box(
        "Model",
        "Roberta",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

with col2:
    create_metric_box(
        "Clase",
        "3 categorii",
        "Negativ, Neutru, Pozitiv"
    )

with col3:
    create_metric_box(
        "Domeniu",
        "Social Media",
        "Antrenat pe tweet-uri"
    )

# Load model
st.markdown('<h2 class="section-header">Analiza interactiva</h2>', unsafe_allow_html=True)

with st.spinner("Se incarca modelul de sentiment..."):
    tokenizer, model = load_sentiment_model()

if tokenizer is None or model is None:
    st.error("Nu s-a putut incarca modelul. Verifica conexiunea la internet si dependentele.")
    st.stop()

# Text input
st.markdown('<h3 class="subsection-header">Introdu textul pentru analiza:</h3>', unsafe_allow_html=True)

# Example texts
examples = {
    "Pozitiv - Produs": "I absolutely love this new smartphone! The camera quality is amazing and the battery lasts all day. Highly recommended!",
    "Negativ - Restaurant": "Terrible service and the food was cold. The waiters were rude and we waited 45 minutes for our order. Never going back!",
    "Neutru - Informativ": "The weather forecast shows rain tomorrow with temperatures around 15 degrees Celsius. People should bring umbrellas.",
    "Mixt - Recenzie": "The movie had great visual effects and excellent acting, but the plot was confusing and the ending was disappointing.",
    "Sarcastic": "Oh wonderful, another meeting that could have been an email. I just love wasting my time like this!"
}

selected_example = st.selectbox(
    "Selecteaza un exemplu sau scrie propriul text:",
    ["Text personalizat"] + list(examples.keys())
)

if selected_example == "Text personalizat":
    default_text = "Write your own text here..."
else:
    default_text = examples[selected_example]

text_input = st.text_area(
    "Text pentru analiza:",
    value=default_text,
    height=100,
    help="Introdu un text in limba engleza pentru a analiza sentimentul"
)

# Analysis parameters
with st.expander("Parametri avansati", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        max_tokens = st.slider("Lungimea maxima (tokens):", 128, 512, 256)
        shap_samples = st.slider("Esantioane SHAP:", 50, 500, 100)
    
    with col2:
        lime_samples = st.slider("Esantioane LIME:", 100, 1000, 200)
        top_features = st.slider("Top features afisate:", 5, 20, 10)

# Analysis button
if st.button("Analizeaza sentimentul", type="primary"):
    if text_input.strip() and text_input != "Write your own text here...":
        with st.spinner("Se analizeaza textul..."):
            
            # Make prediction
            predictions, predicted_class = predict_sentiment(text_input, tokenizer, model)
            
            if predictions is not None:
                labels = ["Negativ", "Neutru", "Pozitiv"]
                label_colors = ["#ff4444", "#ffaa44", "#44ff44"]
                
                # Results section
                st.markdown('<h2 class="section-header">Rezultate predictie</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Main prediction
                    confidence = predictions[0][predicted_class].item()
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3 style="text-align: center; margin-bottom: 1rem;">Predictia modelului</h3>
                    <h2 style="text-align: center; color: {label_colors[predicted_class]}; margin: 1rem 0;">
                    {labels[predicted_class]}
                    </h2>
                    <p style="text-align: center; font-size: 1.2rem;">
                    <strong>Confidenta: {confidence:.1%}</strong>
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    pred_df = pd.DataFrame({
                        'Sentiment': labels,
                        'Probabilitate': predictions[0].numpy(),
                        'Culoare': label_colors
                    })
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.bar(pred_df['Sentiment'], pred_df['Probabilitate'], 
                                 color=pred_df['Culoare'], alpha=0.8)
                    ax.set_ylabel('Probabilitate')
                    ax.set_title('Distributia probabilitatilor')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, pred_df['Probabilitate']):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Text analysis
                    st.markdown('<h3 class="subsection-header">Analiza textului</h3>', unsafe_allow_html=True)
                    
                    # Text statistics
                    words = text_input.split()
                    sentences = text_input.split('.')
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        create_metric_box("Cuvinte", str(len(words)), "Total cuvinte")
                    
                    with metrics_col2:
                        create_metric_box("Propozitii", str(len([s for s in sentences if s.strip()])), "Total propozitii")
                    
                    with metrics_col3:
                        create_metric_box("Caractere", str(len(text_input)), "Total caractere")
                    
                    # Show original text with highlighting
                    st.markdown("**Text analizat:**")
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 8px; 
                               border-left: 4px solid {label_colors[predicted_class]};">
                    <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">
                    "{text_input}"
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # SHAP Analysis
                st.markdown('<h2 class="section-header">Explicatii SHAP</h2>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile SHAP..."):
                    try:
                        # SHAP explanation function
                        def predict_proba(texts):
                            results = []
                            for text in texts:
                                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
                                with torch.no_grad():
                                    outputs = model(**inputs)
                                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                results.append(probs[0].numpy())
                            return np.array(results)
                        
                        # Create SHAP explainer and get explanations
                        explainer = shap.Explainer(predict_proba, tokenizer)
                        shap_values = explainer([text_input])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<h3 class="subsection-header">Importanta cuvintelor (SHAP)</h3>', unsafe_allow_html=True)
                            
                            # Get tokens and their SHAP values for the predicted class
                            tokens = shap_values[0].data
                            values = shap_values[0].values[:, predicted_class]
                            
                            # Create dataframe for better visualization
                            token_importance = pd.DataFrame({
                                'Token': tokens,
                                'Importanta_SHAP': values,
                                'Abs_Importanta': np.abs(values)
                            }).sort_values('Abs_Importanta', ascending=False)
                            
                            # Display top influential tokens
                            top_tokens = token_importance.head(top_features)
                            
                            # Color code the importance
                            def color_importance(val):
                                if val > 0:
                                    return f"background-color: rgba(76, 175, 80, {min(abs(val) * 5, 1)})"
                                else:
                                    return f"background-color: rgba(244, 67, 54, {min(abs(val) * 5, 1)})"
                            
                            styled_df = top_tokens[['Token', 'Importanta_SHAP']].style.applymap(
                                color_importance, subset=['Importanta_SHAP']
                            ).format({'Importanta_SHAP': '{:.4f}'})
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Legend
                            st.markdown("""
                            <div style="font-size: 0.9rem; margin-top: 1rem;">
                            <span style="background-color: rgba(76, 175, 80, 0.3); padding: 0.2rem 0.5rem; border-radius: 3px;">Verde: Contribuie la predictie</span>
                            <span style="background-color: rgba(244, 67, 54, 0.3); padding: 0.2rem 0.5rem; border-radius: 3px; margin-left: 1rem;">Rosu: Impotriva predictiei</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<h3 class="subsection-header">Vizualizare SHAP</h3>', unsafe_allow_html=True)
                            
                            # SHAP bar plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Get top positive and negative contributors
                            top_positive = token_importance[token_importance['Importanta_SHAP'] > 0].head(5)
                            top_negative = token_importance[token_importance['Importanta_SHAP'] < 0].head(5)
                            
                            combined = pd.concat([top_positive, top_negative]).sort_values('Importanta_SHAP')
                            
                            colors = ['red' if x < 0 else 'green' for x in combined['Importanta_SHAP']]
                            
                            bars = ax.barh(range(len(combined)), combined['Importanta_SHAP'], color=colors, alpha=0.7)
                            ax.set_yticks(range(len(combined)))
                            ax.set_yticklabels(combined['Token'])
                            ax.set_xlabel('Valoare SHAP')
                            ax.set_title(f'Top contributori pentru "{labels[predicted_class]}"')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add value labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + (0.001 if width > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                                       f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    
                    except Exception as e:
                        st.error(f"Eroare la calcularea SHAP: {str(e)}")
                        st.info("Incearca cu un text mai scurt sau ajusteaza parametrii.")
                
                # LIME Analysis
                st.markdown('<h2 class="section-header">Explicatii LIME</h2>', unsafe_allow_html=True)
                
                with st.spinner("Se calculeaza valorile LIME..."):
                    try:
                        # LIME prediction function
                        def lime_predict_proba(texts):
                            results = []
                            for text in texts:
                                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
                                with torch.no_grad():
                                    outputs = model(**inputs)
                                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                results.append(probs[0].numpy())
                            return np.array(results)
                        
                        # Create LIME explainer
                        lime_explainer = LimeTextExplainer(class_names=labels)
                        lime_exp = lime_explainer.explain_instance(
                            text_input, 
                            lime_predict_proba, 
                            num_features=top_features,
                            num_samples=lime_samples
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<h3 class="subsection-header">Importanta cuvintelor (LIME)</h3>', unsafe_allow_html=True)
                            
                            # Get LIME features for predicted class
                            lime_features = lime_exp.as_list()
                            lime_df = pd.DataFrame(lime_features, columns=['Cuvant', 'Importanta_LIME'])
                            lime_df['Abs_Importanta'] = np.abs(lime_df['Importanta_LIME'])
                            lime_df = lime_df.sort_values('Abs_Importanta', ascending=False)
                            
                            # Style the dataframe
                            def color_lime_importance(val):
                                if val > 0:
                                    return f"background-color: rgba(76, 175, 80, {min(abs(val) * 3, 1)})"
                                else:
                                    return f"background-color: rgba(244, 67, 54, {min(abs(val) * 3, 1)})"
                            
                            styled_lime_df = lime_df[['Cuvant', 'Importanta_LIME']].style.applymap(
                                color_lime_importance, subset=['Importanta_LIME']
                            ).format({'Importanta_LIME': '{:.4f}'})
                            
                            st.dataframe(styled_lime_df, use_container_width=True)
                        
                        with col2:
                            st.markdown('<h3 class="subsection-header">Vizualizare LIME</h3>', unsafe_allow_html=True)
                            
                            # LIME bar plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            words, importance = zip(*lime_features)
                            colors = ['red' if imp < 0 else 'green' for imp in importance]
                            
                            bars = ax.barh(range(len(words)), importance, color=colors, alpha=0.7)
                            ax.set_yticks(range(len(words)))
                            ax.set_yticklabels(words)
                            ax.set_xlabel('Importanta LIME')
                            ax.set_title(f'Contributii LIME pentru "{labels[predicted_class]}"')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add value labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                                       f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        # LIME HTML visualization
                        st.markdown('<h3 class="subsection-header">Text highlighting (LIME)</h3>', unsafe_allow_html=True)
                        
                        # Create HTML visualization
                        html_exp = lime_exp.as_html()
                        
                        # Display in an expandable section
                        with st.expander("Vezi textul evidentiat", expanded=True):
                            # Extract and clean the HTML
                            import re
                            # Find the div with the explanation
                            html_match = re.search(r'<div[^>]*>.*?</div>', html_exp, re.DOTALL)
                            if html_match:
                                html_content = html_match.group()
                                st.components.v1.html(html_content, height=200)
                            else:
                                st.markdown("Nu s-a putut genera vizualizarea HTML.")
                    
                    except Exception as e:
                        st.error(f"Eroare la calcularea LIME: {str(e)}")
                        st.info("Incearca cu un text mai scurt sau ajusteaza parametrii.")
    
    else:
        st.warning("Te rog sa introduci un text valid pentru analiza.")

# Educational section
st.markdown('<h2 class="section-header">De ce conteaza fiecare metoda?</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
    <h3>SHAP in analiza sentimentelor</h3>
    <h4>Avantaje:</h4>
    <ul>
    <li><strong>Rigoare matematica:</strong> Valorile se insumeaza exact la diferenta fata de baseline</li>
    <li><strong>Consistenta:</strong> Acelasi cuvant va avea aceeasi importanta in contexte similare</li>
    <li><strong>Explicatii globale:</strong> Poate arata comportamentul modelului pe intreg dataset-ul</li>
    <li><strong>Atribuiri precise:</strong> Fiecare token primeste o valoare exacta</li>
    </ul>
    <h4>Limitari:</h4>
    <ul>
    <li>Calculul poate fi lent pentru texte lungi</li>
    <li>Necesita intelegerea conceptelor matematice</li>
    <li>Poate fi suprasensibil la preprocessing</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h3>LIME in analiza sentimentelor</h3>
    <h4>Avantaje:</h4>
    <ul>
    <li><strong>Intuitivitate:</strong> Usor de inteles pentru non-experti</li>
    <li><strong>Viteza:</strong> Calcul rapid chiar si pentru texte lungi</li>
    <li><strong>Vizualizari:</strong> Evidentiera textului in culori</li>
    <li><strong>Model-agnostic:</strong> Functioneaza cu orice model</li>
    </ul>
    <h4>Limitari:</h4>
    <ul>
    <li>Doar explicatii locale pentru instante individuale</li>
    <li>Rezultatele pot varia intre rulari</li>
    <li>Dependenta de strategia de sampling</li>
    <li>Aproximari care uneori pot fi inexacte</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Ai experimentat cu analiza sentimentelor!</strong></p>
<p>Continua sa explorezi celelalte tehnici XAI in urmatoarele pagini</p>
</div>
""", unsafe_allow_html=True)
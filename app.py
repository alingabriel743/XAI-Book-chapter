import streamlit as st
from utils.styles import apply_custom_css

# Page config
st.set_page_config(
    page_title="XAI in Procesarea Limbajului Natural",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_css()

# Main page content
st.markdown('<h1 class="page-title">XAI in Procesarea Limbajului Natural</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Demonstratii interactive cu SHAP si LIME pentru procesarea limbajului natural</p>', unsafe_allow_html=True)

# Welcome section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="info-box">
    <h2 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Bun venit la demonstratiile XAI!</h2>
    <p style="font-size: 1.1rem; line-height: 1.6;">
    Aceasta aplicatie interactiva demonstreaza aplicarea tehnicilor de explicabilitate in inteligenta artificiala (XAI) in procesarea limbajului natural. Vei explora cum functioneaza SHAP si LIME pentru a intelege deciziile modelelor de inteligenta artificiala.
    </p>
    </div>
    """, unsafe_allow_html=True)

# Navigation instructions
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
<h3 style="color: white; text-align: center; margin-bottom: 1rem;">Navigare prin aplicatie</h3>
<p style="color: white; text-align: center; font-size: 1.1rem;">
Foloseste meniul din <strong>sidebar-ul din stanga</strong> pentru a explora diferitele sectiuni ale aplicatiei. Fiecare pagina ofera demonstratii interactive si explicatii detaliate.
</p>
</div>
""", unsafe_allow_html=True)

# Pages overview
st.markdown('<h2 class="section-header">Sectiunile aplicatiei</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
    <h3>Introducere</h3>
    <p>Conceptele fundamentale ale XAI, SHAP si LIME. Perfect pentru incepatori care vor sa inteleaga bazele explicabilitatii in inteligenta artificiala.</p>
    <ul>
    <li>Ce este XAI si de ce este important</li>
    <li>Introducere in SHAP si LIME</li>
    <li>Aplicatii in procesarea limbajului natural</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Traducere automata</h3>
    <p>Exploreaza cum modelele de traducere automata gandesc si care cuvinte considera importante in procesul de traducere.</p>
    <ul>
    <li>Traducere engleza → romana</li>
    <li>Analiza matricilor de atentie</li>
    <li>Importanta token-urilor in traducere</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h3>Analiza sentimentelor</h3>
    <p>Demonstratii practice ale analizei sentimentelor cu explicatii SHAP si LIME pentru a intelege de ce un model clasifica textul intr-un anumit fel.</p>
    <ul>
    <li>Clasificare sentiment pozitiv/neutru/negativ</li>
    <li>Explicatii SHAP si LIME</li>
    <li>Vizualizari interactive</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h3>Generare text</h3>
    <p>Intelege cum modelele generative precum GPT-2 creaza text si analizeaza probabilitatile pentru fiecare cuvant generat.</p>
    <ul>
    <li>Generare text cu GPT-2</li>
    <li>Analiza probabilitatilor</li>
    <li>Parametri de control (temperatura, lungime)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Comparison section
st.markdown("""
<div class="error-box">
<h3>Comparatie SHAP vs LIME</h3>
<p>O analiza detaliata a diferentelor intre cele doua principale tehnici XAI, cu recomandari despre cand sa folosesti fiecare metoda.</p>
<ul>
<li>Avantajele si dezavantajele fiecarei metode</li>
<li>Cazuri de utilizare practice</li>
<li>Ghid de alegere a metodei potrivite</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Technical requirements
st.markdown('<h2 class="section-header">Cerinte tehnice</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #1f77b4;">Modele folosite</h4>
    <ul style="margin: 0;">
    <li><strong>Sentiment:</strong> RoBERTa</li>
    <li><strong>Traducere:</strong> MarianMT</li>
    <li><strong>Generare:</strong> GPT-2</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #ff7f0e;">Biblioteci XAI</h4>
    <ul style="margin: 0;">
    <li><strong>SHAP:</strong> v0.42.0+</li>
    <li><strong>LIME:</strong> v0.2.0+</li>
    <li><strong>Transformers:</strong> v4.30.0+</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #2ca02c;">Resurse sistem</h4>
    <ul style="margin: 0;">
    <li><strong>RAM:</strong> min. 4GB</li>
    <li><strong>Storage:</strong> ~2GB</li>
    <li><strong>Internet:</strong> pentru download modele</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Instructions
st.markdown('<h2 class="section-header">Cum sa incepi</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 2px solid #dee2e6;">
<h4 style="color: #495057; margin-bottom: 1rem;">Pasi pentru utilizare:</h4>
<ol style="font-size: 1.1rem; line-height: 1.8;">
<li><strong>Navigheaza</strong> prin meniul lateral din stanga pentru a selecta o sectiune</li>
<li><strong>Citeste</strong> explicatiile teoretice din fiecare pagina</li>
<li><strong>Experimenteaza</strong> cu exemplele interactive</li>
<li><strong>Analizeaza</strong> rezultatele si explicatiile oferite</li>
<li><strong>Compara</strong> diferitele metode XAI in sectiunea dedicata</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>XAI in Procesarea Limbajului Natural</strong></p>
<p>Aplicatie demonstrativa pentru invatarea conceptelor de explicabilitate in inteligenta artificiala</p>
<p><em>Selecteaza o pagina din meniul lateral pentru a incepe!</em> 👈</p>
</div>
""", unsafe_allow_html=True)
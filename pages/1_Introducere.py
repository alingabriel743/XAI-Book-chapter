import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styles import apply_custom_css, create_info_box

# Page config
st.set_page_config(
    page_title="Introducere - XAI in NLP",
    page_icon="📚",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Page header
st.markdown('<h1 class="page-title">Introducere in XAI pentru NLP</h1>', unsafe_allow_html=True)

# Introduction section
st.markdown("""
<div class="info-box">
<h2 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">Ce este XAI?</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
<strong>Explicabilitatea in inteligenta artificiala (XAI)</strong> este o disciplina care se concentreaza pe crearea de modele AI transparente si interpretabile. In contextul procesarii limbajului natural, XAI ne ajuta sa intelegem:
</p>
<ul style="font-size: 1.1rem; line-height: 1.6;">
<li><strong>Ce</strong> influenteaza deciziile modelului</li>
<li><strong>Cum</strong> sunt procesate informatiile din text</li>
<li><strong>De ce</strong> modelul face anumite predictii</li>
<li><strong>Cand</strong> sa avem incredere in rezultate</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Why XAI matters
st.markdown('<h2 class="section-header">De ce este importanta XAI?</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="success-box">
    <h3>Incredere si siguranta</h3>
    <p>XAI ne permite sa identificam bias-ul, sa detectam erorile si sa construim sisteme mai sigure si mai etice.</p>
    <ul>
    <li>Detectarea prejudecatilor</li>
    <li>Identificarea erorilor</li>
    <li>Conformitate regulamentara</li>
    <li>Transparenta decizionala</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h3>Debugging si imbunatatire</h3>
    <p>Explicabilitatea ne ajuta sa identificam unde se blocheaza modelul si cum sa-l imbunatatim.</p>
    <ul>
    <li>Diagnoticarea problemelor</li>
    <li>Optimizarea performantei</li>
    <li>Validarea comportamentului</li>
    <li>Ghidarea dezvoltarii</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
    <h3>Intelegere si invatare</h3>
    <p>XAI democratizeaza inteligenta artificiala, facandu-l accesibil si comprehensibil pentru non-experti.</p>
    <ul>
    <li>Educatia utilizatorilor</li>
    <li>Adoptarea tehnologiei</li>
    <li>Colaborarea om-masina</li>
    <li>Inovatia responsabila</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Main XAI techniques
st.markdown('<h2 class="section-header">Principalele tehnici XAI</h2>', unsafe_allow_html=True)

# SHAP section
st.markdown('<h3 class="subsection-header">SHAP (Shapley Additive exPlanations)</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="success-box">
    <h4>Fundamentul matematic</h4>
    <p>SHAP se bazeaza pe <strong>teoria jocurilor</strong> si conceptul de <strong>valori Shapley</strong>, oferind o metoda matematica pentru atribuirea importantei fiecarui feature.</p>
    
    <h4>Proprietati garantate</h4>
    <ul>
    <li><strong>Eficienta:</strong> Suma contributiilor = diferenta fata de baseline</li>
    <li><strong>Simetrie:</strong> Features cu contributii egale primesc scoruri egale</li>
    <li><strong>Dummy:</strong> Features irelevante primesc scorul 0</li>
    <li><strong>Aditivitate:</strong> Pentru modele composite, valorile se aduna</li>
    </ul>
    
    <h4>Tipuri de explicatii</h4>
    <ul>
    <li><strong>Locale:</strong> Pentru predictii individuale</li>
    <li><strong>Globale:</strong> Pentru intelegerea modelului in ansamblu</li>
    <li><strong>Cohorte:</strong> Pentru grupuri specifice de date</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #1f77b4;">Avantaje SHAP</h4>
    <ul>
    <li>Explicatii consistente</li>
    <li>Suport pentru orice model</li>
    <li>Vizualizari bogate</li>
    <li>Scaling la modele mari</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #dc3545;">Limitari SHAP</h4>
    <ul>
    <li>Computational costisitor</li>
    <li>Complexitate conceptuala</li>
    <li>Timp de calcul mare</li>
    <li>Interpretare dificila</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# LIME section
st.markdown('<h3 class="subsection-header">LIME (Local Interpretable Model-agnostic Explanations)</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="warning-box">
    <h4>Metodologia de perturbatie</h4>
    <p>LIME functioneaza prin <strong>perturbarea</strong> datelor de intrare si observarea cum se schimba predictiile. 
    Construieste un <strong>model surrogate simplu</strong> care aproximeaza comportamentul modelului complex local.</p>
    
    <h4>Procesul LIME</h4>
    <ol>
    <li><strong>Perturbatie:</strong> Genereaza variante ale textului original</li>
    <li><strong>Predictie:</strong> Ruleaza modelul pe fiecare varianta</li>
    <li><strong>Ponderare:</strong> Atribuie importanta bazata pe similaritate</li>
    <li><strong>Aproximare:</strong> Antreneaza un model linear simplu</li>
    <li><strong>Explicatie:</strong> Extrage coeficientii ca importanta</li>
    </ol>
    
    <h4>Caracteristici cheie</h4>
    <ul>
    <li><strong>Model-agnostic:</strong> Functioneaza cu orice algoritm</li>
    <li><strong>Local:</strong> Explica predictii individuale</li>
    <li><strong>Interpretabil:</strong> Foloseste modele simple si usor de inteles</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #28a745;">Avantaje LIME</h4>
    <ul>
    <li>Foarte rapid</li>
    <li>Usor de inteles</li>
    <li>Model-agnostic</li>
    <li>Vizualizari intuitive</li>
    <li>Implementare simpla</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #dc3545;">Limitari LIME</h4>
    <ul>
    <li>Doar explicatii locale</li>
    <li>Instabilitate rezultate</li>
    <li>Dependenta de sampling</li>
    <li>Aproximari uneori gresite</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# NLP Applications
st.markdown('<h2 class="section-header">Aplicatii XAI in NLP</h2>', unsafe_allow_html=True)

applications = [
    {
        "title": "Analiza sentimentelor",
        "description": "Intelegerea care cuvinte contribuie la clasificarea unui text ca pozitiv, negativ sau neutru.",
        "examples": ["Recenzii produse", "Feedback clienti", "Monitorizare media sociala", "Analiza opiniilor"],
        "color": "info-box"
    },
    {
        "title": "Traducere automata",
        "description": "Explorarea mecanismelor de atentie pentru a vedea care cuvinte din text influenteaza traducerea.",
        "examples": ["Traducere documente", "Localizare software", "Comunicare multilingva", "Asistenta diplomatica"],
        "color": "success-box"
    },
    {
        "title": "Generare de text",
        "description": "Analiza procesului de generare pentru a intelege cum modelele creaza text coerent.",
        "examples": ["Asistente virtuale", "Generare continut", "Scrierea automata", "Completare text"],
        "color": "warning-box"
    },
    {
        "title": "Detectarea fake news",
        "description": "Identificarea indicatorilor lingvistici care semnaleaza informatii false sau misleadere.",
        "examples": ["Verificare factuala", "Moderare continut", "Jurnalism asistat", "Media literacy"],
        "color": "error-box"
    }
]

cols = st.columns(2)
for i, app in enumerate(applications):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="{app['color']}">
        <h3>{app['title']}</h3>
        <p>{app['description']}</p>
        <h4>Exemple de utilizare:</h4>
        <ul>
        {''.join([f'<li>{example}</li>' for example in app['examples']])}
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Technical challenges
st.markdown('<h2 class="section-header">Provocari tehnice in XAI pentru NLP</h2>', unsafe_allow_html=True)

challenges = [
    {
        "title": "Natura discreta a textului",
        "description": "Spre deosebire de imaginile cu pixeli continui, textul consta din token-uri discrete, facand perturbarea mai dificila.",
        "solutions": ["Inlocuirea cuvintelor cu sinonime", "Mascarea token-urilor", "Parafrazarea contextuala"]
    },
    {
        "title": "Context si secventialitate",
        "description": "Cuvintele au sens doar in context, iar ordinea lor este cruciala pentru intelegere.",
        "solutions": ["Analiza dependentelor", "Masuri de similaritate contextuala", "Atentia secventiala"]
    },
    {
        "title": "Scalabilitate",
        "description": "Textele pot fi foarte lungi, iar vocabularul foarte mare, creand provocari computationale.",
        "solutions": ["Sampling inteligent", "Aproximari eficiente", "Paralelizare GPU"]
    },
    {
        "title": "Ambiguitate si polisemie",
        "description": "Aceleasi cuvinte pot avea intelegsuri diferite in contexte diferite.",
        "solutions": ["Word embeddings contextualizate", "Analiza multi-nivel", "Dezambiguizare semantica"]
    }
]

for i, challenge in enumerate(challenges):
    if i % 2 == 0:
        st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
        cols = st.columns(2)
    
    with cols[i % 2]:
        st.markdown(f"""
        <div class="{'warning-box' if i % 2 == 0 else 'info-box'}">
        <h3>{challenge['title']}</h3>
        <p>{challenge['description']}</p>
        <h4>Solutii existente:</h4>
        <ul>
        {''.join([f'<li>{solution}</li>' for solution in challenge['solutions']])}
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if i % 2 == 1 or i == len(challenges) - 1:
        st.markdown('</div>', unsafe_allow_html=True)

# Best practices
st.markdown('<h2 class="section-header">Bune practici pentru XAI in NLP</h2>', unsafe_allow_html=True)

best_practices = [
    "**Alege metoda potrivita pentru cazul tau:** SHAP pentru acuratete, LIME pentru viteza",
    "**Valideaza explicatiile:** Testeaza pe cazuri cunoscute si verifica consistenta",
    "**Combina multiple metode:** Foloseste atat SHAP cat si LIME pentru perspective diferite",
    "**Testeaza pe date diverse:** Asigura-te ca explicatiile functioneaza pe diferite tipuri de text",
    "**Adapteaza pentru audienta:** Simplificati explicatiile pentru utilizatorii finali",
    "**Monitorizeaza in timp:** Explicabilitatea poate sa se schimbe odata cu modelul",
    "**Documenteaza limitarile:** Fii transparent despre ce nu poate explica metoda",
    "**Considera implicatiile etice:** Asigura-te ca explicatiile nu perpetueaza bias-uri"
]

cols = st.columns(2)
for i, practice in enumerate(best_practices):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="success-box" style="margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 1rem;">{practice}</p>
        </div>
        """, unsafe_allow_html=True)

# Next steps
st.markdown('<h2 class="section-header">Urmatorii pasi</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white;">
<h3 style="color: white; text-align: center; margin-bottom: 1rem;">Exploreaza aplicatia!</h3>
<p style="font-size: 1.1rem; line-height: 1.6; text-align: center;">
Acum ca intelegi conceptele fundamentale, este timpul sa vezi XAI in actiune! 
Navigheaza prin celelalte pagini pentru a experimenta cu:
</p>
<div style="display: flex; justify-content: space-around; margin-top: 1.5rem; flex-wrap: wrap;">
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Sentiment</h4>
<p>Analizeaza sentimente si vezi care cuvinte conteaza</p>
</div>
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Traducere</h4>
<p>Exploreaza mecanismele de atentie</p>
</div>
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Generare</h4>
<p>Intelege cum se genereaza textul</p>
</div>
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Comparare</h4>
<p>Afla cand sa folosesti fiecare metoda</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p>Acum ca stii bazele, sa experimentam cu XAI in practica!</p>
<p><em>Selecteaza urmatoarea pagina din sidebar pentru a continua!</em></p>
</div>
""", unsafe_allow_html=True)
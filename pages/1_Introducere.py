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
<strong>Explicabilitatea in inteligenta artificiala (XAI)</strong> se concentreaza pe crearea de modele AI transparente si interpretabile. In contextul procesarii limbajului natural, XAI ne ajuta sa intelegem:
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
    <p>XAI ne permite sa identificam prejudecatile, sa detectam erorile si sa construim sisteme mai sigure si mai etice.</p>
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
    <h3>Depanare si imbunatatire</h3>
    <p>Explicabilitatea ne ajuta sa identificam unde se blocheaza modelul si cum sa-l imbunatatim.</p>
    <ul>
    <li>Diagnosticarea erorilor</li>
    <li>Optimizarea performantei</li>
    <li>Validarea comportamentului</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
    <h3>Intelegere si invatare</h3>
    <p>XAI democratizeaza inteligenta artificiala, facandu-l accesibil si usor de inteles pentru utilizatori fara experienta.</p>
    <ul>
    <li>Educarea utilizatorilor</li>
    <li>Adoptarea tehnologiei</li>
    <li>Colaborarea om-computer</li>
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
    <p>SHAP se bazeaza pe <strong>teoria jocurilor</strong> si conceptul de <strong>valori Shapley</strong>, oferind o metoda matematica pentru atribuirea importantei fiecarei variabile independente.</p>
    
    <h4>Proprietati</h4>
    <ul>
    <li><strong>Eficienta:</strong> Suma contributiilor = diferenta fata de nivelul de referinta</li>
    <li><strong>Simetrie:</strong> Predictori cu contributii egale primesc scoruri egale</li>
    <li><strong>Dummy:</strong> Variabilele irelevante primesc scorul 0</li>
    <li><strong>Aditivitate:</strong> Pentru modelele compuse, valorile se aduna</li>
    </ul>
    
    <h4>Tipuri de explicatii</h4>
    <ul>
    <li><strong>Locale:</strong> Pentru predictii individuale</li>
    <li><strong>Globale:</strong> Pentru intelegerea modelului in ansamblu</li>
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
    <li>Scalabilitate pentru modele mari</li>
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
    <h4>Metodologia de perturbare</h4>
    <p>LIME functioneaza prin <strong>modificarea</strong> datelor de intrare si observarea modului in care se schimba predictiile. 
    Creaza un <strong>model simplificat</strong> care imita comportamentul modelului complex in jurul unui exemplu concret.</p>
    
    <h4>Procesul LIME</h4>
    <ol>
    <li><strong>Perturbare:</strong> Genereaza variante ale textului initial</li>
    <li><strong>Predictie:</strong> Ruleaza modelul pe fiecare varianta</li>
    <li><strong>Ponderare:</strong> Atribuie importanta pe baza similaritatii</li>
    <li><strong>Aproximare:</strong> Antreneaza un model liniar simplu</li>
    <li><strong>Explicatie:</strong> Extrage coeficientii ca masura a importantei</li>
    </ol>
    
    <h4>Caracteristici principale</h4>
    <ul>
    <li><strong>Independenta de model:</strong> Functioneaza cu orice algoritm</li>
    <li><strong>Local:</strong> Explica predictii individuale</li>
    <li><strong>Usor de interpretat:</strong> Foloseste modele simple si intuitive</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #28a745;">Avantajele LIME</h4>
    <ul>
    <li>Rapid</li>
    <li>Usor de inteles</li>
    <li>Functioneaza cu orice model</li>
    <li>Ofera vizualizari intuitive</li>
    <li>Implementare simpla</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-container">
    <h4 style="color: #dc3545;">Limitarile LIME</h4>
    <ul>
    <li>Ofera doar explicatii locale</li>
    <li>Rezultatele pot fi instabile</li>
    <li>Depinde mult de modul de esantionare</li>
    <li>Aproximarile pot fi uneori eronate</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# NLP Applications
st.markdown('<h2 class="section-header">Aplicatii XAI in NLP</h2>', unsafe_allow_html=True)

applications = [
    {
        "title": "Analiza sentimentelor",
        "description": "Arata ce cuvinte influenteaza clasificarea unui text ca pozitiv, negativ sau neutru.",
        "examples": ["Recenzii de produse", "Feedback de la clienti", "Monitorizarea retelelor sociale", "Analiza opiniilor"],
        "color": "info-box"
    },
    {
        "title": "Traducere automata",
        "description": "Permite explorarea mecanismelor de atentie pentru a intelege ce cuvinte influenteaza traducerea.",
        "examples": ["Traducerea documentelor", "Localizarea software-ului", "Comunicare multilingva", "Asistenta diplomatica"],
        "color": "success-box"
    },
    {
        "title": "Generare de text",
        "description": "Ajuta la intelegerea modului in care modelele genereaza text coerent.",
        "examples": ["Asistenti virtuali", "Crearea de continut", "Scriere automata", "Completarea textului"],
        "color": "warning-box"
    },
    {
        "title": "Detectarea stirilor false",
        "description": "Identifica tiparele lingvistice care tradeaza informatii false sau inselatoare.",
        "examples": ["Verificarea faptelor", "Moderarea continutului", "Jurnalism asistat", "Educatie media"],
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
        "description": "Spre deosebire de imagini (cu valori continue), textul este alcatuit din token-uri discrete, ceea ce face perturbarea mai dificila.",
        "solutions": ["Inlocuirea cuvintelor cu sinonime", "Mascarea token-urilor", "Parafrazare contextuala"]
    },
    {
        "title": "Context si secventialitate",
        "description": "Cuvintele au sens doar in context, iar ordinea lor este esentiala pentru intelegere.",
        "solutions": ["Analiza dependentelor", "Masuri de similaritate contextuala", "Atentie secventiala"]
    },
    {
        "title": "Scalabilitate",
        "description": "Textele pot fi foarte lungi, iar vocabularul foarte mare, ceea ce duce la provocari de calcul.",
        "solutions": ["Esantionare inteligenta", "Aproximari eficiente", "Paralelizare pe GPU"]
    },
    {
        "title": "Ambiguitate si polisemie",
        "description": "Aceleasi cuvinte pot avea sensuri diferite in contexte diferite.",
        "solutions": ["Reprezentari contextuale ale cuvintelor", "Analiza pe mai multe niveluri", "Dezambiguizare semantica"]
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
        <h4>Solutii posibile:</h4>
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
    "**Verifica explicatiile:** Testeaza pe exemple cunoscute si urmareste consistenta",
    "**Combina metode diferite:** Foloseste atat SHAP, cat si LIME pentru perspective complementare",
    "**Testeaza pe date variate:** Asigura-te ca explicatiile functioneaza pe texte diferite",
    "**Adapteaza explicatiile la public:** Simplifica pentru utilizatorii non-tehnici",
    "**Monitorizeaza in timp:** Explicabilitatea se poate schimba odata cu modelul",
    "**Noteaza limitarile:** Fii transparent in privinta lucrurilor pe care metoda nu le poate explica",
    "**Ia in calcul aspectele etice:** Asigura-te ca explicatiile nu perpetueaza prejudecati"
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
Acum ca ai inteles conceptele de baza, e momentul sa vezi XAI in actiune! 
Navigheaza prin celelalte pagini pentru a experimenta cu:
</p>
<div style="display: flex; justify-content: space-around; margin-top: 1.5rem; flex-wrap: wrap;">
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Analiza sentimentelor</h4>
<p>Descopera ce cuvinte fac diferenta</p>
</div>
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Traducere</h4>
<p>Exploreaza mecanismele de atentie</p>
</div>
<div style="text-align: center; margin: 0.5rem;">
<h4 style="color: white;">Generare</h4>
<p>Intelege cum este creat textul</p>
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
<p>Acum ca stapanesti bazele, hai sa experimentam cu XAI in practica!</p>
<p><em>Selecteaza urmatoarea pagina din bara laterala pentru a continua!</em></p>
</div>
""", unsafe_allow_html=True)
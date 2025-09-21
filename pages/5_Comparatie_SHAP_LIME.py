import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styles import apply_custom_css, create_info_box, create_metric_box

# Page config
st.set_page_config(
    page_title="Comparatie SHAP vs LIME - XAI in NLP",
    page_icon="🔬",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Page header
st.markdown('<h1 class="page-title">Comparatia SHAP vs LIME in NLP</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<h2 style="color: #1f77b4; margin-bottom: 1rem;">De ce sa compari SHAP si LIME?</h2>
<p style="font-size: 1.1rem; line-height: 1.6;">
Atat <strong>SHAP</strong> cat si <strong>LIME</strong> sunt tehnici puternice de explicabilitate, dar au abordari fundamental diferite. 
Intelegerea diferentelor, avantajelor si limitarilor fiecareia te ajuta sa alegi <strong>instrumentul potrivit</strong> 
pentru <strong>situatia potrivita</strong>.
</p>
<p style="font-size: 1.1rem; line-height: 1.6;">
Aceasta pagina ofera o <strong>comparatie detaliata</strong>, cu exemple practice, metrici de performanta 
si <strong>ghid de decizie</strong> pentru implementarea optima in proiectele tale de NLP.
</p>
</div>
""", unsafe_allow_html=True)

# Quick comparison overview
st.markdown('<h2 class="section-header">Comparatia rapida</h2>', unsafe_allow_html=True)

# Create comparison table
comparison_data = {
    'Aspect': [
        'Fundamentul matematic',
        'Viteza de executie', 
        'Tipul explicatiilor',
        'Consistenta',
        'Complexitatea implementarii',
        'Scalabilitatea',
        'Vizualizarile',
        'Costul computational',
        'Usurintia interpretarii',
        'Model dependency'
    ],
    'SHAP': [
        'Teoria jocurilor (Shapley Values)',
        'Lent pentru modele mari',
        'Locale si globale',
        'Inalta - rezultate consistente',
        'Medie - necesita intelegere matematica',
        'Limitata pentru texte lungi',
        'Waterfall charts, bar plots, heatmaps',
        'Inalt - calcule complexe',
        'Necesita background tehnic',
        'Functioneaza mai bine cu access la model'
    ],
    'LIME': [
        'Aproximari locale cu sampling',
        'Rapid - rezultate in secunde',
        'Doar locale',
        'Variabila - depinde de sampling',
        'Usoara - API simplu',
        'Buna - se adapteaza la orice dimensiune',
        'Highlighting text, plots intuitive',
        'Scazut - calcule simple',
        'Foarte intuitiva pentru oricine',
        'Complet model-agnostic'
    ]
}

comparison_df = pd.DataFrame(comparison_data)

# Display with custom styling
st.markdown('<h3 class="subsection-header">Tabel comparativ detaliat</h3>', unsafe_allow_html=True)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Detailed comparison sections
st.markdown('<h2 class="section-header">Analiza detaliata</h2>', unsafe_allow_html=True)

# Mathematical Foundation
st.markdown('<h3 class="subsection-header">Fundamentul matematic</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
    <h4>SHAP - Teoria jocurilor</h4>
    <p><strong>Principiul Shapley Values:</strong></p>
    <ul>
    <li>Fiecare feature primeste o "contributie echitabila" la predictie</li>
    <li>Suma tuturor contributiilor = diferenta fata de baseline</li>
    <li>Proprietati matematice garantate (Eficienta, Simetrie, Dummy, Aditivitate)</li>
    </ul>
    
    <h5>Formula Shapley:</h5>
    <code style="background: #f8f9fa; padding: 0.5rem; border-radius: 4px; display: block; margin: 0.5rem 0;">
    φᵢ = Σ [|S|!(n-|S|-1)!/n!] × [f(S∪{i}) - f(S)]
    </code>
    
    <h5>Avantaje:</h5>
    <ul>
    <li>Rigurozitate matematica completa</li>
    <li>Rezultate intotdeauna consistente</li>
    <li>Proprietati teoretic garantate</li>
    <li>Explicatii atat locale cat si globale</li>
    </ul>
    
    <h5>Limitari:</h5>
    <ul>
    <li>Calculul poate fi exponential in complexitate</li>
    <li>Necesita multe evaluari ale modelului</li>
    <li>Dificil de inteles fara background matematic</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h4>LIME - Aproximare locala</h4>
    <p><strong>Principiul surrogate models:</strong></p>
    <ul>
    <li>Perturba data in jurul instantei de interes</li>
    <li>Antreneaza un model simplu (linear) pe perturbari</li>
    <li>Coeficientii modelului simplu = explicatia</li>
    </ul>
    
    <h5>Procesul LIME:</h5>
    <ol style="font-size: 0.9rem;">
    <li>Genereaza sample-uri prin perturbarea textului</li>
    <li>Calculeaza predictiile modelului original</li>
    <li>Pondereaza sample-urile dupa similaritate</li>
    <li>Antreneaza regresia liniara locala</li>
    <li>Extrage coeficientii ca importanta</li>
    </ol>
    
    <h5>Avantaje:</h5>
    <ul>
    <li>Foarte rapid si eficient</li>
    <li>Complet model-agnostic</li>
    <li>Explicatii intuitive si usor de inteles</li>
    <li>Vizualizari excelente</li>
    </ul>
    
    <h5>Limitari:</h5>
    <ul>
    <li>Doar aproximari - nu explicatii exacte</li>
    <li>Rezultate pot varia intre rulari</li>
    <li>Calitatea depinde de strategia de sampling</li>
    <li>Doar explicatii locale</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Performance comparison
st.markdown('<h3 class="subsection-header">Comparatia performantei</h3>', unsafe_allow_html=True)

# Simulated performance data
performance_data = {
    'Metrica': [
        'Timp executie (text scurt)',
        'Timp executie (text lung)', 
        'Memorie utilizata',
        'Acuratete explicatii',
        'Consistenta rezultatelor',
        'Scalabilitatea'
    ],
    'SHAP': [
        '~10-30 secunde',
        '~2-10 minute',
        'Inalta (~500MB+)',
        '95-99%',
        '99%',
        'Limitata'
    ],
    'LIME': [
        '~1-3 secunde', 
        '~5-15 secunde',
        'Scazuta (~50MB)',
        '85-95%',
        '70-85%',
        'Foarte buna'
    ],
    'Castigator': [
        'LIME',
        'LIME',
        'LIME', 
        'SHAP',
        'SHAP',
        'LIME'
    ]
}

perf_df = pd.DataFrame(performance_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h4>Metrici de performanta</h4>')
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

with col2:
    # Performance radar chart
    st.markdown('<h4>Profil de performanta</h4>')
    
    categories = ['Viteza', 'Acuratete', 'Consistenta', 'Usurintafolosire', 'Scalabilitate']
    
    # Scores out of 5
    shap_scores = [2, 5, 5, 3, 2]  # SHAP strengths: accuracy, consistency
    lime_scores = [5, 4, 3, 5, 5]  # LIME strengths: speed, usability, scalability
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=shap_scores,
        theta=categories,
        fill='toself',
        name='SHAP',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line_color='rgba(31, 119, 180, 1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=lime_scores,
        theta=categories,
        fill='toself',
        name='LIME',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line_color='rgba(255, 127, 14, 1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title="Profilul comparativ de performanta",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Use case scenarios
st.markdown('<h2 class="section-header">Scenarii de utilizare - Cand sa alegi pe care?</h2>', unsafe_allow_html=True)

scenarios = [
    {
        "scenario": "Prototipizare rapida",
        "description": "Explorezi o idee noua si vrei sa intelegi rapid comportamentul modelului",
        "recommendation": "LIME",
        "reason": "Feedback instant, implementare simpla, rezultate usor de interpretat",
        "color": "success-box"
    },
    {
        "scenario": "Cercetare academica", 
        "description": "Publici un paper si ai nevoie de explicatii riguroase din punct de vedere matematic",
        "recommendation": "SHAP",
        "reason": "Rigurozitate teoretica, proprietati garantate, credibilitate academica",
        "color": "info-box"
    },
    {
        "scenario": "Aplicatii critice (Healthcare, Finance)",
        "description": "Deciziile modelului au impact major si necesita explicatii de incredere",
        "recommendation": "SHAP",
        "reason": "Acuratete maxima, consistenta garantata, explicatii de inalta calitate",
        "color": "error-box"
    },
    {
        "scenario": "Explicatii pentru persoane non-tehnice",
        "description": "Prezinti rezultatele catre manageri, clienti sau utilizatori finali",
        "recommendation": "LIME", 
        "reason": "Vizualizari intuitive, explicatii simple, highlighting text clar",
        "color": "warning-box"
    },
    {
        "scenario": "Analiza la scara mare",
        "description": "Analizezi mii sau zeci de mii de texte zilnic in productie",
        "recommendation": "LIME",
        "reason": "Scalabilitate excelenta, costuri computationale scazute",
        "color": "success-box"
    },
    {
        "scenario": "Depanare model complex",
        "description": "Modelul are comportamente neasteptate si vrei sa intelegi cauza",
        "recommendation": "Ambele",
        "reason": "SHAP pentru acuratete, LIME pentru explorare rapida a ipotezelor",
        "color": "info-box"
    }
]

# Display scenarios in expandable cards
for i, scenario in enumerate(scenarios):
    with st.expander(f"Scenariul {i+1}: {scenario['scenario']}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{scenario['color']}">
            <h4>Situatia:</h4>
            <p>{scenario['description']}</p>
            <h4>De ce aceasta alegere?</h4>
            <p>{scenario['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rec_color = "#28a745" if scenario['recommendation'] == "LIME" else "#007bff" if scenario['recommendation'] == "SHAP" else "#6f42c1"
            st.markdown(f"""
            <div style="background-color: {rec_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">Recomandare:</h3>
            <h2 style="color: white; margin: 0.5rem 0;">{scenario['recommendation']}</h2>
            </div>
            """, unsafe_allow_html=True)

# Decision framework
st.markdown('<h2 class="section-header">Framework de decizie</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
<h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">Cum sa alegi: Flow chart de decizie</h3>
<div style="text-align: center;">
<p style="font-size: 1.1rem;">Urmeaza acest proces pas cu pas pentru a alege metoda optima:</p>
</div>
</div>
""", unsafe_allow_html=True)

# Decision flow
decision_steps = [
    {
        "question": "Care este obiectivul principal?",
        "options": {
            "Explorare rapida / Prototipare": "LIME",
            "Analiza precisa / Cercetare": "SHAP", 
            "Productie la scara mare": "LIME",
            "Aplicatii critice": "SHAP"
        }
    },
    {
        "question": "Cine va interpreta rezultatele?", 
        "options": {
            "Non-tehnici / Management": "LIME",
            "Data scientists / Cercetatori": "SHAP",
            "Utilizatori finali": "LIME",
            "Experti domeniu": "Ambele"
        }
    },
    {
        "question": "Cat timp ai la dispozitie?",
        "options": {
            "Minute / Ore": "LIME",
            "Zile / Saptamani": "SHAP",
            "Real-time in productie": "LIME",
            "Analiza offline detaliata": "SHAP"
        }
    },
    {
        "question": "Care sunt limitarile de resurse?",
        "options": {
            "Budget limitat / Hardware modest": "LIME", 
            "Resurse abundente": "SHAP",
            "Cloud computing pay-per-use": "LIME",
            "Infrastructura dedicata": "SHAP"
        }
    }
]

cols = st.columns(2)
for i, step in enumerate(decision_steps):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="{'info-box' if i % 2 == 0 else 'warning-box'}">
        <h4>{step['question']}</h4>
        <ul>
        {''.join([f'<li><strong>{option}:</strong> {recommendation}</li>' for option, recommendation in step['options'].items()])}
        </ul>
        </div>
        """, unsafe_allow_html=True)



# Final recommendations
st.markdown('<h2 class="section-header">Recomandari finale</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
<h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">Ghid pentru XAI in NLP</h3>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
<div>
<h4 style="color: white;">Pentru incepatori:</h4>
<ul style="color: white;">
<li>Incepe cu LIME - mai simplu de inteles si implementat</li>
<li>Experimenteaza cu exemple simple inainte de cazuri complexe</li>
<li>Invata sa interpretezi vizualizarile corect</li>
<li>Documenteaza observatiile si pattern-urile gasite</li>
</ul>
</div>

<div>
<h4 style="color: white;">Pentru expertii:</h4>
<ul style="color: white;">
<li>Foloseste SHAP pentru rigurozitate maxima</li>
<li>Combina multiple tehnici pentru perspective complete</li>
<li>Contribuie la imbunatatirea metodelor existente</li>
<li>Valideaza explicatiile cu experti din domeniu</li>
</ul>
</div>
</div>

<div style="text-align: center; margin-top: 2rem;">
<h4 style="color: white;">Regula de aur a XAI:</h4>
<p style="font-size: 1.2rem; font-weight: bold; color: white;">
"Cea mai buna explicatie este cea pe care audienta ta o intelege si in care are incredere"
</p>
</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Ai parcurs cu succes aplicatia de comparatie SHAP vs LIME!</strong></p>
<p>Acum ai toate cunostintele necesare pentru a implementa XAI in proiectele tale de NLP</p>
<p><em>Experienta invata - aplica aceste concepte in practica!</em></p>
</div>
""", unsafe_allow_html=True)
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

# Model options
model_options = {
    "GPT-2": {
        "model_name": "gpt2",
        "description": "Generative Pre-trained Transformer 2",
        "parameters": "117M",
        "vocab_size": "50,257",
        "strengths": ["Rapid", "Usor de folosit", "Bun pentru incepatori"],
        "weaknesses": ["Uneori incoherent", "Texte scurte", "Repetitiv"]
    },
    "Qwen2-0.5B": {
        "model_name": "Qwen/Qwen2-0.5B",
        "description": "Qwen 2.0 - Compact and efficient model",
        "parameters": "0.5B",
        "vocab_size": "151,936",
        "strengths": ["Foarte rapid", "Modern", "Eficient", "Multilingual"],
        "weaknesses": ["Mai mic vocabular activ", "Texte mai scurte", "Capacitate limitata"]
    },
    "OPT-350M": {
        "model_name": "facebook/opt-350m",
        "description": "Meta's Open Pre-trained Transformer",
        "parameters": "350M",
        "vocab_size": "50,272",
        "strengths": ["Meta/Facebook", "Bun compromis", "Rapid", "Open source"],
        "weaknesses": ["Mai putin specializat", "Performanta moderata", "Capacitate limitata"]
    }
}

# Model selection
selected_model_name = st.selectbox(
    "Alege modelul pentru generarea de text:",
    list(model_options.keys()),
    help="Fiecare model are caracteristici diferite - vezi detaliile mai jos"
)

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

# Model comparison table
with st.expander("Compara toate modelele disponibile", expanded=False):
    comparison_data = []
    for name, info in model_options.items():
        comparison_data.append({
            "Model": name,
            "Parametri": info["parameters"],
            "Vocabular": info["vocab_size"],
            "Puncte forte": ", ".join(info["strengths"][:2]),
            "Limitari": ", ".join(info["weaknesses"][:2])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Strengths and weaknesses detail
    cols = st.columns(len(model_options))
    for i, (name, info) in enumerate(model_options.items()):
        with cols[i]:
            color = "success-box" if name == selected_model_name else "info-box"
            st.markdown(f"""
            <div class="{color}">
            <h4>{name}</h4>
            <h5>✅ Avantaje:</h5>
            <ul>
            {''.join([f'<li>{strength}</li>' for strength in info['strengths']])}
            </ul>
            <h5>⚠️ Limitari:</h5>
            <ul>
            {''.join([f'<li>{weakness}</li>' for weakness in info['weaknesses']])}
            </ul>
            </div>
            """, unsafe_allow_html=True)

# Load model
st.markdown('<h2 class="section-header">Generare interactiva cu analiza XAI</h2>', unsafe_allow_html=True)

# Load the selected model
model_name = selected_model["model_name"]

with st.spinner(f"Se incarca modelul {selected_model_name}..."):
    tokenizer, model = load_generation_model(model_name)

if tokenizer is None or model is None:
    st.error("Nu s-a putut incarca modelul. Verifica conexiunea la internet si dependentele.")
    st.stop()

# Input section
st.markdown('<h3 class="subsection-header">Configureaza generarea de text:</h3>', unsafe_allow_html=True)

# Model-specific example prompts
if selected_model_name == "GPT-2":
    examples = {
        "Tehnologie - AI": "The future of artificial intelligence is",
        "Stiinta - Spatiu": "Space exploration will help humanity",
        "Educatie": "The most important skill students need to learn",
        "Mediu - Clima": "Climate change requires immediate action because",
        "Arta - Creativitate": "Creative writing helps people express",
        "Poveste - Aventura": "Once upon a time, in a distant galaxy, there was a brave explorer who",
        "Filosofico": "The meaning of life can be understood through",
        "Sport - Motivatie": "Athletes achieve greatness by"
    }
elif selected_model_name == "Qwen2-0.5B":
    examples = {
        "Multilingual - EN": "The benefits of learning multiple languages include",
        "Multilingual - CN": "人工智能的未来发展趋势是",
        "Technical Explanation": "Machine learning algorithms work by",
        "Creative Writing": "In a world where technology and nature coexist,",
        "Business Analysis": "The key factors for startup success are",
        "Educational Content": "The scientific method involves these steps:",
        "Cultural Exchange": "Understanding different cultures helps us",
        "Innovation": "The next breakthrough in renewable energy might be"
    }
elif selected_model_name == "OPT-350M":
    examples = {
        "General Knowledge": "The most important discovery in science was",
        "Technology": "Artificial intelligence will change the world by",
        "Creative Writing": "In the year 2050, people will live in cities that",
        "Education": "The best way to learn new skills is",
        "Philosophy": "The meaning of happiness can be found through",
        "Innovation": "The next big technological breakthrough will be",
        "Society": "Social media has transformed communication by",
        "Future": "In the future, work will be different because"
    }

# Model-specific information
model_info_colors = {
    "GPT-2": "info-box",
    "Qwen2-0.5B": "success-box", 
    "OPT-350M": "warning-box"
}

model_tips = {
    "GPT-2": "💡 *Tip pentru GPT-2*: Foloseste prompt-uri scurte si directe. Modelul poate deveni repetitiv cu texte lungi.",
    "Qwen2-0.5B": "💡 *Tip pentru Qwen2-0.5B*: Model compact si modern, suporta multiple limbi. Functioneaza bine cu prompt-uri scurte si directe. Ideal pentru deployment-uri cu resurse limitate.",
    "OPT-350M": "💡 *Tip pentru OPT-350M*: Model de la Meta/Facebook, bun compromis intre performanta si eficienta. Functioneaza bine cu prompt-uri generale si explicative."
}

st.markdown(f"""
<div class="{model_info_colors[selected_model_name]}">
<h4>Modelul selectat: {selected_model_name}</h4>
<p>{model_tips[selected_model_name]}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    selected_example = st.selectbox(
        "Selecteaza un prompt exemplu:",
        ["Prompt personalizat"] + list(examples.keys())
    )
    
    if selected_example == "Prompt personalizat":
        default_prompt = "Write your prompt here..."
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
    if prompt.strip() and prompt != "Write your prompt here...":
        # Store the generation request in session state
        st.session_state['current_generation'] = {
            'prompt': prompt,
            'max_length': max_length,
            'temperature': temperature,
            'model_name': selected_model_name
        }
        with st.spinner("Se genereaza textul si se analizeaza procesul..."):
            
            # Generate text with analysis
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
                    
                    prompt_tokens = tokenizer.encode(prompt)
                    total_tokens = tokenizer.encode(generated_text)
                    generated_tokens = total_tokens[len(prompt_tokens):]
                    
                    create_metric_box("Token-uri prompt", str(len(prompt_tokens)), "Input")
                    create_metric_box("Token-uri generate", str(len(generated_tokens)), "Output nou")
                    create_metric_box("Total token-uri", str(len(total_tokens)), "Complet")
                    
                    # Generation efficiency
                    efficiency = len(generated_tokens) / max_length * 100 if max_length > 0 else 0
                    create_metric_box("Eficienta", f"{efficiency:.1f}%", "Utilizare parametri")
                
                # Token-by-token analysis
                st.markdown('<h2 class="section-header">Analiza token cu token</h2>', unsafe_allow_html=True)
                
                # Store analysis in session state to prevent recalculation on selectbox change
                # Use a combination of prompt and generated text length as key
                analysis_key = f"step_analysis_{len(prompt)}_{len(generated_tokens)}"
                current_generation_id = f"{prompt}_{generated_text}"
                
                # Check if we have the analysis for this exact generation
                if (analysis_key not in st.session_state or 
                    st.session_state.get(f"{analysis_key}_id") != current_generation_id):
                    # Get detailed generation info with probabilities
                    with torch.no_grad():
                        # Re-run generation to get step-by-step probabilities
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
                    st.markdown('<h3 class="subsection-header">Diversitatea si creativitatea</h3>', unsafe_allow_html=True)
                    
                    # Calculate diversity metrics
                    all_chosen_probs = [step['chosen_prob'] for step in step_by_step_analysis]
                    
                    if all_chosen_probs:
                        avg_prob = np.mean(all_chosen_probs)
                        min_prob = np.min(all_chosen_probs)
                        max_prob = np.max(all_chosen_probs)
                        std_prob = np.std(all_chosen_probs)
                        
                        metrics_data = [
                            {"Metric": "Probabilitate medie", "Valoare": f"{avg_prob:.1%}", 
                             "Interpretare": "Mai mare = mai predictibil"},
                            {"Metric": "Prob. minima", "Valoare": f"{min_prob:.1%}", 
                             "Interpretare": "Token-ul cel mai riscant ales"},
                            {"Metric": "Prob. maxima", "Valoare": f"{max_prob:.1%}", 
                             "Interpretare": "Token-ul cel mai sigur ales"},
                            {"Metric": "Deviatie standard", "Valoare": f"{std_prob:.3f}", 
                             "Interpretare": "Mai mare = mai diverse alegeri"}
                        ]
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        
                        # Creativity assessment
                        if avg_prob > 0.5:
                            creativity_assessment = "Conservativ - alege token-uri sigure"
                        elif avg_prob > 0.2:
                            creativity_assessment = "Echilibrat - mix intre siguranta si creativitate"
                        else:
                            creativity_assessment = "Creativ - alege token-uri mai riscanti"
                        
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>Evaluarea creativitatii:</h4>
                        <p style="font-size: 1.1rem;"><strong>{creativity_assessment}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h3 class="subsection-header">Evolutia probabilitatilor</h3>', unsafe_allow_html=True)
                    
                    if len(all_chosen_probs) > 1:
                        # Plot probability evolution
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        steps = range(1, len(all_chosen_probs) + 1)
                        ax.plot(steps, all_chosen_probs, 'o-', linewidth=2, markersize=6, color='#1f77b4')
                        
                        ax.axhline(y=np.mean(all_chosen_probs), color='red', linestyle='--', 
                                  alpha=0.7, label='Medie')
                        
                        ax.set_xlabel('Pasul generarii')
                        ax.set_ylabel('Probabilitatea token-ului ales')
                        ax.set_title('Evolutia increderii modelului')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Add value labels
                        for step, prob in zip(steps, all_chosen_probs):
                            ax.annotate(f'{prob:.2f}', (step, prob), 
                                      textcoords="offset points", xytext=(0,10), ha='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    else:
                        st.info("Nu sunt suficiente date pentru vizualizarea evolutiei.")

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

# Educational section
st.markdown('<h2 class="section-header">Importanta XAI in generarea de text</h2>', unsafe_allow_html=True)

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
    <li><strong>Bias Detection:</strong> Identificarea prejudecatilor in generare</li>
    </ul>
    
    <h4>Pentru utilizatori:</h4>
    <ul>
    <li><strong>Transparenta:</strong> Intelegerea procesului de generare</li>
    <li><strong>Incredere:</strong> Validarea calitatii output-ului</li>
    <li><strong>Control:</strong> Ajustarea parametrilor pentru stiluri diferite</li>
    <li><strong>Educatie:</strong> Invatarea despre AI generativ</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
    <h3>Limitari si provocari</h3>
    <h4>Limitari XAI:</h4>
    <ul>
    <li><strong>Complexitate:</strong> Modelele mari sunt greu de interpretat complet</li>
    <li><strong>Context:</strong> Dependente pe distante lungi sunt dificil de analizat</li>
    <li><strong>Emergenta:</strong> Comportamente emergente nu sunt previzibile</li>
    <li><strong>Scalabilitate:</strong> Analiza detaliata pentru texte lungi este costisitoare</li>
    </ul>
    
    <h4>Bune practici:</h4>
    <ul>
    <li>Analizeaza sample reprezentative, nu doar cazuri izolate</li>
    <li>Compara comportamentul cu diferite temperature</li>
    <li>Documenteaza patterns observate pentru imbunatatiri</li>
    <li>Foloseste analiza pentru ghidarea utilizatorilor</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Ai explorat lumea generarii automate de text!</strong></p>
<p>Descopera in urmatoarea pagina comparatia detaliata intre SHAP si LIME</p>
</div>
""", unsafe_allow_html=True)
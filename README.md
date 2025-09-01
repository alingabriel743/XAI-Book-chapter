# XAI in Procesarea Limbajului Natural

Aplicatie interactiva Streamlit pentru demonstrarea tehnicilor de Explicabilitate in Inteligenta Artificiala (XAI) aplicata procesarii limbajului natural, cu focus pe SHAP si LIME.

## Caracteristici

### Sectiuni interactive

1. **Introducere** - Concepte fundamentale XAI, SHAP si LIME
2. **Analiza sentimentelor** - Demonstratii interactive cu modele transformer
3. **Traducere automata** - Explicarea atentiei in modelele de traducere
4. **Generare text** - Analiza procesului de generare cu GPT-2
5. **Comparatie SHAP vs LIME** - Evaluare comparativa a metodelor

### Functionalitati XAI

- **SHAP (Shapley Additive exPlanations)**:
  - Explicatii bazate pe teoria jocurilor
  - Analiza contributiei fiecarui token/cuvant
  - Vizualizari waterfall si importance plots
  - Explicatii globale si locale

- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Explicatii locale pentru instante individuale
  - Perturbari ale textului de intrare
  - Vizualizari intuitive

### Modele suportate

- **Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Translation**: `Helsinki-NLP/opus-mt-en-ro` (Engleza -> Romana)
- **Text Generation**: `GPT-2`

## Instalare

### Prerequisite
- Python 3.8+
- pip sau conda

### Pasi de instalare

1. **Cloneaza sau descarca proiectul**:
```bash
cd "Capitol XAI"
```

2. **Creaza un mediu virtual (recomandat)**:
```bash
# Folosind venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# sau
venv\Scripts\activate     # Windows

# Folosind conda
conda create -n xai-nlp python=3.9
conda activate xai-nlp
```

3. **Instaleaza dependentele**:
```bash
pip install -r requirements.txt
```

### Dependente principale

- `streamlit` - Framework pentru aplicatia web
- `shap` - Explicabilitate SHAP
- `lime` - Explicabilitate LIME  
- `transformers` - Modele Hugging Face
- `torch` - PyTorch pentru modele
- `matplotlib`, `seaborn` - Vizualizari
- `pandas`, `numpy` - Procesarea datelor

## Rulare

### Start aplicatia Streamlit:
```bash
streamlit run app.py
```

Aplicatia va fi disponibila la: `http://localhost:8501`

### Configurare alternativa

Pentru sisteme cu resurse limitate, poti modifica modelele in `app.py`:
```python
# Modele mai mici pentru teste rapide
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment
model_name = "Helsinki-NLP/opus-mt-en-de"                       # Translation (en→de)
```

## Utilizare

### 1. Analiza sentimentelor
- Introdu text in engleza
- Vizualizeaza predictiile modelului
- Exploreaza explicatiile SHAP si LIME
- Compara importanta diferitelor cuvinte

### 2. Traducere automata  
- Introdu text in engleza pentru traducere in romana
- Analizeaza matricile de atentie
- Intelege care cuvinte influenteaza traducerea

### 3. Generare text
- Ofera un prompt pentru GPT-2
- Ajusteaza parametrii (lungime, temperatura)
- Analizeaza probabilitatile token-urilor generate

### 4. Comparatie metode
- Studiaza diferentele intre SHAP si LIME
- Intelege cand sa folosesti fiecare metoda
- Recomandari practice de implementare

## Exemple de utilizare

### Sentiment Analysis
```python
text = "I absolutely love this new smartphone! The camera quality is amazing."
# Rezultat: Sentiment pozitiv cu explicatii detaliate
```

### Translation  
```python
text = "The weather is beautiful today and I feel very happy."
# Rezultat: "Vremea este frumoasa astazi si ma simt foarte fericit."
```

### Text Generation
```python
prompt = "The future of artificial intelligence is"
# Rezultat: Text generat cu analiza probabilitatilor
```

## Personalizare

### Adaugarea de noi modele:
1. Modifica functiile `@st.cache_resource` in `app.py`
2. Adauga noul model in requirements.txt daca este necesar
3. Adapteaza functiile de predictie

### Stilizare CSS:
Modifica sectiunea CSS din `app.py` pentru a personaliza aspectul aplicatiei.

## Performanta

### Cerinte de sistem:
- **RAM**: Minimum 4GB, recomandat 8GB+
- **Storage**: ~2GB pentru modele
- **CPU**: Orice procesor modern (GPU optional)

### Optimizari:
- Modelele sunt cached cu `@st.cache_resource`
- Incarcare lazy a modelelor
- Limitare la 512 tokens pentru eficienta

## Contributii

Proiectul este creat pentru uz educational in cadrul unei carti despre XAI. 

### Imbunatatiri sugerate:
- [ ] Suport pentru mai multe limbi
- [ ] Modele mai mari (BERT, GPT-3.5)
- [ ] Export rezultate (PDF, CSV)
- [ ] Batch processing pentru multiple texte
- [ ] Integrare cu APIs externe

## Resurse suplimentare

- [Documentatia SHAP](https://shap.readthedocs.io/)
- [Documentatia LIME](https://lime-ml.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io/)


## Nota importanta

Aplicatia descarca modele mari la prima rulare (~1-2GB). Asigura-te ca ai o conexiune internet stabila si spatiu suficient pe disk.

---

*Creat pentru demonstrarea conceptelor XAI in procesarea limbajului natural*
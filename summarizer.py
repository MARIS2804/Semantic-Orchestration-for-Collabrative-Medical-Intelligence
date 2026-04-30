import pandas as pd
import spacy
import re
import streamlit as st


# --- 1. LOAD SCISPACY MODEL ---
# This model is specifically for biomedical, scientific text
@st.cache_resource
def load_medical_nlp():
    try:
        # en_core_sci_sm is the efficient version
        return spacy.load("en_core_sci_sm")
    except:
        st.error("ScispaCy model not found. Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
        return None

# --- 2. LOAD DATA ---
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')
    df.columns = df.columns.str.strip()
    df['Normal_Min'] = pd.to_numeric(df['Normal_Min'], errors='coerce')
    df['Normal_Max'] = pd.to_numeric(df['Normal_Max'], errors='coerce')
    return df

# --- 3. THE SEMANTIC CLEANING ENGINE ---
def extract_clean_entities(text, lab_results_list, nlp):
    doc = nlp(text)
    cleaned_entities = []
    
    # 1. THE "GRAPH KILLER" STOP LIST
    # These words add noise to a knowledge graph and should be stripped
    STRICT_STOP = {
        # Demographics
        'male', 'female', 'woman', 'man', 'year', 'old', 'years', 'months',
        # Clinical Verbs/Fillers
        'history', 'reports', 'reported', 'presents', 'presented', 'showing', 
        'signs', 'clinical', 'diagnosis', 'diagnosed', 'ordered', 'result', 
        'results', 'findings', 'noted', 'visible', 'rule', 'back', 'came',
        # Non-Specific Adjectives
        'significant', 'excessive', 'frequent', 'poorly', 'acute', 'onset', 
        'chronic', 'severe', 'mild', 'moderate', 'persistent', 'distinct', 
        'prominent', 'visible', 'deep', 'crushing', 'radiating', 'likely'
    }

    # Identify existing labs to avoid redundancy
    existing_labs = [re.sub(r'\s(high|low|normal)$', '', lab).lower() for lab in lab_results_list]

    for ent in doc.ents:
        # A. Fix hyphen spacing and normalize
        raw_text = re.sub(r'\s*-\s*', '-', ent.text).lower().strip()
        
        # B. PHRASE SCRUBBING
        # Split the phrase and remove stop words from INSIDE the phrase
        # e.g., "patient reports significant joint pain" -> "joint pain"
        words = raw_text.split()
        filtered_words = [w for w in words if w not in STRICT_STOP and not nlp.vocab[w].is_stop]
        
        # Reconstruct the phrase
        clean_phrase = " ".join(filtered_words)
        
        # Remove any leftover punctuation like brackets or dots
        clean_phrase = re.sub(r'[^\w\s-]', '', clean_phrase).strip()

        # C. FINAL VALIDATION RULES
        # 1. Skip if empty after scrubbing
        if not clean_phrase:
            continue
            
        # 2. Skip demographics/ages (Regex for "45-year-old" or "32 year")
        if re.search(r'\d+', clean_phrase) or "year-old" in clean_phrase:
            continue
            
        # 3. Skip redundant lab mentions (if we already have the lab status)
        if any(lab in clean_phrase for lab in existing_labs) and ("test" in raw_text or "level" in raw_text):
            continue
            
        # 4. Length check (ignore single letters or common noise fragments)
        if len(clean_phrase) > 2:
            # Avoid adding generic body parts unless they are part of a finding
            if clean_phrase not in ['bridge', 'nose', 'cheeks', 'face', 'chest', 'jaw', 'scleral']:
                cleaned_entities.append(clean_phrase)

    return list(set(cleaned_entities))

# --- 4. STREAMLIT UI ---
st.title("🩺 h-medgraph: Advanced Clinical Input")
nlp = load_medical_nlp()
ref_df = load_data('acp_reference_range.csv')

if 'added_labs' not in st.session_state:
    st.session_state.added_labs = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("Lab Entry")
    test_label = st.selectbox("Search Lab", options=[""] + sorted(ref_df['Labels'].tolist()))
    if test_label:
        row = ref_df[ref_df['Labels'] == test_label].iloc[0]
        val = st.number_input(f"Value ({row['units']})", step=0.01)
        
        # Math logic
        status = "high" if val > row['Normal_Max'] else "low" if val < row['Normal_Min'] else "normal"
        current_entry = f"{test_label} {status}"
        
        if st.button("➕ Add Lab"):
            st.session_state.added_labs.append(current_entry)

    for lab in st.session_state.added_labs: st.code(lab)

with col2:
    st.subheader("Doctor's Notes")
    notes = st.text_area("Observations", height=200, placeholder="Enter clinical description...")

# --- 5. FINAL GRAPH GENERATION ---
st.divider()
if st.button("🚀 Generate Graph Nodes"):
    if nlp is None: 
        st.error("NLP Model missing.")
    else:
        # Combine Lab Logic + ScispaCy Semantic Extraction
        final_nodes = list(st.session_state.added_labs)
        
        if notes:
            with st.spinner("ScispaCy Semantic Analysis..."):
                med_entities = extract_clean_entities(notes, final_nodes, nlp)
                final_nodes.extend(med_entities)
        
        # Deduplicate
        final_nodes = list(set(final_nodes))
        
        st.subheader("✅ h-medgraph Final Output")
        st.write(final_nodes)
        st.json(final_nodes)
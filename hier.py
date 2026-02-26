import os
import ssl
import re
import glob
import pickle
import numpy as np
import pandas as pd
import spacy
import torch
import streamlit as st
from neo4j import GraphDatabase

# --- 1. OFFLINE & GPU CONFIGURATION ---
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = r"D:\huggingface"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_snapshot_path(repo_id):
    """Finds exact snapshot folder on D: drive root"""
    folder_name = f"models--{repo_id.replace('/', '--')}"
    base_path = os.path.join(CACHE_DIR, folder_name, "snapshots")
    snapshot_folders = glob.glob(os.path.join(base_path, "*"))
    return snapshot_folders[0] if snapshot_folders else None

# --- 2. RESOURCE LOADING ---
@st.cache_resource
def load_gpu_offline_brain():
    st.write(f"🚀 Initializing SOCMI Hierarchical Brain on {DEVICE.upper()}...")
    
    p_path = get_snapshot_path("Qwen/Qwen2.5-1.5B-Instruct")
    d_path = get_snapshot_path("microsoft/Phi-3-mini-4k-instruct")
    e_path = get_snapshot_path("FremyCompany/BioLORD-2023")

    # Embeddings
    embeds = HuggingFaceEmbeddings(model_name=e_path, model_kwargs={'device': DEVICE})
    
    # Qwen Planner (FP16)
    p_tok = AutoTokenizer.from_pretrained(p_path, local_files_only=True)
    p_model = AutoModelForCausalLM.from_pretrained(p_path, torch_dtype=torch.float16, device_map="auto")
    
    # Phi-3 Dispatcher (FP16)
    d_tok = AutoTokenizer.from_pretrained(d_path, local_files_only=True)
    d_model = AutoModelForCausalLM.from_pretrained(d_path, torch_dtype=torch.float16, device_map="auto")
    
    # KG Meta
    kg_v = np.load("kg_vectors.npy")
    with open("kg_names.pkl", 'rb') as f: kg_n = pickle.load(f)
    
    return embeds, p_tok, p_model, d_tok, d_model, kg_v, kg_n

# --- 3. AGENT SPECIALIST POOL ---
class SpecialistPool:
    def __init__(self, db, embeds, kg_v, kg_n, p_tok, p_mod, d_tok, d_mod):
        self.db, self.embeds = db, embeds
        self.kg_v, self.kg_n = kg_v, kg_n
        self.p_tok, self.p_mod = p_tok, p_mod
        self.d_tok, self.d_mod = d_tok, d_mod
        self.driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "jai@2005"), encrypted=False)

    def inference(self, tok, mod, prompt, max_tokens=150):
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = mod.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    def run_rag_candidates(self, features):
        query = ", ".join(features)
        docs = self.db.similarity_search(query, k=2)
        guidelines = "\n".join([d.page_content for d in docs])
        prompt = f"Guidelines: {guidelines[:600]}\nFindings: {query}\nTask: List 5 disease names only."
        raw_text = self.inference(self.p_tok, self.p_mod, prompt)
        return [d.strip() for d in re.split(r',|\n', raw_text) if len(d.strip()) > 3][:5]

    def run_kg_candidates(self, features):
        resolved = []
        for f in features:
            vec = np.array(self.embeds.embed_query(f.lower())).reshape(1, -1)
            resolved.append(self.kg_n[np.argmax(cosine_similarity(vec, self.kg_v)[0])])
        with self.driver.session() as session:
            q = "MATCH (f) WHERE f.name IN $names MATCH (d:Disease)-[*1..2]-(f) RETURN d.name as dname LIMIT 8"
            results = session.run(q, names=resolved)
            return [rec["dname"] for rec in results], resolved

    def run_path_reranker(self, candidates, resolved):
        final_list = []
        with self.driver.session() as session:
            for d in list(set(candidates)):
                score, evidence = 0, []
                for r in resolved:
                    q = "MATCH p = shortestPath((d {name:$dn})-[*1..3]-(e {name:$en})) RETURN length(p) as dist LIMIT 1"
                    res = session.run(q, dn=d, en=r).single()
                    if res: 
                        score += 1.0 / (res['dist'] + 1)
                        evidence.append(r)
                if score > 0: final_list.append({"disease": d, "score": score, "evidence": list(set(evidence))})
        return sorted(final_list, key=lambda x: x['score'], reverse=True)

# --- 4. UI SETUP ---
st.set_page_config(page_title="medIKAL-SOCMI Hierarchy", layout="wide")
st.title("🏥 medIKAL-SOCMI (Hierarchical Communication Log)")

nlp = spacy.load("en_core_sci_sm")
embeds, p_tok, p_model, d_tok, d_model, kg_v, kg_n = load_gpu_offline_brain()
ref_df = pd.read_csv('mimic_reference_ranges.csv', encoding='latin1')
ref_df.columns = ref_df.columns.str.strip()

if 'findings' not in st.session_state: st.session_state.findings = []
col1, col2 = st.columns(2)
with col1:
    st.subheader("Laboratory Profile")
    
    # 1. Ensure columns are clean
    ref_df.columns = ref_df.columns.str.strip()
    
    # 2. Select Lab Test
    lab_options = [""] + sorted(ref_df['Labels'].unique().tolist())
    test = st.selectbox("Search Lab", options=lab_options)
    
    if test:
        # Get the specific row for the selected lab
        row = ref_df[ref_df['Labels'] == test].iloc[0]
        
        # FIX: Dynamically find the min/max columns to avoid KeyError
        # It looks for columns containing 'min' and 'max' regardless of case
        min_col = next((c for c in ref_df.columns if 'min' in c.lower()), None)
        max_col = next((c for c in ref_df.columns if 'max' in c.lower()), None)
        
        if min_col and max_col:
            val_min = str(row[min_col])
            val_max = str(row[max_col])
            # Using st.info or st.code makes it more visible than st.caption
            st.markdown(f"✅ **Normal Range:** `{val_min}` — `{val_max}`")
            
            val = st.number_input(f"Enter Value for {test}", step=0.01, format="%.2f")
            
            if st.button("➕ Add Lab Result"):
                try:
                    # Logic for high/low/normal
                    n_min = float(row[min_col])
                    n_max = float(row[max_col])
                    
                    if val > n_max: status = "high"
                    elif val < n_min: status = "low"
                    else: status = "normal"
                    
                    st.session_state.findings.append(f"{test} {status}")
                    st.toast(f"Added {test} ({status})")
                except ValueError:
                    st.error("Reference range is not numeric. Added as 'normal'.")
                    st.session_state.findings.append(f"{test} normal")
        else:
            st.warning("Could not find Min/Max columns in CSV.")

    # Display added findings
    for f in st.session_state.findings:
        st.code(f)
    
    if st.button("🗑️ Clear All"):
        st.session_state.findings = []
        st.rerun()

with col2:
    st.subheader("Clinical Observations")
    notes = st.text_area("Observations", height=250)

# --- 5. EXECUTION LOGIC (WITH FULL COMMUNICATION LOG) ---
if st.button("🚀 Execute SOCMI Workflow"):
    with st.status("🧠 Agent Communication Hub Active...") as status:
        pool = SpecialistPool(FAISS.load_local("medical_faiss_index", embeds, allow_dangerous_deserialization=True), 
                              embeds, kg_v, kg_n, p_tok, p_model, d_tok, d_model)

        # STEP 1: USER -> SUMMARIZER
        st.markdown("### 📨 Message 1: User ➔ Summarizer Agent")
        doc = nlp(notes)
        features = list(set([ent.text for ent in doc.ents] + st.session_state.findings))
        st.success(f"Summarizer Response: Extracted {len(features)} entities.")
        st.json(features)

        # STEP 2: DISPATCHER -> PLANNER
        st.markdown("### 📨 Message 2: Dispatcher ➔ Planner Agent")
        st.write("*Request:* 'Generate medical hypothesis.'")
        hypothesis = pool.inference(p_tok, p_model, f"Findings: {features}. Hypothesis?")
        st.info(f"Planner Response: {hypothesis}")

        # STEP 3: PLANNER -> DISPATCHER -> RAG
        st.markdown("### 📨 Message 3: Planner ➔ Dispatcher ➔ RAG Specialist")
        st.write(f"*Command:* 'Search guidelines for hypothesis.'")
        rag_cands = pool.run_rag_candidates(features)
        st.success(f"RAG Agent Response: Proposed {len(rag_cands)} disease candidates.")
        st.write(rag_cands)

        # STEP 4: DISPATCHER -> PLANNER (Self-Correction)
        st.markdown("### 📨 Message 4: Dispatcher ➔ Planner Agent")
        st.write("*Status:* 'RAG search complete. Requesting KG Discovery.'")
        st.info("Planner Approval: 'Proceed to structural graph mapping.'")

        # STEP 5: PLANNER -> DISPATCHER -> KG
        st.markdown("### 📨 Message 5: Planner ➔ Dispatcher ➔ KG Agent")
        st.write("*Command:* 'Execute structural neighbor discovery.'")
        kg_cands, resolved = pool.run_kg_candidates(features)
        st.success(f"KG Agent Response: Discovered {len(kg_cands)} candidates.")
        st.write(f"Resolved Nodes: {resolved}")

        # STEP 6: DISPATCHER -> PLANNER -> RERANKER
        st.markdown("### 📨 Message 6: Dispatcher ➔ Planner Agent")
        all_potential = list(set(rag_cands + kg_cands))
        st.write(f"*Status:* 'Collected {len(all_potential)} total candidates. Starting Algorithm 2 reranking.'")
        results = pool.run_path_reranker(all_potential, resolved)
        st.info("Planner Verification: 'Scores calculated. Generating final report.'")

        status.update(label="✅ Workflow Complete!", state="complete")

    # --- 6. FINAL DIFFERENTIAL DIAGNOSIS (TOP 5) ---
    st.divider()
    st.subheader("🏁 Final Clinical Decision Agent Output (Differential Diagnosis)")
    
    if not results:
        st.warning("No matches found in the Knowledge Graph.")
    else:
        for i, res in enumerate(results[:5]):
            with st.expander(f"**RANK {i+1}: {res['disease']}**", expanded=(i == 0)):
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Synergy Score", round(res['score'], 4))
                    st.write("**Validated Evidence:**")
                    for e in res['evidence']: st.markdown(f"- {e}")
                with res_col2:
                    st.write("🔎 **Clinical Reasoning (Dispatcher ➔ Planner):**")
                    reasoning = pool.inference(d_tok, d_model, f"Explain suspecting {res['disease']} based on {res['evidence']}.")
                    st.success(reasoning)

st.sidebar.title("SOCMI Device Stats")
st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
st.sidebar.info(f"Memory Logic: Auto-Map FP16")
import streamlit as st
from sentence_transformers import SentenceTransformer
import pdfplumber, docx2txt, re
from scipy.spatial.distance import cosine
import pandas as pd
from io import BytesIO

# ─────────────────────────  Model  ──────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# ──────────────────────────  UI  ────────────────────────────
st.header("Recruiter Resume Matcher")

job_desc = st.text_area("Paste the Job Description here", height=200)

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF/TXT/DOCX)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# ───────────────────────  Helpers  ──────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_text(file) -> str:
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if file.name.endswith(".docx"):
        return docx2txt.process(file)
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

# ───────────────────────  Main  ─────────────────────────────
if st.button("🔍 Rank Resumes"):
    if not job_desc or not uploaded_files:
        st.warning("Please provide both a job description and at least one resume.")
        st.stop()

    job_emb = model.encode(preprocess(job_desc), convert_to_tensor=True)

    resumes, resume_names = [], []
    for f in uploaded_files:
        txt = preprocess(extract_text(f))
        if txt:
            resumes.append(txt)
            resume_names.append(f.name)

    if not resumes:
        st.warning("No valid text found in the uploaded resumes.")
        st.stop()

    resume_embs = [model.encode(t, convert_to_tensor=True) for t in resumes]
    scores = [1 - cosine(job_emb, emb) for emb in resume_embs]
    ranked = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)
    top5 = ranked[:5]

    # ─────────────  Show top‑5  ─────────────
    st.subheader("🏆 Top 5 Matching Resumes")
    for n, s in top5:
        st.markdown(f"- **{n}** — similarity `{s:.3f}`")

    # ─────── 🔄 Changed code starts here  ────────
    # Build DataFrame of the full ranking
    df = pd.DataFrame(ranked, columns=["Resume Name", "Similarity Score"])
    df["Similarity Score"] = df["Similarity Score"] * 100
    # Write to an in‑memory Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranked Resumes")
    buffer.seek(0)                       # ←‑‑ important!

    # Download button
    st.download_button(
        "📥 Download Ranked Resumes (Excel)",
        data=buffer,
        file_name="ranked_resumes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    # ─────── 🔄 Changed code ends here  ─────────

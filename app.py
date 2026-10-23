from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq

from pathlib import Path
from io import BytesIO
from pypdf import PdfReader

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_RESUME = BASE_DIR / "resume" / "resume1.pdf"

st.set_page_config(
    page_title="CareerAI | Job Application Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Streamlit state before rendering widgets.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
        .stApp {
            background: #0b1020;
            color: #edf2ff;
        }

        [data-testid="stSidebar"] {
            background: #11182b;
            border-right: 1px solid #27324b;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            padding: 26px 28px;
            border: 1px solid #293653;
            border-radius: 20px;
            background: linear-gradient(135deg, #172443, #11182b);
            margin-bottom: 22px;
        }

        .hero h1 {
            font-size: 2.05rem;
            margin: 0 0 8px 0;
            color: #ffffff;
        }

        .hero p {
            color: #b6c4df;
            margin: 0;
            font-size: 1rem;
        }

        .status {
            color: #7ee7b2;
            font-size: 0.85rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid #273653;
            border-radius: 15px;
            padding: 12px 16px;
            background: #111a2e;
            margin-bottom: 12px;
        }

        .stButton > button {
            border-radius: 10px;
            min-height: 42px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_groq_api_key():
    """
    Read the API key from:
    1. Environment variable / local .env
    2. Streamlit Community Cloud Secrets
    """
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        # No secrets.toml file is normal for local use when .env is configured.
        api_key = ""

    return api_key or None


def extract_resume_text(pdf_source):
    """Extract text from a local PDF or uploaded Streamlit PDF."""

    if isinstance(pdf_source, (str, Path)):
        pdf_bytes = Path(pdf_source).read_bytes()
    else:
        pdf_bytes = pdf_source.getvalue()

    reader = PdfReader(BytesIO(pdf_bytes), strict=False)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    ).strip()

    if not text:
        raise ValueError(
            "No readable text found in the PDF. "
            "Please upload a text-based PDF."
        )

    return text


def create_llm():
    """Create the Groq-hosted model; this does not use local Ollama."""
    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Add it to your local .env file, "
            "or configure GROQ_API_KEY in Streamlit Cloud → App settings → Secrets."
        )

    return ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,
        max_tokens=2048,
    )


def generate_reply(user_message, resume_text, company, role):
    """Generate a response using the resume and previous conversation."""
    system_prompt = f"""
You are CareerAI, a professional job application assistant.

Use the resume below as the only source of candidate-specific facts.
Never invent experience, qualifications, achievements, contact details,
or profile links. If information is missing, omit it or ask the user.

RESUME:
--- RESUME START ---
{resume_text}
--- RESUME END ---

Target company: {company.strip() or "Not specified"}
Target role: {role.strip() or "Not specified"}

Help with personalized job application emails, cover letters, recruiter
messages, interview preparation, and related job-application tasks.
For application emails, include a subject line, professional greeting,
concise body, and closing. Keep answers accurate and professional.
"""

    messages = [("system", system_prompt)]

    # The current user message is already appended to chat_history.
    # Include earlier messages only, then add the current message once.
    for item in st.session_state.chat_history[:-1]:
        message_role = "human" if item["role"] == "user" else "ai"
        messages.append((message_role, item["content"]))

    messages.append(("human", user_message))

    response = create_llm().invoke(messages)
    return response.content


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💼 CareerAI")
    st.caption("Powered by Groq API")
    st.divider()

    st.markdown("### 🎯 Job details")
    company = st.text_input(
        "Company name",
        value="TCS",
        placeholder="e.g. TCS",
        key="company_name",
    )
    role = st.text_input(
        "Job role",
        value="AI/ML Engineer",
        placeholder="e.g. Python Developer",
        key="job_role",
    )

    st.divider()
    st.markdown("### 📄 Resume")

    uploaded_pdf = st.file_uploader(
        "Upload a resume PDF",
        type=["pdf"],
        help="Upload a PDF or use resume/resume1.pdf when running locally.",
    )

    resume_text = ""

    if uploaded_pdf is not None:
        try:
            resume_text = extract_resume_text(uploaded_pdf)
            st.success(f"Resume loaded: {uploaded_pdf.name}")
        except Exception as exc:
            st.error(f"Could not read uploaded PDF: {exc}")

    elif DEFAULT_RESUME.exists():
        try:
            resume_text = extract_resume_text(str(DEFAULT_RESUME))
            st.success("Default resume loaded")
            st.caption("resume/resume1.pdf")
        except Exception as exc:
            st.error(f"Could not read default resume: {exc}")

    else:
        st.info(
            "Upload your resume to start. For the deployed app, upload it here; "
            "your local PDF does not need to be stored in GitHub."
        )

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption(f"Model: {MODEL_NAME}")


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>👋 AI Job Application Assistant</h1>
        <p>Generate tailored application emails, cover letters, and recruiter
        messages using your resume.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="status">● Model configured: {MODEL_NAME}</p>',
    unsafe_allow_html=True,
)

if not resume_text:
    st.info(
        "Upload a readable resume PDF from the sidebar to start chatting."
    )

if not st.session_state.chat_history:
    st.markdown("### Try one of these")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✉️ Application email", use_container_width=True):
            st.session_state.pending_prompt = (
                f"Write a short professional job application email for the "
                f"{role} position at {company}, using my resume."
            )
            st.rerun()

    with col2:
        if st.button("💬 Recruiter message", use_container_width=True):
            st.session_state.pending_prompt = (
                f"Write a concise WhatsApp message to a recruiter about the "
                f"{role} role at {company}, based on my resume."
            )
            st.rerun()

    with col3:
        if st.button("📝 Cover letter", use_container_width=True):
            st.session_state.pending_prompt = (
                f"Write a concise cover letter for the {role} position at "
                f"{company}, based only on my resume."
            )
            st.rerun()


# Display existing conversation.
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])


# Process a suggested prompt or a typed chat message.
pending_prompt = st.session_state.pending_prompt
typed_prompt = st.chat_input("Message CareerAI…")
user_input = typed_prompt or pending_prompt

if user_input:
    st.session_state.pending_prompt = None

    if not resume_text:
        st.warning("Please upload a readable resume PDF first.")

    elif not get_groq_api_key():
        st.error(
            "GROQ_API_KEY is missing. Add it to your local .env file, or set "
            "GROQ_API_KEY in Streamlit Cloud → App settings → Secrets."
        )

    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("CareerAI is preparing your response…"):
                    reply = generate_reply(
                        user_message=user_input,
                        resume_text=resume_text,
                        company=company,
                        role=role,
                    )
                st.markdown(reply)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )

        except Exception as exc:
            st.error(
                f"Could not generate a response: {exc}\n\n"
                "Check your internet connection, Groq API key, and whether "
                "the model is available to your Groq account."
            )

# MedRAG Streamlit frontend: a calm medical chat over the grounded agent.
import os
import io
import re
import sys
import logging
import warnings
import contextlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Silence third-party noise before the heavy libs load; we surface our own steps.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from main import MedRAG


# ----------------------------- look & feel -----------------------------

PALETTE = {
    "teal": "#0E8C7F",
    "deep": "#0B5D54",
    "mint": "#EAF4F2",
    "ink": "#15302C",
    "line": "#CFE6E1",
}

CSS = f"""
<style>
.block-container {{ max-width: 820px; padding-top: 2.2rem; }}
#med-head {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.1rem; }}
#med-head .dot {{
    width:12px; height:12px; border-radius:50%;
    background:{PALETTE['teal']}; box-shadow:0 0 0 4px {PALETTE['mint']};
}}
#med-head h1 {{ font-size:1.7rem; margin:0; color:{PALETTE['deep']}; letter-spacing:.3px; }}
#med-sub {{ color:#5C7A74; margin:.1rem 0 1.1rem; font-size:.95rem; }}
[data-testid="stChatMessage"] {{
    background:#fff; border:1px solid {PALETTE['line']};
    border-radius:14px; padding:.4rem .9rem;
}}
.stChatMessage p {{ color:{PALETTE['ink']}; }}
[data-testid="stStatusWidget"] {{ border-radius:12px; }}
hr {{ border-color:{PALETTE['line']}; }}
.disclaimer {{ color:#7A938D; font-size:.82rem; }}
</style>
"""


# ----------------------------- progress prettifier -----------------------------

TOOL_LABELS = {
    "find_diseases_by_symptoms": "Searching diseases that match the symptoms",
    "get_disease_details": "Reading the disease record",
    "get_medicine_details": "Looking up the medicine details",
    "search_medicines": "Searching the medicine database",
}

# (regex, replacement) where replacement is a string or a fn(match) -> str|None.
RULES = [
    (r"\[tool\*?\]\s*(?:recovered\s*)?(\w+)", lambda m: TOOL_LABELS.get(m.group(1))),
    (r"Initializing MedRAG", "Starting MedRAG"),
    (r"Connected to Neo4j", "Connected to the knowledge graph"),
    (r"Indexed (\d+) symptoms, (\d+) diseases",
     lambda m: f"Indexed {int(m.group(1)):,} symptoms and {int(m.group(2)):,} diseases"),
    (r"Connected to ChromaDB \((\d+) chunks\)",
     lambda m: f"Connected to the medicine database ({int(m.group(1)):,} chunks)"),
    (r"Loaded (\d+) medicine records",
     lambda m: f"Loaded {int(m.group(1)):,} medicine records"),
    (r"Loading embedder", "Loading the embedding model"),
    (r"\[Reranker\] Loading", "Loading the reranking model"),
    (r"\[Core Agent\] Ready", "Reasoning agent ready"),
    (r"^Ready!?$", "Ready"),
    (r"Symptom retrieval", "Matching symptoms in the knowledge graph"),
    (r"Disease retrieval:\s*(.+)", lambda m: f"Reading the record for “{m.group(1).strip()}”"),
    (r"Semantic search", "Searching the medicine database"),
    (r"Expanded to (\d+) diseases", lambda m: f"Found {m.group(1)} related diseases"),
    (r"Retrieved (\d+) medicines", lambda m: f"Retrieved {m.group(1)} medicines"),
]

NOISE = ("you:", "thank you", "interactive mode", "seeds:")


def prettify(line: str):
    """Turn a raw log line into a friendly step, or None to drop it."""
    line = line.strip()
    if not line or not any(c.isalnum() for c in line):  # banners / rulers
        return None
    if any(n in line.lower() for n in NOISE):
        return None
    for pattern, repl in RULES:
        m = re.search(pattern, line)
        if m:
            return repl(m) if callable(repl) else repl
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", line)  # fallback: drop the [Tag]


class UIStream(io.TextIOBase):
    """Capture stdout line by line, forwarding clean lines and hiding warnings/errors."""

    SKIP = ("warn", "error", "fail", "traceback", "exception")

    def __init__(self, on_step):
        """Store the per-step callback and init the line buffer."""
        self.on_step = on_step
        self._buf = ""

    def write(self, s):
        """Buffer written text and emit completed lines."""
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self):
        """Emit any buffered partial line."""
        if self._buf:
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, line):
        """Prettify one line and forward it to the callback unless it's noise."""
        if any(k in line.lower() for k in self.SKIP):
            return
        step = prettify(line)
        if step:
            self.on_step(step)


@contextlib.contextmanager
def captured(on_step):
    """Route stdout through the prettifier and swallow stderr."""
    with contextlib.redirect_stdout(UIStream(on_step)), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


# ----------------------------- engine -----------------------------

@st.cache_resource(show_spinner=False)
def get_engine(_on_step):
    """Build and cache the MedRAG engine, streaming boot steps to the UI."""
    with captured(_on_step):
        med = MedRAG()
    return med


# ----------------------------- page -----------------------------

st.set_page_config(page_title="MedRAG", page_icon="\U0001FA7A", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div id="med-head"><span class="dot"></span><h1>MedRAG</h1></div>'
    '<div id="med-sub">Grounded answers about diseases and medicines.</div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Session")
    if st.button("New conversation", use_container_width=True):
        if "engine" in st.session_state:
            st.session_state.engine.reset()
        st.session_state.history = []
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p class="disclaimer">Educational use only — not medical advice. '
        "Consult a doctor.</p>",
        unsafe_allow_html=True,
    )

# First load: stream the boot steps so the wait feels alive.
if "engine" not in st.session_state:
    with st.status("Waking up MedRAG…", expanded=True) as boot:
        try:
            st.session_state.engine = get_engine(lambda step: st.write(f"• {step}"))
            boot.update(label="MedRAG is ready", state="complete", expanded=False)
        except Exception:
            boot.update(label="Could not start MedRAG", state="error")
            st.error(
                "MedRAG could not connect to its data stores. Make sure Neo4j and "
                "ChromaDB are set up (see the project README) and reload the page."
            )
            st.stop()

med = st.session_state.engine

# Replay the conversation so far.
for role, text in st.session_state.history:
    with st.chat_message(role, avatar="\U0001FA7A" if role == "assistant" else None):
        st.markdown(text)

prompt = st.chat_input("Ask about symptoms, a disease, or a medicine…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="\U0001FA7A"):
        answer_box = st.empty()
        with st.status("Thinking…", expanded=True) as status:
            try:
                with captured(lambda step: st.write(f"• {step}")):
                    answer = med.ask(prompt)
                status.update(label="Answer ready", state="complete", expanded=False)
            except Exception:
                status.update(label="Something went wrong", state="error")
                answer = "Sorry, I ran into a problem answering that. Please try again."
        answer_box.markdown(answer)

    st.session_state.history.append(("assistant", answer))

"""
app.py

AI QuizGen - Streamlit application.
Turns pasted or uploaded educational content into an interactive AI-generated
quiz (Multiple Choice, Short Answer, Fill in the Blank, True/False).

Run with:
    python -m streamlit run app.py

NOTE ON FIRST RUN: the first time you click "Generate Questions", the app
downloads the spaCy and T5 question-generation models. This is a one-time
download (cached afterwards under ~/.cache) and can take a few minutes
depending on your connection — it is not stuck.
"""

from __future__ import annotations

import re
import traceback

import streamlit as st

from utils.text_extractor import extract_text, TextExtractionError
from utils.question_generator import (
    QuestionGenerator,
    QuestionGenerationError,
)

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI QuizGen",
    page_icon="🧠",
    layout="centered",
)


def _load_css() -> None:
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


_load_css()


# --------------------------------------------------------------------------
# Cached, lazy model loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_generator() -> QuestionGenerator:
    """Loaded only the first time question generation is actually
    requested, then cached for the lifetime of the app process."""
    return QuestionGenerator().load()


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------

_DEFAULTS = {
    "extracted_text": "",
    "content_ready": False,
    "questions": [],
    "last_error": "",
    "last_traceback": "",
    "submitted": False,
    "user_answers": {},
    "shortfall": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<div class="quizgen-title">AI QuizGen</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="quizgen-subtitle">Turn your study material into intelligent quizzes</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 1. Input method
# --------------------------------------------------------------------------

st.markdown('<div class="quizgen-section-header">1. Provide Your Content</div>', unsafe_allow_html=True)

input_method = st.radio(
    "Choose an input method",
    options=["Paste Text", "Upload Document"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_method == "Paste Text":
    pasted_text = st.text_area(
        "Paste your study material here",
        height=220,
        placeholder="Paste notes, textbook content, or any educational material...",
        label_visibility="collapsed",
    )
    if pasted_text and pasted_text.strip():
        st.session_state.extracted_text = pasted_text.strip()
        st.session_state.content_ready = True
    else:
        st.session_state.content_ready = False

else:  # Upload Document
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Extracting text from document..."):
                extracted = extract_text(uploaded_file)
            st.session_state.extracted_text = extracted
            st.session_state.content_ready = True
            st.markdown(
                '<div class="quizgen-success-banner">✅ Document extracted successfully!</div>',
                unsafe_allow_html=True,
            )
        except TextExtractionError as e:
            st.session_state.content_ready = False
            st.error(str(e))
    else:
        st.session_state.content_ready = False


# --------------------------------------------------------------------------
# 2. Quiz settings
# --------------------------------------------------------------------------

st.markdown('<div class="quizgen-section-header">2. Quiz Settings</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    question_type = st.selectbox(
        "Question Type",
        options=["Multiple Choice", "Short Answer", "Fill in the Blank", "True/False"],
    )
with col2:
    difficulty = st.selectbox("Difficulty", options=["Easy", "Medium", "Hard"])

num_questions = st.select_slider(
    "Number of Questions",
    options=[5, 10, 15, 20],
    value=5,
)

# --------------------------------------------------------------------------
# 3. Generate
# --------------------------------------------------------------------------

st.markdown('<div class="quizgen-section-header">3. Generate</div>', unsafe_allow_html=True)

generate_clicked = st.button("🚀 Generate Questions", use_container_width=True)

if generate_clicked:
    st.session_state.questions = []
    st.session_state.last_error = ""
    st.session_state.last_traceback = ""
    st.session_state.submitted = False
    st.session_state.user_answers = {}
    st.session_state.shortfall = None

    content = st.session_state.extracted_text.strip()

    if not content:
        st.session_state.last_error = "Please paste some text or upload a document before generating questions."
    elif len(content.split()) < 20:
        st.session_state.last_error = (
            "Your content is too short. Please provide at least a few "
            "sentences of educational material."
        )
    else:
        try:
            with st.spinner(
                "Analyzing content and generating questions... "
                "on the very first run this also downloads the AI models "
                "(a few minutes, one time only)."
            ):
                generator = get_generator()
                questions = generator.generate_questions(
                    text=content,
                    question_type=question_type,
                    difficulty=difficulty,
                    num_questions=num_questions,
                )
            st.session_state.questions = questions
            st.session_state.shortfall = generator.last_shortfall
        except QuestionGenerationError as e:
            st.session_state.last_error = str(e)
        except Exception:
            st.session_state.last_error = (
                "Something went wrong while generating questions. Please "
                "try again, or use shorter/simpler content."
            )
            st.session_state.last_traceback = traceback.format_exc()

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.last_traceback:
        with st.expander("Show technical details"):
            st.code(st.session_state.last_traceback)


# --------------------------------------------------------------------------
# 4. Interactive quiz
# --------------------------------------------------------------------------

def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", value.strip().lower()).strip()


def _is_close_enough(user_value: str, correct_value: str) -> bool:
    """Lenient grading for free-text answers: exact match, or the
    shorter normalized string is fully contained in the longer one."""
    u, c = _normalize(user_value), _normalize(correct_value)
    if not u:
        return False
    if u == c:
        return True
    shorter, longer = (u, c) if len(u) <= len(c) else (c, u)
    return len(shorter) >= 3 and shorter in longer


if st.session_state.questions:
    if st.session_state.shortfall:
        got, requested = st.session_state.shortfall
        st.info(
            f"Generated {got} of {requested} requested questions — this "
            f"content didn't have enough distinct material for more. Try "
            f"pasting a longer passage for a full set."
        )

    st.markdown('<div class="quizgen-section-header">Quiz</div>', unsafe_allow_html=True)

    if not st.session_state.submitted:
        with st.form("quiz_form"):
            for i, q in enumerate(st.session_state.questions, start=1):
                st.markdown(
                    f'<div class="quizgen-card">'
                    f'<div class="quizgen-card-header">'
                    f'<span class="quizgen-q-number">Question {i}</span>'
                    f'<span class="quizgen-q-badge">{q["type"]} · {q["difficulty"]}</span>'
                    f'</div>'
                    f'<div class="quizgen-question-text">{q["question"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if q["type"] == "Multiple Choice" and q.get("options"):
                    letters = ["A", "B", "C", "D"]
                    labeled = [f"{letter}. {opt}" for letter, opt in zip(letters, q["options"])]
                    st.radio(
                        "Choose one:", options=labeled, index=None,
                        key=f"resp_{i}", label_visibility="collapsed",
                    )
                elif q["type"] == "True/False":
                    st.radio(
                        "Choose one:", options=["True", "False"], index=None,
                        key=f"resp_{i}", label_visibility="collapsed",
                    )
                else:  # Short Answer / Fill in the Blank
                    st.text_input(
                        "Your answer:", key=f"resp_{i}", label_visibility="collapsed",
                    )

            submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True)

        if submitted:
            st.session_state.submitted = True
            st.session_state.user_answers = {
                i: st.session_state.get(f"resp_{i}", "")
                for i in range(1, len(st.session_state.questions) + 1)
            }
            st.rerun()

    else:
        score = 0
        total = len(st.session_state.questions)

        for i, q in enumerate(st.session_state.questions, start=1):
            user_answer = st.session_state.user_answers.get(i) or ""

            if q["type"] == "Multiple Choice":
                chosen_text = user_answer.split(". ", 1)[-1] if user_answer else ""
                is_correct = chosen_text.strip().lower() == q["answer"].strip().lower()
            elif q["type"] == "True/False":
                is_correct = user_answer.strip().lower() == q["answer"].strip().lower()
            else:
                is_correct = _is_close_enough(user_answer, q["answer"])

            if is_correct:
                score += 1

            verdict_class = "quizgen-option-correct" if is_correct else ""
            verdict = "✅ Correct" if is_correct else "❌ Incorrect"

            card_html = [
                '<div class="quizgen-card">',
                '<div class="quizgen-card-header">',
                f'<span class="quizgen-q-number">Question {i}</span>',
                f'<span class="quizgen-q-badge">{q["type"]} · {q["difficulty"]}</span>',
                '</div>',
                f'<div class="quizgen-question-text">{q["question"]}</div>',
            ]

            if q["type"] == "Multiple Choice" and q.get("options"):
                letters = ["A", "B", "C", "D"]
                for letter, option in zip(letters, q["options"]):
                    css_class = "quizgen-option"
                    if option.strip().lower() == q["answer"].strip().lower():
                        css_class += " quizgen-option-correct"
                    card_html.append(f'<div class="{css_class}">{letter}. {option}</div>')

            display_user_answer = user_answer if user_answer else "(no answer given)"
            card_html.append(
                f'<div class="quizgen-answer-block">'
                f'<span class="quizgen-answer-label">Your answer:</span>&nbsp;{display_user_answer}'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;<span class="{verdict_class}">{verdict}</span></div>'
            )
            if not is_correct:
                card_html.append(
                    f'<div class="quizgen-answer-block">'
                    f'<span class="quizgen-answer-label">Correct answer:</span>&nbsp;{q["answer"]}</div>'
                )

            card_html.append('</div>')
            st.markdown("".join(card_html), unsafe_allow_html=True)

        st.markdown(
            f'<div class="quizgen-success-banner">Score: {score} / {total}</div>',
            unsafe_allow_html=True,
        )

        if st.button("🔁 Retake This Quiz"):
            st.session_state.submitted = False
            st.session_state.user_answers = {}
            for i in range(1, total + 1):
                st.session_state.pop(f"resp_{i}", None)
            st.rerun()

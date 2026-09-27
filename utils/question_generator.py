"""
utils/question_generator.py

The NLP core of AI QuizGen.

Pipeline:
    Educational Content
        -> Preprocessing (sentence segmentation, cleanup)
        -> Key-information / answer-span extraction (spaCy NER + noun chunks)
        -> Answer-aware Question Generation (fine-tuned T5 model, highlight format)
        -> Distractor generation (for MCQ)
        -> True/False statement construction
        -> Fill-in-the-blank construction
        -> Validation (dedup, empty-answer checks, MCQ shape checks)
        -> Final formatted question objects

The heavy models (spaCy pipeline, T5 question-generation model) are loaded
lazily via `QuestionGenerator.load()` so importing this module, and starting
the Streamlit app, is instant. app.py is responsible for caching the loaded
`QuestionGenerator` instance (st.cache_resource) so the models are loaded
only once per process.

MODEL CHOICE
------------
QG_MODEL_NAME below defaults to "valhalla/t5-small-qg-hl": a small
(~250 MB), CPU-friendly T5 model fine-tuned specifically for answer-aware
question generation using the "highlight" format it was trained on
(<hl> answer <hl> inside the sentence, prefixed with "generate question: ").
This keeps first-run downloads quick.

If you have bandwidth/time to spare and want noticeably more fluent
questions, swap QG_MODEL_NAME to "mrm8488/t5-base-finetuned-question-generation-ap"
(~1.2 GB) and set QG_FORMAT = "answer_context" instead of "highlight" —
both formats are implemented below in `_generate_question_for_answer`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional


class QuestionGenerationError(Exception):
    """Raised when question generation cannot proceed or produces nothing usable."""
    pass


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Question:
    q_type: str                 # "Multiple Choice" | "Short Answer" | "Fill in the Blank" | "True/False"
    question: str
    answer: str
    options: Optional[List[str]] = field(default=None)   # only for Multiple Choice
    difficulty: str = "Medium"

    def to_dict(self) -> dict:
        return {
            "type": self.q_type,
            "question": self.question,
            "answer": self.answer,
            "options": self.options,
            "difficulty": self.difficulty,
        }


# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------

QG_MODEL_NAME = "valhalla/t5-small-qg-hl"   # small & fast to download (~250 MB)
QG_FORMAT = "highlight"                      # "highlight" or "answer_context" — must match the model above
SPACY_MODEL_NAME = "en_core_web_sm"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Tries spaCy first, falls back to a
    regex splitter so the app still degrades gracefully."""
    text = text.strip()
    if not text:
        return []

    try:
        nlp = _get_spacy()
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sentences:
            return sentences
    except Exception:
        pass

    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _is_usable_sentence(sentence: str, min_words: int = 6) -> bool:
    words = sentence.split()
    if len(words) < min_words:
        return False
    if not re.search(r"[a-zA-Z]", sentence):
        return False
    return True


# --------------------------------------------------------------------------
# Lazy singletons for heavy models
# --------------------------------------------------------------------------

_spacy_nlp = None
_qg_tokenizer = None
_qg_model = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError as exc:
            raise QuestionGenerationError(
                f"The spaCy model '{SPACY_MODEL_NAME}' is not installed. "
                f"Run: python -m spacy download {SPACY_MODEL_NAME}"
            ) from exc
    return _spacy_nlp


def _get_qg_model():
    global _qg_tokenizer, _qg_model
    if _qg_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_NAME)
        _qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME)
    return _qg_tokenizer, _qg_model


def preload_models() -> None:
    """Force-load every heavy model. Called once, up front, by app.py's
    cached factory function so all subsequent generations are fast."""
    _get_spacy()
    _get_qg_model()


# --------------------------------------------------------------------------
# Key-information / candidate-answer extraction
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    text: str          # the answer span
    sentence: str       # sentence it came from
    label: str          # "ENTITY:<TYPE>" or "NOUN_CHUNK"


# Generic/vague head nouns that produce meaningless answers or distractors
# on their own (e.g. "the process", "this process", "the same method") —
# these are almost always anaphoric references to something named
# elsewhere, not real content.
_VAGUE_HEADS = {
    "process", "thing", "things", "part", "parts", "type", "types",
    "way", "ways", "method", "methods", "system", "systems", "form",
    "forms", "kind", "kinds", "example", "examples", "case", "cases",
    "point", "points", "result", "results", "purpose", "reason",
    "reasons", "step", "steps", "stage", "stages", "aspect", "aspects",
}


def _extract_candidates(sentence: str) -> List[Candidate]:
    """Find important spans in a sentence that are worth asking about:
    named entities first (dates, people, places, quantities, etc.),
    then noun chunks as a fallback."""
    nlp = _get_spacy()
    doc = nlp(sentence)

    candidates: List[Candidate] = []
    seen = set()

    for ent in doc.ents:
        key = ent.text.strip().lower()
        if key and key not in seen and len(ent.text.strip()) > 1:
            candidates.append(Candidate(ent.text.strip(), sentence, f"ENTITY:{ent.label_}"))
            seen.add(key)

    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        key = text.lower()
        if len(text.split()) < 1 or key in seen:
            continue
        if chunk.root.pos_ == "PRON":
            continue
        if len(text) <= 2:
            continue
        # Skip chunks that are entirely stopwords/determiners (e.g. "this", "the same")
        if all(tok.is_stop or tok.is_punct for tok in chunk):
            continue
        # Skip vague references like "the process" / "this process" / "the
        # same method" — a chunk headed by a generic noun with no other
        # descriptive content word is almost never a real answer.
        content_words = [tok for tok in chunk if not (tok.is_stop or tok.is_punct)]
        if chunk.root.lemma_.lower() in _VAGUE_HEADS and len(content_words) <= 1:
            continue
        candidates.append(Candidate(text, sentence, "NOUN_CHUNK"))
        seen.add(key)

    return candidates


def _rank_candidates(candidates: List[Candidate], difficulty: str) -> List[Candidate]:
    """Order candidates so easier, more explicit facts (named entities)
    are preferred for 'Easy', while longer / conceptual noun chunks are
    favored for 'Hard'."""
    entities = [c for c in candidates if c.label.startswith("ENTITY")]
    chunks = [c for c in candidates if c.label == "NOUN_CHUNK"]

    if difficulty == "Easy":
        ordered = entities + chunks
    elif difficulty == "Hard":
        chunks.sort(key=lambda c: len(c.text.split()), reverse=True)
        ordered = chunks + entities
    else:  # Medium
        ordered = entities + chunks
        random.shuffle(ordered)

    return ordered


# --------------------------------------------------------------------------
# Answer-aware question generation (T5)
# --------------------------------------------------------------------------

_QUESTION_PREFIX_RE = re.compile(r"^\s*question\s*:\s*", re.IGNORECASE)


def _generate_question_for_answer(sentence: str, answer: str) -> Optional[str]:
    """Use the T5 QG model to turn (sentence, answer) into a natural
    question. Returns None if generation fails or looks degenerate."""
    tokenizer, model = _get_qg_model()

    if answer not in sentence:
        return None

    if QG_FORMAT == "highlight":
        highlighted = sentence.replace(answer, f"<hl> {answer} <hl>", 1)
        input_text = f"generate question: {highlighted} </s>"
    else:  # "answer_context"
        input_text = f"answer: {answer} context: {sentence} </s>"

    try:
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=256
        )
        output_ids = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        question = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception:
        return None

    if not question:
        return None

    # The answer_context model prefixes its raw output with "question: ".
    question = _QUESTION_PREFIX_RE.sub("", question).strip()
    if not question:
        return None

    question = question[0].upper() + question[1:]
    if not question.endswith("?"):
        question += "?"

    if len(question.split()) < 4:
        return None

    # Reject questions that just restate the answer verbatim
    if answer.lower() in question.lower() and len(answer.split()) > 2:
        return None

    return question


# --------------------------------------------------------------------------
# Distractor generation (for Multiple Choice)
# --------------------------------------------------------------------------

def _wordnet_distractors(answer: str, n: int) -> List[str]:
    """Pull plausible-but-wrong options from WordNet co-hyponyms."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    words = answer.split()
    if len(words) > 2:
        return []  # WordNet distractors work best for single concepts

    synsets = wn.synsets(answer.replace(" ", "_"))
    if not synsets:
        return []

    distractors = set()
    for syn in synsets[:3]:
        for hyper in syn.hypernyms():
            for hyponym in hyper.hyponyms():
                lemma = hyponym.lemmas()[0].name().replace("_", " ")
                if lemma.lower() != answer.lower():
                    distractors.add(lemma)
        if len(distractors) >= n:
            break

    return list(distractors)[:n]


def _pool_distractors(
    answer: str,
    label: str,
    all_candidates: List[Candidate],
    n: int,
) -> List[str]:
    """Prefer distractors of the *same* entity type pulled from elsewhere
    in the document, then WordNet, then any other noun chunk."""
    same_type = [
        c.text for c in all_candidates
        if c.label == label and c.text.lower() != answer.lower()
    ]
    random.shuffle(same_type)

    distractors: List[str] = []
    for text in same_type:
        if text.lower() not in {d.lower() for d in distractors}:
            distractors.append(text)
        if len(distractors) >= n:
            return distractors

    for text in _wordnet_distractors(answer, n - len(distractors)):
        if text.lower() not in {d.lower() for d in distractors} and text.lower() != answer.lower():
            distractors.append(text)
        if len(distractors) >= n:
            return distractors

    others = [
        c.text for c in all_candidates
        if c.text.lower() != answer.lower() and c.text.lower() not in {d.lower() for d in distractors}
    ]
    random.shuffle(others)
    for text in others:
        distractors.append(text)
        if len(distractors) >= n:
            break

    return distractors[:n]


# --------------------------------------------------------------------------
# True / False construction
# --------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    (r"\bis\b", "is not"),
    (r"\bare\b", "are not"),
    (r"\bwas\b", "was not"),
    (r"\bwere\b", "were not"),
    (r"\bcan\b", "cannot"),
    (r"\bdoes\b", "does not"),
    (r"\bdo\b", "do not"),
    (r"\bhas\b", "does not have"),
    (r"\bhave\b", "do not have"),
]


def _negate_sentence(sentence: str) -> Optional[str]:
    for pattern, replacement in _NEGATION_PATTERNS:
        if re.search(pattern, sentence):
            return re.sub(pattern, replacement, sentence, count=1)
    return None


def _make_true_false(sentence: str, all_candidates: List[Candidate]) -> Optional[Question]:
    make_false = random.random() < 0.5

    if not make_false:
        return Question(q_type="True/False", question=sentence.rstrip("."), answer="True")

    sentence_candidates = [c for c in _extract_candidates(sentence) if c.label.startswith("ENTITY")]
    random.shuffle(sentence_candidates)
    for cand in sentence_candidates:
        swap_options = _pool_distractors(cand.text, cand.label, all_candidates, 1)
        if swap_options:
            false_sentence = sentence.replace(cand.text, swap_options[0], 1)
            if false_sentence != sentence:
                return Question(q_type="True/False", question=false_sentence.rstrip("."), answer="False")

    negated = _negate_sentence(sentence)
    if negated:
        return Question(q_type="True/False", question=negated.rstrip("."), answer="False")

    return Question(q_type="True/False", question=sentence.rstrip("."), answer="True")


# --------------------------------------------------------------------------
# Fill in the blank construction
# --------------------------------------------------------------------------

def _make_fill_blank(sentence: str, candidate: Candidate) -> Optional[Question]:
    if candidate.text not in sentence:
        return None
    blanked = sentence.replace(candidate.text, "_____", 1)
    if blanked == sentence:
        return None
    return Question(q_type="Fill in the Blank", question=blanked, answer=candidate.text)


# --------------------------------------------------------------------------
# Short answer construction
# --------------------------------------------------------------------------

_DEFINITION_RE = re.compile(
    r"^(?P<subject>.{2,60}?)\s+(?:is|are|refers to|means)\s+(?P<predicate>.{5,200})$",
    re.IGNORECASE,
)


def _make_short_answer(sentence: str, candidate: Candidate) -> Optional[Question]:
    question_text = _generate_question_for_answer(sentence, candidate.text)
    if not question_text:
        return None

    match = _DEFINITION_RE.match(sentence.strip().rstrip("."))
    if match and candidate.text.lower() in match.group("subject").lower():
        answer = match.group("predicate").strip()
    else:
        answer = candidate.text

    if answer.lower() == question_text.lower():
        return None

    return Question(q_type="Short Answer", question=question_text, answer=answer)


# --------------------------------------------------------------------------
# Multiple choice construction
# --------------------------------------------------------------------------

def _make_multiple_choice(
    sentence: str, candidate: Candidate, all_candidates: List[Candidate]
) -> Optional[Question]:
    question_text = _generate_question_for_answer(sentence, candidate.text)
    if not question_text:
        return None

    distractors = _pool_distractors(candidate.text, candidate.label, all_candidates, 3)
    if len(distractors) < 3:
        return None

    options = distractors[:3] + [candidate.text]
    random.shuffle(options)

    return Question(
        q_type="Multiple Choice",
        question=question_text,
        answer=candidate.text,
        options=options,
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate(q: Question) -> bool:
    if not q.question or not q.answer:
        return False
    if len(q.question.strip()) < 5:
        return False
    if q.answer.strip().lower() == q.question.strip().lower():
        return False
    if not re.search(r"[a-zA-Z]", q.question):
        return False

    if q.q_type == "Multiple Choice":
        if not q.options or len(q.options) != 4:
            return False
        if len(set(o.lower().strip() for o in q.options)) != 4:
            return False
        if q.answer.lower().strip() not in [o.lower().strip() for o in q.options]:
            return False

    if q.q_type == "Fill in the Blank" and "_____" not in q.question:
        return False

    if q.q_type == "True/False" and q.answer not in ("True", "False"):
        return False

    return True


def _dedupe(questions: List[Question]) -> List[Question]:
    seen = set()
    unique = []
    for q in questions:
        key = q.question.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

class QuestionGenerator:
    """Thin, cache-friendly wrapper around the module-level pipeline.
    Instantiate once and reuse; call `.load()` once (ideally behind
    st.cache_resource) before generating."""

    last_shortfall: Optional[tuple] = None  # (num_generated, num_requested)

    def load(self) -> "QuestionGenerator":
        preload_models()
        return self

    def generate_questions(
        self,
        text: str,
        question_type: str,
        difficulty: str,
        num_questions: int,
    ) -> List[dict]:
        if not text or not text.strip():
            raise QuestionGenerationError("No content was provided.")

        sentences = [s for s in split_sentences(text) if _is_usable_sentence(s)]
        if len(sentences) < 2:
            raise QuestionGenerationError(
                "The provided content is too short to generate meaningful "
                "questions. Please provide more educational material."
            )

        all_candidates: List[Candidate] = []
        per_sentence_candidates = {}
        for sentence in sentences:
            cands = _extract_candidates(sentence)
            per_sentence_candidates[sentence] = cands
            all_candidates.extend(cands)

        results: List[Question] = []
        max_attempts = max(60, num_questions * 15)
        attempts = 0

        if question_type == "True/False":
            pool = sentences * 3
            random.shuffle(pool)

            for sentence in pool:
                if len(results) >= num_questions or attempts >= max_attempts:
                    break
                attempts += 1
                q = _make_true_false(sentence, all_candidates)
                if q is not None:
                    q.difficulty = difficulty
                    if _validate(q):
                        results.append(q)
                results = _dedupe(results)

        else:
            pairs: List[tuple] = []
            for sentence in sentences:
                ranked = _rank_candidates(per_sentence_candidates.get(sentence, []), difficulty)
                for candidate in ranked:
                    pairs.append((sentence, candidate))
            random.shuffle(pairs)

            for sentence, candidate in pairs:
                if len(results) >= num_questions or attempts >= max_attempts:
                    break
                attempts += 1

                q: Optional[Question] = None
                if question_type == "Multiple Choice":
                    q = _make_multiple_choice(sentence, candidate, all_candidates)
                elif question_type == "Short Answer":
                    q = _make_short_answer(sentence, candidate)
                elif question_type == "Fill in the Blank":
                    q = _make_fill_blank(sentence, candidate)
                else:
                    raise QuestionGenerationError(f"Unknown question type: {question_type}")

                if q is not None:
                    q.difficulty = difficulty
                    if _validate(q):
                        results.append(q)
                        results = _dedupe(results)

        results = results[:num_questions]

        if not results:
            raise QuestionGenerationError(
                "No valid questions could be generated from this content. "
                "Try providing longer or more detailed educational material."
            )

        self.last_shortfall = (len(results), num_questions) if len(results) < num_questions else None

        return [q.to_dict() for q in results]

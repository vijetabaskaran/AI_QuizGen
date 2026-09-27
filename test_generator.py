"""
test_generator.py

Standalone smoke test for the question-generation pipeline, independent
of Streamlit. Useful for quickly checking that models load and produce
sane output after an environment change — and for pre-downloading the
models with a visible progress bar before ever touching the Streamlit UI.

Run with:
    python test_generator.py
"""

from utils.question_generator import QuestionGenerator, QuestionGenerationError

SAMPLE_TEXT = """
Photosynthesis is the process by which green plants convert light energy
into chemical energy. Chlorophyll is the green pigment found in
chloroplasts that absorbs sunlight. Plants use carbon dioxide and water,
along with sunlight, to produce glucose and oxygen. This process mostly
takes place in the leaves of the plant. Oxygen is released into the
atmosphere as a byproduct of photosynthesis, while glucose is used by
the plant as a source of energy for growth.
"""

QUESTION_TYPES = ["Multiple Choice", "Short Answer", "Fill in the Blank", "True/False"]


def main() -> None:
    print("Loading models (this may take a few minutes on the very first run)...")
    generator = QuestionGenerator().load()
    print("Models loaded.\n")

    for q_type in QUESTION_TYPES:
        print(f"--- {q_type} ---")
        try:
            questions = generator.generate_questions(
                text=SAMPLE_TEXT,
                question_type=q_type,
                difficulty="Medium",
                num_questions=3,
            )
        except QuestionGenerationError as e:
            print(f"  Generation error: {e}")
            continue

        for i, q in enumerate(questions, start=1):
            print(f"  Q{i}: {q['question']}")
            if q.get("options"):
                for letter, opt in zip("ABCD", q["options"]):
                    marker = " <-- correct" if opt == q["answer"] else ""
                    print(f"       {letter}. {opt}{marker}")
            else:
                print(f"       Answer: {q['answer']}")
        print()


if __name__ == "__main__":
    main()

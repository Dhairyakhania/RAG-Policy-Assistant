EVAL_QUESTIONS = [
    ("How long does a refund take?", "Answerable"),
    ("Can I cancel after shipping?", "Partial"),
    ("Do you ship internationally?", "Answerable"),
    ("Is there a student discount?", "Unanswerable"),
    ("Who is the CEO?", "Out of Scope"),
]

def print_eval_questions():
    for q, t in EVAL_QUESTIONS:
        print(f"- {q} ({t})")

# Prompt Engineering & RAG Mini Project

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) question-answering assistant**
over company policy documents.  
The primary focus is **prompt engineering, hallucination control, retrieval quality, and evaluation** —
not UI or scale.

The system retrieves relevant policy text, generates **grounded answers only**, and
**explicitly refuses** to answer questions that are not supported by the documents.

---

## Objectives

The project demonstrates the ability to:

- Design **clear, strict prompts** for controlled LLM behavior
- Build a **correct RAG pipeline**
- Prevent hallucinations using **multiple guardrails**
- Evaluate and reason about LLM outputs
- Make practical engineering trade-offs

---

## Data Source

The knowledge base consists of **publicly available Amazon.in policy documents**, including:

- Return Policy
- Refund Policy
- Cancellation Policy
- Delivery Options
- Customer Service FAQs

These documents are used **strictly for educational and demonstration purposes**.

---

## Architecture

User Query
↓
Embedding Model (Sentence Transformers)
↓
FAISS Vector Store
↓
Retriever (Top-K)
↓
Confidence-Based Gating
↓
Strict Prompt + Context
↓
Local LLM (Mistral via Ollama)
↓
Validated Output → User-Friendly Answer


---

## Models & Tools

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **LLM**: Mistral (local inference via Ollama)
- **Framework**: LangChain (modern LCEL-based chains)
- **Language**: Python

No proprietary APIs or API keys are used.

---

## Data Preparation

### Chunking Strategy

- **Chunk size**: 500 tokens  
- **Overlap**: 80 tokens  

**Rationale**:
Policy documents contain dense clauses and exceptions.  
This chunk size preserves semantic completeness while minimizing retrieval noise
and hallucination risk.

---

## RAG Pipeline

1. Documents are embedded using SentenceTransformers
2. Embeddings are stored in FAISS
3. Top-K chunks are retrieved per query
4. Retrieval confidence is computed
5. Weak retrieval → explicit refusal
6. Strong retrieval → context passed to LLM

---

## Prompt Engineering

### Prompt Version 1 (Initial)

- Free-form answers
- Occasional hallucinations
- Weak refusal behavior

### Prompt Version 2 (Final)

Key improvements:
- Explicit grounding rules
- Hard refusal enforcement
- No external knowledge allowed
- Structured internal output
- Forbids yes/no or single-word answers

**Result**:
- Zero hallucinations for out-of-scope queries
- Consistent, explainable answers

---

## Hallucination Control (Key Highlight)

The system enforces **four independent guardrails**:

1. **Hard Refusal Enforcement**  
   If the answer is not explicitly present, the system outputs exactly:  
   > “The provided documents do not contain this information.”

2. **Confidence-Based Retrieval Gating**  
   Weak semantic similarity → automatic refusal

3. **Source-Aligned Answers**  
   Every answer is grounded in retrieved policy documents

4. **Post-Generation Validation**  
   - Rejects yes/no answers  
   - Rejects malformed outputs  
   - Rejects unsupported claims  

This layered approach mirrors real-world production RAG systems.

---

## Output Design

- **Internal Output**: Structured JSON (for validation and control)
- **User-Facing Output**: Clean, readable text with:
  - Policy-grounded explanation
  - Source document names
  - Confidence score

This separation balances **engineering robustness** with **user experience**.

---

## Evaluation

A small evaluation set was manually created, covering:

- Fully answerable questions
- Partially answerable questions
- Unanswerable questions
- Out-of-scope questions

### Evaluation Criteria

- Accuracy
- Hallucination avoidance
- Answer clarity

### Sample Results

| Question | Expected | Result |
|--------|---------|--------|
| Can I cancel an order after shipping? | Partial | ✅ |
| Is same-day delivery available in Pune? | Answerable | ✅ |
| Are student discounts available? | Unanswerable | ✅ Proper refusal |
| Who is the CEO of Amazon? | Out of scope | ✅ Proper refusal |

---

## Edge Case Handling

- **No relevant documents found** → Hard refusal
- **Weak retrieval confidence** → Hard refusal
- **Out-of-scope questions** → Hard refusal

The system prefers **no answer over an incorrect answer**.

---

## Optional Bonus Coverage

| Bonus Item | Status |
|---------|--------|
| Prompt templating (LangChain) | ✅ |
| Simple reranking / gating | ✅ |
| Output schema validation | ✅ |
| Prompt version comparison | ✅ |
| Logging / tracing | ❌ (intentionally omitted) |

---

## Trade-offs & Design Decisions

- Chose **local LLM** over cloud APIs for reproducibility
- Prioritized **hallucination control** over verbosity
- Avoided UI and agents to keep scope focused
- Chose simplicity over over-engineering

---

## How to Run

```bash
pip install -r requirements.txt
ollama pull mistral
python main.py
```

## Optional Streamlit UI

A minimal Streamlit interface is included for interactive exploration.
The UI is intentionally lightweight and sits on top of the same RAG pipeline
used by the CLI.

To run:
```bash
streamlit run streamlit_app.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was developed as part of a prompt engineering and
retrieval-augmented generation (RAG) exercise.

The implementation leverages open-source tools and libraries, including
LangChain, SentenceTransformers, FAISS, and Ollama, which made it possible
to build and evaluate a fully local, reproducible RAG pipeline.

Publicly available Amazon.in policy documents were used strictly for
educational and demonstration purposes.

The project design emphasizes reasoning quality, hallucination control,
and practical system thinking inspired by real-world GenAI applications.
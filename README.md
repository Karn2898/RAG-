# RAG-: Technical Documentation

This repository implements a notebook-driven Retrieval-Augmented Generation (RAG) system for PDF question answering. The pipeline uses LangChain orchestration, Chroma vector indexing, NVIDIA embedding endpoints, and an NVIDIA chat model for grounded response generation.

## 1. System Architecture

End-to-end flow:

1. Ingestion
	 - PDF is loaded with `PyPDFLoader` into LangChain `Document` objects.
2. Segmentation
	 - `RecursiveCharacterTextSplitter` creates fixed-size chunks for retrieval.
3. Embedding
	 - Chunk text is embedded via `NVIDIAEmbeddings(model="nvidia/nv-embed-v1")`.
4. Indexing
	 - Embedded chunks are persisted in a Chroma vector store.
5. Retrieval
	 - Similarity search retrieves top-$k$ chunks for a user query.
6. Generation
	 - Retrieved context + user input are passed to `ChatNVIDIA` through a retrieval chain.
7. Response
	 - Model outputs concise answers constrained by the system prompt.

## 2. Repository Layout

- `notebook.ipynb`
	- Primary implementation of the full RAG flow.
- `Requirements.txt`
	- Python dependency list.
- `He_Deep_Residual_Learning_CVPR_2016_paper.pdf`
	- Input corpus used in the notebook.
- `nvidia.py`
	- Local environment helper file. Do not store real credentials in source files.

## 3. Core Components

### 3.1 Loader

- Class: `langchain_community.document_loaders.PyPDFLoader`
- Input: absolute path to the PDF
- Output: list of documents with per-page content and metadata

### 3.2 Splitter

- Class: `langchain_text_splitters.RecursiveCharacterTextSplitter`
- Current configuration: `chunk_size=1000`
- Effect:
	- Larger chunks: better context continuity, higher token overhead
	- Smaller chunks: higher recall granularity, potential context fragmentation

### 3.3 Embedding Model

- Class: `langchain_nvidia_ai_endpoints.NVIDIAEmbeddings`
- Model: `nvidia/nv-embed-v1`
- Usage:
	- Query embeddings and document embeddings share the same vector space
	- API key is read from `NVIDIA_API_KEY` (env) or hidden runtime prompt

### 3.4 Vector Store

- Backend: `langchain_chroma.Chroma`
- Initialization: `Chroma.from_documents(documents=docs, embedding=embeddings)`
- Retrieval mode: similarity search
- Current retriever config: `k=10`

### 3.5 Generator Model

- Class: `langchain_nvidia_ai_endpoints.ChatNVIDIA`
- Model: `meta/llama-3.1-8b-instruct`
- Current parameters:
	- `temperature=0.3`
	- `max_tokens=500`

### 3.6 Chain Composition

- `create_stuff_documents_chain(llm, prompt)`
	- Concatenates retrieved chunks into prompt context.
- `create_retrieval_chain(retriever, question_answer_chain)`
	- Wraps retrieval and answer generation in a single invocation path.

## 4. Prompting Strategy

The system prompt enforces:

- grounded answering from retrieved context,
- explicit fallback when uncertain,
- concise output (max three sentences).

This reduces hallucination risk and keeps outputs deterministic for notebook experiments.

## 5. Environment and Execution

### 5.1 Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Requirements.txt
```

### 5.2 Authentication

Preferred:

```bash
export NVIDIA_API_KEY="<your_key>"
```

Alternative: place key in `.env` and load via `python-dotenv`.

### 5.3 Notebook Execution Order

Run cells sequentially to preserve stateful objects:

1. Loader initialization and PDF load
2. Document inspection
3. Splitter execution
4. Embedding client initialization
5. Chroma index creation
6. Retriever invocation tests
7. Chat model initialization
8. Retrieval chain execution with interactive query

## 6. Retrieval and Quality Tuning

Primary tuning levers:

- `chunk_size`
	- Increase for richer local context
	- Decrease for more precise chunk targeting
- `k` (top-k retrieved chunks)
	- Increase for recall
	- Decrease for precision and lower prompt size
- Prompt strictness
	- Stronger "use only context" instructions reduce hallucinations
- Generation params
	- Lower `temperature` for stability
	- Adjust `max_tokens` based on desired answer length

## 7. Operational Risks and Mitigations

- Secret leakage risk
	- Do not commit API keys to tracked files.
	- Rotate exposed keys immediately.
- Dependency drift
	- Pin package versions if reproducibility is required.
- Notebook state coupling
	- Re-run from top after dependency installs or key changes.

## 8. Troubleshooting

- `ModuleNotFoundError`
	- Reinstall dependencies in active environment:
		- `pip install -r Requirements.txt`
- Authentication failure / 401
	- Confirm `NVIDIA_API_KEY` exists in current notebook kernel environment.
- Low-quality or irrelevant answers
	- Adjust `chunk_size`, retriever `k`, and query phrasing.
- Runtime object not defined
	- Execute prior cells to rebuild notebook state.

## 9. Future Engineering Improvements

- Add source citation output (document page references)
- Persist Chroma index to disk for faster cold start
- Add evaluation harness (retrieval precision and answer faithfulness)
- Replace interactive input with function/API interface for automation

## License

See `LICENSE`.
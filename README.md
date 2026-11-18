# Graph RAG for Spatial Neglect Literature

A retrieval-augmented generation (RAG) system that answers clinical questions about **unilateral spatial neglect** (assessment, rehabilitation, VR / eye-tracking, prism adaptation, etc.) from a collection of research papers.

Instead of a plain vector store, the papers are turned into a **knowledge graph in Neo4j**: biomedical entities are extracted with scispaCy, linked by typed relations (`TREATS`, `ASSESSED_BY`, `SHOWED_IMPROVEMENT_IN`, ...), and connected back to the text chunks they came from. At query time the graph is traversed, candidates are re-ranked with BM25, and a local LLM served by **Ollama** (Mistral by default) writes the final answer.

## How it works

```
PDFs ──► text cleaning ──► sentence chunking ──► NER + relation extraction ──► Neo4j graph
                                                                                    │
Question ──► NER ──► fulltext entity match ──► graph traversal + chunk retrieval ◄───┘
                                                        │
                                              BM25 re-ranking
                                                        │
                                           Ollama LLM answer
```

**Ingestion** ([src/ingest.py](src/ingest.py))
1. PyMuPDF reads each PDF in `pdfs/`, handling two-column layouts, dropping headers/footers and journal boilerplate, and cutting off the reference section.
2. The text is cleaned: broken ligatures (`bene fi cial` → `beneficial`), hyphenation remnants, and inline citation numbers are fixed, while real numbers (sample sizes, p-values, years) are kept. Figure, table and box captions are re-inserted next to the paragraph that mentions them.
3. The text is split into sentence-aligned chunks (`CHUNK_SIZE` characters, 1-sentence overlap).
4. For each chunk ([src/ner.py](src/ner.py)):
   - scispaCy (`en_core_sci_sm`) plus a custom entity ruler pull out entities, which are normalized (e.g. `VR` → `virtual reality`) and filtered for generic words.
   - Keyword rules turn sentences into `(subject, RELATION, object)` triples, including negation handling (`DOES_NOT_...`, `NO_DIFFERENCE_IN`).
5. Everything is written to Neo4j ([src/graph_db.py](src/graph_db.py)):
   - `(:Entity)-[:RELATION]->(:Entity)`, where each relation stores its source chunk text, chunk ids and file names.
   - `(:Entity)-[:MENTIONED_IN]->(:Chunk)`
   - Fulltext indexes on entity names (`entityIndex`) and chunk text (`chunkTextIndex`).

**Question answering** ([src/main.py](src/main.py))
1. Entities are extracted from the question and matched against `entityIndex`.
2. The best matches become seed nodes. From them the system collects co-occurring chunks, directly linked chunks, fulltext chunk hits, and facts from subgraph traversal.
3. Snippets (top 8) and graph facts (top 15) are re-ranked with BM25 against the question.
4. [src/llm.py](src/llm.py) builds a grounded prompt, calls Ollama with temperature 0, and strips any citations from the answer.
5. Latency and token metrics are returned with every answer.

## Repository layout

| Path | Purpose |
|---|---|
| `pdfs/` | Source papers to ingest (every `*.pdf` in here is processed). The papers are not included in the repo; add your own. |
| `test_set.csv` | Evaluation set: `question`, `reference_answer` |
| `Dockerfile`, `docker-compose.yml` | Neo4j + Python app containers |
| `requirements.txt` | Python dependencies (CPU PyTorch) |
| `.env.example` | Template for the `.env` file the app container loads |
| `src/config.py` | All settings, read from environment variables |
| `src/ingest.py` | PDF → cleaned text → chunks → graph |
| `src/ner.py` | Entity extraction, normalization, relation classification |
| `src/graph_db.py` | Neo4j schema, writes, and retrieval queries |
| `src/llm.py` | Ollama client and answer prompt |
| `src/main.py` | Retrieval pipeline and interactive Q&A loop |
| `src/reset.py` | Deletes every node and relationship in the graph |
| `src/evaluate.py` | `RAGEvaluator`: cosine similarity and BERTScore |
| `src/run_evaluation.py` | Runs the test set through the pipeline and scores it |
| `src/run_evaluation_drive.py` | Template for running the same evaluation on another (embedding-based) RAG system; fill in the `embed` / `retrieve` placeholders |
| `src/evaluation_results.*`, `src/evaluation_summary.json` | Output of the latest evaluation run |
| `src/pdf_to_text.txt` | Debug dump of the last PDF's cleaned text, written during ingestion |

## Getting started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose
- [Ollama](https://ollama.com/) installed **on the host**. The Ollama container in `docker-compose.yml` is commented out, so the app reaches the host's Ollama through `host.docker.internal`.

### 1. Pull the LLM

```bash
ollama pull mistral
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Then edit `.env`. Inside the Docker network the app needs to reach Neo4j by its service name and Ollama on the host:

| Variable | Default (in `config.py`) | Typical value under Docker |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | `bolt://neo4j:7687` |
| `NEO4J_USER` | `neo4j` | `neo4j` |
| `NEO4J_PASSWORD` | demo value in `config.py` | must match `NEO4J_AUTH` in `docker-compose.yml` |
| `OLLAMA_URL` | `http://localhost:11434` | `http://host.docker.internal:11434` |
| `OLLAMA_MODEL` | `mistral` | any model you have pulled |
| `OLLAMA_NUM_CTX` | `8192` | prompts reach ~3k tokens; Ollama's default context is too small |
| `PDF_DIR` | `../pdfs` | `/app/pdfs` |
| `CHUNK_SIZE` | `1000` | characters per chunk |
| `SPACY_MODEL` | `en_core_sci_sm` | installed in the Dockerfile |

`.env` is listed in `.gitignore`, so it stays out of commits.

### 3. Start the containers

```bash
docker compose up -d --build
```

This starts `graphrag-neo4j` (Neo4j 5 with APOC) and `graphrag-app`. The project root is mounted into the app container at `/app`, so evaluation output written there appears in your working tree. HuggingFace models are cached in a named volume.

### 4. Build the graph

Put the papers you want to index in `pdfs/`. The repo doesn't include them, and `pdfs/*.pdf` is git-ignored.

```bash
# optional: clear a previous graph first
docker exec -it graphrag-app python reset.py

docker exec -it graphrag-app python ingest.py
```

You can browse the graph at <http://localhost:7474>.

### 5. Ask questions

```bash
docker exec -it graphrag-app python main.py
```

Type a question at the `Q:` prompt (`exit` to quit). Each answer shows the extracted entities, the matched graph nodes, and a latency/token breakdown.

## Evaluation

```bash
docker exec -it graphrag-app python run_evaluation.py
```

The script answers every question in `test_set.csv` and compares each answer to its reference:

- **Semantic similarity**: cosine similarity of `all-MiniLM-L6-v2` sentence embeddings
- **BERTScore**: precision, recall, F1
- **Performance**: NER, Neo4j and LLM latency; prompt and generation tokens/sec

Per-question results, including the retrieved facts and snippets, go to `src/evaluation_results.csv`. Averages go to `src/evaluation_summary.json`.

Latest run (15 questions, Mistral via Ollama):

| Metric | Value |
|---|---|
| Mean cosine similarity | 0.838 |
| Mean BERTScore F1 | 0.896 (P 0.887 / R 0.906) |
| Mean Neo4j retrieval latency | 0.27 s |
| Mean LLM latency | 43.5 s |
| Mean total latency | 43.9 s |

Almost all of the latency comes from LLM generation; graph retrieval takes well under a second.

## Notes and limitations

- Relation extraction is rule-based (keyword patterns per sentence), so recall is limited to the phrasings listed in `RELATION_KEYWORDS` in `src/ner.py`. Chunk-level retrieval makes up for relations it misses.
- PDF cleaning has heuristics tuned to the journals in `pdfs/` (for example BMJ *Practical Neurology* footers). New sources may need extra junk patterns in `load_pdf_text`.
- Ingestion is not idempotent for chunks from changed PDFs. Run `reset.py` before re-ingesting a modified corpus.

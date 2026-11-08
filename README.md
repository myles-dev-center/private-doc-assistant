# Private Document Assistant — built for one kind of document, measured on real ones

Ask questions about a pile of business documents and get **answers with citations**, **tables you can export to Excel**, and **a measured accuracy report**: how often the answers are right, and how often the system makes something up.

**There is no general-purpose RAG that is accurate on everything.** Every document collection has its own identifiers, its own boilerplate and its own "list all … that …" questions, and a pipeline that ignores them fails in ways a demo never shows. This project is the method I use on client work: a small, inspectable pipeline that is **configured for one domain** (here: commercial contracts), and **an evaluation that proves on documents it has never seen** whether that configuration works — and what it costs compared with simply handing the documents to a large model.

Runs **on-premises with a local model** (LM Studio / Ollama / vLLM — no document leaves the machine) or with any OpenAI-compatible cloud API. Switching is a config change.

![Asking across 30 real agreements: the answer comes from the extracted table, with the quote behind every value; one click opens the page](docs/img/demo-ask.gif)

*Local 9B model on a laptop, 30 real agreements. "List all …" questions are answered from a table extracted at ingestion: the answer shows the matching rows, the SQL, and the sentence and page each value was taken from. Questions about one agreement cite the document and page.* **[▶ 76-second demo with narration](docs/demo.mp4)**: adding a contract, asking across all of them, asking about one, a question the documents can't answer, and checking the extracted table.

## Eight ways to answer questions about real contracts

**90 commercial agreements from public SEC filings** ([CUAD](https://www.atticusprojectai.org/cuad), labeled by lawyers) in three blind sets of 30, **never used while building the system**, **159 questions**: 96 about one named agreement (governing law, dates, renewal notice — including questions that use only a short name, "our agreement with JOHNSON"), 39 across all agreements ("which agreements contain a non-compete clause?", "list all agreements governed by the same law as …"), 24 whose answer is not in the documents. Every method gets the same documents, the same questions and the same deterministic scorer, and was run **once**.

| Method | Accuracy (95% CI) | One agreement | Across all | Says "not found" | Hallucination | Median time / question | Model calls / question | Cost / question | One-off ingestion, 30 agreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Local · this pipeline** | **80%** (73–85%) | **96%** | **36%** | **88%** | **17%** | 14 s | 1.4 | hardware only | 19 min |
| Local · model reads every document | not run ¹ | | | | | | | | |
| Local · agent with search tools | 72% (64–78%) | 88% | 23% | 88% | 27% | 32 s | 6.5 | hardware only | — |
| Local · Codex CLI on the document folder | not run ² | | | | | | | | |
| **Cloud · this pipeline** | **89%** (83–93%) | **99%** | **59%** | **96%** | **11%** | **4 s** | 1.3 | **$0.010** | $0.69, 8 min |
| Cloud · all documents in the prompt | 87% (81–91%) | 98% | 54% | 96% | 12% | 9 s | 1 | $0.375 | — |
| Cloud · agent with search tools | 87% (81–91%) | 97% | 59% | 92% | 12% | 15 s | 4.9 | $0.022 | — |
| Cloud · [Codex CLI](https://github.com/openai/codex) on the document folder | 88% (82–92%) | 98% | 59% | 96% | 11% | 26 s | 3.3 commands | $0.040 | — |

**Local** = `qwen3.5-9b` (chat) + `nomic-embed-text-v1.5` (embeddings) in LM Studio, 32k context. **Cloud** = `gpt-6.1-sol` (chat, reasoning effort `low`) + `text-embedding-3-small` (embeddings). Cloud cost is the list price of the tokens actually used ($2 / $0.10 cached / $10 per million input / cached / output tokens; prompts over 272k tokens cost 2× input and 1.5× output). The interval is a 95% Wilson interval over the 159 questions. Results: [`evals/cmp4/`](evals/cmp4/) (pipeline) and [`evals/cmp4b/`](evals/cmp4b/) (other methods); the table is printed by [`scripts/blind_report.py`](scripts/blind_report.py).

¹ **Not run: too slow on the local model.** A 9B model cannot hold 30 agreements at once, so it has to read every agreement in parts for every question: 552 s per question in an earlier 44-question run ([`evals/cmp/`](evals/cmp/)), about **24 hours** for these 159. On that run it scored 77%, with 10% on "across all" questions.
² **Not run: a coding agent needs a strong model.** Codex plans its own searches and writes its own shell and Python; the 9B model already searches badly with a far simpler agent (row 3), and the run would take an estimated 5–13 hours.
Both rows can be run with the same scripts ([`scripts/run_fullread.py`](scripts/run_fullread.py) `--mode map`, [`scripts/run_codex.py`](scripts/run_codex.py) with a local provider) where a stronger local model is available — on a client's GPU server, for example.

**Agent with search tools** is a minimal tool-calling loop (list files, grep, read a page). **Codex CLI** is OpenAI's production coding agent (v0.160.0, same model and reasoning effort), given the agreements as text files and nothing else; every question is a fresh session ([`scripts/run_codex.py`](scripts/run_codex.py)). Codex reports tokens per turn, not per model call, so its cost leaves out the long-prompt surcharge — a lower bound.

What the table shows:

- **With a strong model, every method is about equally accurate — the difference is cost and waiting time.** The cloud pipeline, all-documents-in-the-prompt, the simple agent and Codex all score 87–89%; none of the differences with the pipeline is statistically significant (paired McNemar test, p ≥ 0.25). The pipeline answers in **4 s for $0.010**; Codex takes 26 s (65 s at the 90th percentile) and 4× the cost, reading ~65k tokens per question; all documents in the prompt cost **38×** as much. The pipeline's one-off ingestion ($0.69 for 30 agreements) pays for itself against Codex after about 23 questions.
- **With a small local model, structure matters.** The same 9B model scores 80% in this pipeline and 72% as an agent (p = 0.02), with a hallucination rate of 17% against 27%.
- **A 9B model is not a frontier model.** On the same pipeline, the cloud model is clearly better (89% against 80%, p = 0.001). Most of the gap is "across all" questions (59% against 36%): asked for "all agreements governed by the same law as X", the local model writes SQL that finds X and nothing else (0 of 9; the cloud model 9 of 9) — although the extracted table was the same for both. Local is the choice when documents may not leave the building, not the choice for the best answers.
- **Codex is a good tool for an analyst, not for a team.** It matches the pipeline's accuracy by scanning all 30 files with its own `grep` and Python for every question, which needs a shell, a sandbox and a developer to run it — not something a team of non-developers can ask questions of all day.
- **The chat model matters, the embedding model barely does.** See the next table.

### Mixing local and cloud models

The pipeline's chat model and embedding model can be chosen separately — for example, embeddings computed locally so the document text is only sent out for answering, or a cloud embedding model with a local chat model. All four combinations on the same 159 questions ([`scripts/combos_report.py`](scripts/combos_report.py)):

| Chat model + embedding model | Accuracy (95% CI) | Short name ("our agreement with JOHNSON") | "Same law as …" lists | Hallucination | Median time / question | Cost / question | Ingestion, 30 agreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cloud `gpt-6.1-sol` + cloud `text-embedding-3-small` | **89%** (83–93%) | 18/18 | 9/9 | 11% | 4 s | $0.010 | $0.69 |
| Cloud `gpt-6.1-sol` + local `nomic-embed-text` | 87% (81–92%) | 15/18 | 9/9 | 10% | 4 s | $0.014 | $0.68 |
| Local `qwen3.5-9b` + cloud `text-embedding-3-small` | 80% (73–85%) | 16/18 | 0/9 | 17% | 13 s | hardware only | < $0.01 (embeddings) |
| Local `qwen3.5-9b` + local `nomic-embed-text` | 80% (73–85%) | 16/18 | 0/9 | 17% | 15 s | hardware only | hardware only |

- **Swapping the embedding model changed 0 of 159 results with the local chat model and 4 with the cloud one** (3 better, 1 worse with `text-embedding-3-small`, p = 0.63). Questions that name an agreement read it in full, and "list all" questions go to the table, so vector search is rarely the deciding step on this question set. A question set built on open-ended clause search would depend on it more.
- **Swapping the chat model is what moves accuracy:** cloud over local by 7–9 points with either embedding model (p ≤ 0.012), most of it on "across all" questions.
- **Local embeddings with a cloud chat model** keep the vector index in-house but still send the passages needed for each answer, and the full agreements at ingestion, to the cloud model. It is a cost choice, not a privacy one; only the all-local row keeps every document on the machine.

### What "local" really costs

Local is not free — it moves the cost from per-question fees to hardware and waiting time:

- **Measured on:** a laptop with an Intel Core Ultra 5 338H, 32 GB RAM, integrated Arc GPU, **no dedicated GPU**. The 9B model takes about 8 GB of memory. Ingestion took **about 19 minutes per 30 agreements** (~40 s per agreement, the model reads each one in full); a question takes 14 s at the median and 39 s at the 90th percentile.
- **What a client needs (recommendation, not measured here):** at least 16 GB of free memory for a 9B model. A GPU with 12 GB or more of memory, or a server with one, should make ingestion and questions several times faster. Larger local models (27B and up) need correspondingly more. Hardware, power and setup time are the real cost of the local option.
- **When it is worth it:** when documents may not leave the building, or when question volume is high enough that per-question API fees add up. For a few documents and occasional questions, the cloud configuration is cheaper and faster.

### Extraction accuracy against the lawyers' labels

At ingestion, each agreement is read once and turned into one table row (16 fields, [`schemas/agreement.yaml`](schemas/agreement.yaml)), and every value must come with a quote that is found in the agreement's text — a "yes" without a verifiable quote is stored as unknown. On the 90 blind agreements, the extracted values match CUAD's lawyer labels for **91% of fields with the local model** and **90% with the cloud model** ([`scripts/combos_report.py`](scripts/combos_report.py) `--cuad`):

| Field | Local | Cloud |
|---|---:|---:|
| Governing law | 80/82 | 77/82 |
| Agreement date | 79/83 | 81/83 |
| Non-compete / exclusivity / termination for convenience | 234/270 | 237/270 |
| Cap on liability / audit rights / most favored nation | 250/270 | 239/270 |

## Why it has to be configured for the domain

A general pipeline — split documents into chunks, embed them, retrieve a few, answer — has no idea what a document *is*. On contracts that breaks in predictable ways:

| Question | What a general pipeline does | What this one does instead |
|---|---|---|
| *Which law governs the agreement between X and Y?* | Retrieves "X is a corporation organized under the laws of Delaware" — a sentence that looks right and is about something else. | Recognizes that the question names one agreement and **reads that agreement in full**, together with its extracted record. |
| *What are the payment terms in contract AP-MSA-2022-090?* | Retrieves the payment clause of a *different* contract — templated clauses are identical and never mention the contract number. | Prepends each chunk with the agreement it belongs to, and **searches only the agreement the question names**. |
| *List all agreements governed by the laws of New York.* | Lists whatever is in the six chunks it retrieved. Looks complete, is wrong. | Answers from the **extracted table** (one row per agreement) with a read-only SQL query. |
| *What is the liability cap in the NDA with …?* | NDAs don't have one; the model invents one. | Answers only from the excerpts it was given, cites every claim, and says "not found" otherwise. |

Everything domain-specific lives in **one YAML file**: the fields to extract (with definitions — here taken from the CUAD annotation guide), which fields identify a document, which fields to stamp on every chunk. Moving to invoices, CVs, datasheets or policies means writing that file and a question set from the client's own documents. The pipeline code does not change.

## How it was tested — including the runs that went badly

The pipeline was first built against **30 synthetic contracts** with deliberate traps (identical boilerplate, near-identical contract numbers, a scanned PDF). It reached 100% there — which proves little, because the fixes were written while looking at those questions. So it was tested on **real contracts**, four times, each on agreements never used before:

| Run | What changed before it | Accuracy | Hallucination |
|---|---|---:|---:|
| 1. Real contracts, first time | nothing — schema only | **34%** | 59% |
| 2. 30 new agreements | three bugs fixed (below) | 61% | 18% |
| 3. 30 new agreements | questions that name one agreement are answered from the whole agreement | **86–89%** | 11% |
| 4. 90 new agreements, three blind sets (the comparison above) | nothing (one scoring rule clarified afterwards, see below) | **80–89%** | 11–17% |

Run 1's log: [`evals/results-cuad/run1-before-fixes.log`](evals/results-cuad/run1-before-fixes.log) (its question generator was later tightened to skip redacted or role-only party names, which replaced 4 of those 30 agreements; [`evals/results-cuad/`](evals/results-cuad/) holds the re-check after the fixes, 64%). Runs 2 and 3: [`evals/results-cuad-holdout/`](evals/results-cuad-holdout/) and [`evals/cmp/`](evals/cmp/). Run 4: [`evals/cmp4/`](evals/cmp4/) and [`evals/cmp4b/`](evals/cmp4b/). In run 4, the local model first ran with a 16k context and returned an error on 4 questions whose agreement did not fit; those results were discarded and set 6 was rerun at 32k for both local combinations.

What run 1 exposed — none of it visible on the synthetic set:

- **A malformed model reply crashed the question.** Long party names produced JSON with a broken quote; the answer was an error instead of a fallback. Fix: constrained JSON output (`json_schema` mode), and any failure of the table route falls back to document search.
- **Extraction read only the first 24,000 characters.** Real agreements put governing law near the end. Fix: long documents are read in overlapping parts.
- **Questions about one named agreement went to the table**, which was empty for that field. Fix: a question that names one agreement is answered from that agreement's text.

After run 2, the extracted table already held the right answer for **26 of 26** single-agreement questions, with both the local and the cloud model — but retrieval kept returning the wrong clause. That showed the bottleneck was the design, not the model size, and led to reading the named agreement in full (run 3).

**Scoring is deterministic** — no LLM grading another LLM. Dates and amounts are normalized across formats (`31 March 2027` = `3/31/2027` = `2027-03-31`). An answer to "list all …" must name **exactly** the right agreements: one missing or one extra makes it wrong. A wrong answer counts as a hallucination; "not found" for an answerable question counts as a miss. See [`src/docassist/evaluation.py`](src/docassist/evaluation.py).

One rule was changed after run 4, and it is applied to every method and every earlier run alike ([`scripts/rescore.py`](scripts/rescore.py)): asked *"how many days of notice prevent the automatic renewal of …"* about an agreement without automatic renewal, the cloud pipeline answered *"No notice is required because the agreement does not automatically renew"* — which is right, but the scorer only accepted "not found". Rejecting the false premise now counts as a correct "not found". It moved the cloud pipeline from 81% to 89% and the other methods by 0–2 points (run 3's cloud pipeline: 84% to 86%).

## Limits, stated plainly

- **"Across all agreements" questions about clauses are the weak spot for every method** — 59% at best. Some of that is the question, not the system: for "which agreements can be terminated for convenience?", all strong methods flagged an agreement where *"the Adviser may terminate at any time"*, which the CUAD lawyers did not label as termination for convenience. On a client project these definitions are agreed with the client first.
- **The sample is moderate** (90 agreements, 159 questions; each interval is about ±6 points). Differences of 1–2 points between the strong methods are noise; what holds up is the cost and speed differences (38× against all-documents-in-the-prompt; against Codex 4× per question and about 6× the wait), the gap between the local and the cloud model, and the gap between the local pipeline and the local agent.
- **At a few hundred documents** the "all documents in the prompt" method no longer fits in a 1M-token context, and the agents have much more to search; the pipeline's per-question cost stays flat. That is an extrapolation, not a measurement.

## How it works

```
                 ┌──────────────── ingest (once per document) ────────────────┐
 PDF/DOCX/scan → │ parse (+OCR) → extract fields (LLM, YAML schema, in parts)  │
                 │ → chunk + document header on every chunk → embed + FTS5     │
                 └───────────────┬──────────────────────────┬─────────────────┘
                                 │                          │
                     pages · chunks · vectors      one row per document
                                 │                          │
 question ─┬─ names one document ──→ read that document in full (+ its row) ─→ answer with [citations]
           ├─ across all documents ─→ read-only SQL over the table ──────────→ answer + table
           └─ anything else ────────→ hybrid search (BM25 + vectors, RRF) ───→ answer with [citations]
```

| Problem | Fix | Where |
|---|---|---|
| Exact IDs and names get lost in vector search | **Hybrid retrieval**: SQLite FTS5 (BM25) + embeddings, fused with Reciprocal Rank Fusion; identifiers are searched as phrases | `retrieval.py` |
| Boilerplate clauses don't say which document they belong to | **Context headers**: identifying fields are extracted once and prepended to every chunk before indexing | `pipeline.py` |
| A question about one named document retrieves the wrong clause | **Document scoping**: the question is answered from that document only — the whole of it if it fits (60k characters by default), plus its extracted record | `pipeline.py` |
| "List all / how many / total" questions | **Structured extraction** at ingest into a table defined in a YAML schema; such questions become a **read-only, single-statement SQL** query; if it finds nothing, the model sees the stored values and may loosen it once, then the answer is "no documents match"; all-NULL results fall back to search | `structured.py`, `schemas/` |
| Made-up answers | Answer only from numbered excerpts, cite every claim, reply `NOT_FOUND` otherwise | `pipeline.py` |
| Users say "our agreement with JOHNSON", not the full title | **Name matching**: full names, names without the legal form, and short names unique in the library all identify a document | `names.py` |
| An extracted "yes" that the model guessed | **Quote check**: every extracted value comes with a quote that must be found in the text; an unverified "yes" is stored as unknown, and table answers show the quote behind each value | `structured.py` |
| Scanned PDFs | Pages without a text layer are OCR'd with RapidOCR (ONNX, CPU) | `parsing.py` |

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/), plus a model server. With [LM Studio](https://lmstudio.ai/): load a chat model and `text-embedding-nomic-embed-text-v1.5`, start the server.

```bash
cp .env.example .env              # pick local or cloud models here
uv sync --frozen --extra ocr
uv run python scripts/make_corpus.py          # 30 sample contracts + golden set
uv run docassist ingest data/corpus
uv run docassist ask "Which contracts renew automatically?"
uv run docassist export contracts.xlsx

uv run uvicorn docassist.api:app              # web UI on http://localhost:8000, API docs at /docs
```

Or with Docker (the model server stays on the host):

```bash
docker compose up --build         # UI at http://localhost:8000
```

**Several model setups.** Copy [`profiles.example.yaml`](profiles.example.yaml) to `profiles.yaml` and set `DOCASSIST_PROFILES_PATH`: the UI then switches between presets (local, cloud chat with local embeddings, all cloud). Each embedding model gets its own library, since vectors from different models can't be compared; API keys are written as `${ENV_VAR}` and never reach the browser.

**Parallel model requests.** Ingestion reads the parts of a long document side by side, and uploads and questions can run at the same time; one limit caps the requests in flight to the model, across all of them. By default it is chosen from the hardware: a local model server serves parallel requests from one pool of memory, so on a laptop without a dedicated GPU it is 1 (more would only make every answer slower), and with a GPU it grows with the VRAM; for a cloud API the UI suggests 4. It can be changed per preset in the UI (`Parallel` in the header) or fixed with `DOCASSIST_CONCURRENCY`.

**Long-running work is visible and can be stopped.** Ingestion, re-processing, evaluation runs and questions report their progress to a shared folder (`DOCASSIST_ACTIVITY_DIR`, default: the temp folder), so the UI's *Background work* panel also shows runs started from the command line — useful when a 20-minute ingestion is holding the local model — and its *Stop* button ends them after the current step.

### Reproduce the comparison

Download [CUAD v1](https://zenodo.org/records/4595826) and unzip it. The agreements of every run are in `data/corpus-cuad*/`, with their questions in `evals/cuad-golden*.yaml`. Runs 1–3 are rebuilt exactly by:

```bash
uv run python scripts/make_cuad_set.py --cuad path/to/CUAD_v1                                   # run 1
uv run python scripts/make_cuad_set.py --cuad path/to/CUAD_v1 --seed 2027 --exclude data/corpus-cuad \
    --out data/corpus-cuad-holdout --golden evals/cuad-golden-holdout.yaml                         # run 2
uv run python scripts/make_cuad_set.py --cuad path/to/CUAD_v1 --seed 4242 --exclude data/corpus-cuad \
    --exclude data/corpus-cuad-holdout --out data/corpus-cuad-holdout2 --golden evals/cuad-golden-holdout2.yaml  # run 3
```

Then, with the model settings for local or cloud in the environment:

```bash
SET="--corpus data/corpus-cuad-holdout2 --golden evals/cuad-golden-holdout2.yaml"
DOCASSIST_SCHEMA_PATH=schemas/agreement.yaml uv run python scripts/run_eval.py $SET --configs full --out evals/cmp/local-rag
uv run python scripts/run_fullread.py --mode map $SET --out evals/cmp/local-fullread     # cloud: --mode single
uv run python scripts/run_agent.py $SET --out evals/cmp/local-agent
docker build -t codex-bench:0.160.0 scripts/codex                                       # Codex in a container
uv run python scripts/run_codex.py $SET --out evals/cmp/cloud-codex --codex-config path/to/config.toml
uv run python scripts/compare_report.py                                                  # run 3's table
```

Run 4 (the table above), on `data/corpus-cuad-set4` … `set6`:

```bash
uv run python scripts/run_combos.py --combos evals/cmp4/combos.yaml --sets 4,5,6 --stores path/to/stores  # this pipeline, per model combination
for N in 4 5 6; do
  SET="--corpus data/corpus-cuad-set$N --golden evals/cuad-golden-set$N.yaml"
  uv run python scripts/run_agent.py $SET --out evals/cmp4b/cloud-agent/set$N           # local: evals/cmp4b/local-agent
  uv run python scripts/run_fullread.py --mode single $SET --out evals/cmp4b/cloud-fullread/set$N
  uv run python scripts/run_codex.py $SET --out evals/cmp4b/cloud-codex/set$N --codex-config path/to/config.toml
done
uv run python scripts/blind_report.py                                                    # the table above
uv run python scripts/combos_report.py --cuad path/to/CUAD_v1 --stores path/to/stores     # model combinations + extraction accuracy
```

The synthetic-contract evaluation (each fix measured on its own, plus a held-out set):

```bash
uv run python scripts/run_eval.py             # writes evals/results/report.html
uv run python scripts/check_extraction.py     # field-level extraction accuracy
```

Tests run offline with a fake model (no server, no API key): `uv run pytest`.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/documents` | Upload and ingest files (multipart). Unchanged files are skipped by hash. |
| `POST` | `/ingest-jobs` | Same, in the background; `GET /ingest-jobs/{id}/events` streams every step (page read, OCR, extraction part, quote check, indexing) as NDJSON |
| `GET` | `/documents` | Ingested documents with page / OCR / chunk counts |
| `GET` | `/documents/{filename}/pages` | The document's text by page, its extracted row and the quote behind each value |
| `DELETE` | `/documents/{filename}` | Remove a document and its index entries |
| `POST` | `/documents/delete`, `/documents/reprocess` | Remove several documents; extract and index them again with the current models, in the background |
| `POST` | `/ask` | `{"question": "...", "document": null}` → answer, route, sources (file + page), SQL + rows and the quote behind each value for table answers, tokens and cost. `document` restricts the question to one file. |
| `POST` | `/ask/stream` | Same, streaming each step (named documents found, route, SQL, search) before the answer |
| `GET` | `/records`, `/records.xlsx`, `/evidence` | The extracted table as JSON or Excel; the quote and page behind every value |
| `PUT`, `DELETE`, `GET` | `/records/value`, `/edits` | Correct one extracted value by hand (questions over the table then use it), undo, list corrections |
| `GET`, `PUT` | `/profiles`, `/profiles/active` | Model presets (local / cloud) and switching between them; keys stay on the server |
| `GET`, `PUT` | `/concurrency` | Parallel model requests for the active preset, the recommendation and the hardware it is based on |
| `GET`, `POST` | `/activity`, `/activity/{id}/stop` | Running and recent work on this machine (also from the command line); stop a task after its current step |

## The synthetic set: each fix measured on its own

Where the pipeline was built: **30 synthetic contracts** from one template (26 PDF, 2 DOCX, 1 scanned PDF with no text layer — handled by OCR), with deliberate traps: identical boilerplate, near-identical contract numbers (`…-071` vs `…-017`) and company names, facts written in several formats, totals only in a fee table. 44 questions of the same three kinds. Each column adds one fix on top of the previous one, local model:

| Column | Adds | The trap it handles |
|---|---|---|
| `naive` | nothing: chunks, vector search, answer from the top few | the tutorial RAG, as a starting point |
| `+hybrid` | BM25 keyword search, fused with the vector results (RRF) | contract numbers that differ in two digits look alike to embeddings |
| `+headers` | the document's title, parties and number on every chunk | identical template clauses that never say which contract they are from |
| `+scoping` | a question that names a contract is answered from that contract only | the same clause retrieved from a different contract |
| `full` | a table extracted at ingestion; "list all / how many / total" questions become SQL | a few retrieved passages can never list all 30 contracts |

| | `naive` | `+hybrid` | `+headers` | `+scoping` | `full` |
|---|---:|---:|---:|---:|---:|
| Overall accuracy | 30% | 41% | 75% | 80% | **100%** |
| Lookup (one document) | 19% | 38% | 92% | 100% | **100%** |
| Aggregate (across all documents) | 10% | 0% | 10% | 10% | **100%** |
| Says "not in the documents" when it isn't | 88% | 100% | 100% | 100% | **100%** |
| Hallucination rate | 27% | 27% | 16% | 16% | **0%** |
| Median seconds per question | 6.2 | 7.9 | 7.5 | 6.4 | **4.0** |

A second synthetic set with a different seed (other contract types, dates, amounts, numbering and questions) scores 34% `naive` and **100%** `full`; field extraction is 840 of 840 correct across both (re-run with the current code on 2026-11-08). These numbers are high because the fixes were written while looking at these questions and the documents are clean — which is why the real-contract comparison above, on agreements never seen during development, is the result that matters. Reports: [`evals/results/report.html`](evals/results/report.html), [`evals/results-holdout/report.html`](evals/results-holdout/report.html).

![Extracted data: one row per agreement, a green dot where the value's source sentence was found in the document; exportable to Excel](docs/img/table.png)

## Notes from building it

- **A Windows system proxy hijacked calls to `localhost`.** The OpenAI client honors system proxy settings, so requests to the local model server went to the VPN proxy and came back `502`. Clients for local / LAN endpoints now ignore proxy environment settings.
- **OCR turned clean lines into garbage — the text-direction classifier was the cause.** It flipped some slightly blurred lines by 180°. Business scans are upright; disabling it took garbled lines on the sample scan from 12 of 84 to 0.
- **Generated SQL compared text with `=`** (`'Non-Disclosure Agreement'` vs the stored `'Mutual Non-Disclosure Agreement'`), returned zero rows, and the model faithfully reported "no documents match". Fixes: `LIKE` matching, and an empty result is shown to the model with the values actually stored, so it can loosen the query once before "none" is accepted.
- **A small model reading one document at a time answered "NONE" to "list all agreements governed by New York law"** — it could not list *all* from one document. Asking each part "does *this* document match?" fixed it. The baselines got the same care as the pipeline; a comparison against a broken baseline proves nothing.

## Project layout

```
src/docassist/   config, llm client (+ token/cost tracking), parsing (+OCR), chunking, store (SQLite),
                 retrieval, document names, structured extraction + SQL, pipeline, model profiles,
                 progress events, evaluation, API, CLI, Excel export
schemas/         extraction schemas (YAML) - one per document type: contract.yaml (synthetic), agreement.yaml (CUAD)
scripts/         corpus generators, eval runner, the comparison methods (run_fullread, run_agent, run_codex,
                 run_combos), reports (blind_report, combos_report, compare_report), rescore
evals/           golden question sets and results: results*/ (pipeline), cmp/ (run 3), cmp4/ + cmp4b/ (run 4)
src/docassist/web/  web UI (plain HTML/CSS/JS, served by the API; no build step)
tests/           unit + API tests with a fake LLM
```

The code is MIT-licensed ([`LICENSE`](LICENSE)). The contracts in `data/corpus*` (no `-cuad`) are fictional, generated by `scripts/make_corpus.py`. The agreements in `data/corpus-cuad*` are from **CUAD v1** by The Atticus Project, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); they are public SEC filings.

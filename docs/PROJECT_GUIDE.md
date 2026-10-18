# Reddit RabbitHole RAG: Project and Interview Guide

This guide describes the repository as implemented today and the measured Build 1 and Build 2 runs supplied for this project. Retrieval numbers are preliminary; answer generation has not been evaluated end to end.

## 1. The project in one minute

**Two-sentence pitch.** Reddit RabbitHole RAG retrieves question and answer pairs from Reddit's IAmA community and presents a grounded response with links to source comments. It samples whole threads deterministically, matches the submission author as the host, cleans and filters comment pairs, then combines dense and lexical search with reranking and an abstention gate.

**30-second version.** I built a retrieval-augmented search system over Reddit AMAs. Its main data challenge is that comments are nested and the host must be identified correctly. The pipeline stores comments and extracted pairs in SQLite, builds a dense FAISS index and a BM25 index, combines their rankings with Reciprocal Rank Fusion, reranks candidates, and shows cited answers in a Gradio app. I have preliminary retrieval measurements on an earlier index; I have not measured end-to-end generation quality.

**Two-minute walkthrough.** I started with public IAmA discussions. A flat comment dump loses reply structure, so ingest stores each comment's thread and parent identifiers in SQLite. Sampling uses a stable hash of the thread id: selected threads stay whole, and resuming a scan uses the same sample. Submission metadata supplies the real host account instead of guessing from comment frequency. Extraction follows reply relationships, cleans quoted and formatted text, applies length and score rules, deduplicates pairs by content hash, and caps pairs per AMA.

For search, I embed each title, question, and answer and also index the text with bm25s. Dense and BM25 rankings are fused by rank instead of mixing their differently scaled raw scores. A cross-encoder then scores a smaller candidate list, and a per-thread cap prevents one AMA from filling the results. A two-tier reranker gate can mark a result low confidence or abstain before generation. When generation runs, Gemini receives source excerpts and instructions to cite them; the Gradio app validates citation numbers and shows source links. My next evaluation step is to regenerate synthetic queries for the current index and measure answer behavior on a larger, reviewed set.

## 2. Problem and why AMAs are difficult to search

**An AMA is a conversation tree, not a document.** A question may directly reply to the submission or appear several levels down. Answers can quote earlier context, and many participants are not the AMA host. Useful extraction therefore depends on parent-child relationships and a reliable host identity.

The original baseline guessed the host from top-level comment frequency. That can select an active participant instead of the submission author. It also retained HTML entities, quote walls, and host-to-host replies, and did not deduplicate pairs. This project requests submission metadata from Arctic Shift and uses the submission author when available.

The source is large and time-ordered. A first-N scan may cover an early slice rather than a representative timeline. Streaming resumes from a saved offset, while whole-thread sampling uses a deterministic id hash so the same thread is not randomly included in one run and excluded in another. The measured source speed was about 2,600 rows/s; assuming 50,000 rows/s would seriously understate runtime.

Reddit text is noisy and sometimes adversarial. It can contain markup, URLs, quotes, or instructions aimed at a model. Cleaning reduces formatting noise; generation escapes source text and labels it as data. The application also escapes content before rendering it as HTML.

## 3. Architecture and module map

**The system is a staged pipeline.** Raw comments and metadata become cleaned pairs; pairs become retrieval artifacts; the application loads those artifacts and can generate a cited answer.

```text
Hugging Face IAmA comments           Arctic Shift submission metadata
              |                                      |
              v                                      v
      resumable ingest --------------------------> threads table
              |                                      |
              +--------------------+-----------------+
                                   v
                          Q&A extraction
                                   |
                                   v
                         SQLite pairs table
                                   |
                     +-------------+-------------+
                     v                           v
              dense embeddings                 BM25 tokens
                     |                           |
                   FAISS                    bm25s index
                     +------------+--------------+
                                  v
                          RRF candidate merge
                                  |
                          cross-encoder rerank
                                  |
                       diversity and confidence
                                  |
                         grounded Gemini answer
                                  |
                       citations and source links
```

| File | Responsibility | Key functions or types |
|---|---|---|
| `rabbithole/config.py` | Environment-backed settings and defaults | `Settings`, `get_settings` |
| `rabbithole/db.py` | SQLite schema and state helpers | `connect`, `get_state`, `set_state` |
| `rabbithole/ingest.py` | Resumable stream or parquet ingest | `ingest`, `keep_thread`, `_ingest_parquet` |
| `rabbithole/meta.py` | Submission metadata lookup and retry | `fetch_meta`, `_fetch_batch` |
| `rabbithole/text.py` | Text cleanup, ids, hashes, truncation | `clean_comment`, `content_hash`, `truncate_sentences` |
| `rabbithole/extract.py` | Tree traversal, filters, statistics | `ExtractConfig`, `extract_thread`, `run_extract`, `corpus_stats` |
| `rabbithole/categories.py` | Keyword topic labels | `categorize` |
| `rabbithole/embed.py` | Embedding and reranking backends | `STEmbedder`, `HashEmbedder`, `CrossEncoderReranker` |
| `rabbithole/indexing.py` | FAISS, BM25, document store, manifest | `build_index`, `build_faiss`, `embed_text` |
| `rabbithole/retrieval.py` | Search, rank fusion, diversity, abstention | `Retriever.search`, `rrf_fuse` |
| `rabbithole/generation.py` | Gemini calls, prompts, citation checks | `GeminiClient`, `build_prompt`, `check_citations` |
| `rabbithole/pipeline.py` | Retrieval-to-answer orchestration | `RagPipeline`, `build_pipeline` |
| `rabbithole/eval.py` | Synthetic queries and evaluation metrics | `make_synthetic_queries`, `eval_retrieval`, `wilson_interval` |
| `rabbithole/artifacts.py` | Artifact download and publication | `ensure_artifacts`, `upload_folder`, `upload_file` |
| `rabbithole/cli.py` | Pipeline command line | `main`, parser setup |
| `app.py` | Gradio UI and request controls | `respond`, `sources_html`, `launch_app` |

## 4. Data pipeline

**The pipeline preserves enough structure to extract host answers later.** The details below follow `rabbithole/ingest.py`, `rabbithole/meta.py`, and `rabbithole/extract.py`.

1. **Ingest and resume.** Parquet is the default ingest method; stream is the fallback. Rows are inserted into SQLite with comment id, thread id, parent id, author, body, score, and timestamp. Stream progress stores the consumed row count. Parquet progress stores completed shard names and the consumed count. A real end of stream or all discovered parquet shards completing sets `ingest_complete`; split metadata is not itself a completion signal.
2. **Deterministic whole-thread sampling.** `keep_thread` applies CRC32 to the thread id and compares that value with the configured sample rate. All comments in a thread therefore get the same decision, and resume repeats the decision without storing a random sample map. The cost is that every source row still must be scanned for timeline coverage.
3. **Find the host.** `fetch_meta` requests submission data from Arctic Shift's `/api/posts/ids` endpoint. The submission author is treated as the true OP. An optional heuristic fallback chooses the most active top-level non-bot commenter, but it is disabled by default and marked as heuristic when used.
4. **Clean text.** `clean_comment` unescapes HTML entities up to three passes; can drop quoted lines from answers or remove quote markers from questions; removes Markdown link destinations, URLs, headings, emphasis, superscript markers, zero-width characters, extra spaces, and excessive blank lines. It cleans formatting rather than rewriting meaning.
5. **Extract and filter.** The defaults below come from `ExtractConfig`. Extraction considers replies authored by the OP, finds the parent question, rejects missing parents, OP-parent replies and bot parents, and applies a higher minimum to short follow-up questions.
6. **Deduplicate and cap.** A normalized question/answer content hash has a uniqueness constraint. Candidates are sorted by answer score and answer length, then capped per AMA.
7. **Assign a topic.** `categorize` checks the title first and intro second using ordered regular expressions. The first matching category wins; no match becomes `other`. This is a fast filter, not a trained classifier. Overlapping terms can produce poor labels.

`ExtractConfig` defaults in `rabbithole/extract.py`:

| Setting | Default | Meaning |
|---|---:|---|
| `min_answer_chars` | 80 | Minimum cleaned answer length |
| `max_answer_chars` | 4000 | Maximum cleaned answer length |
| `min_question_chars` | 15 | Minimum cleaned question length |
| `max_question_chars` | 1500 | Maximum cleaned question length |
| `min_answer_score` | 1 | Minimum Reddit score for an OP answer |
| `max_pairs_per_thread` | 200 | Maximum retained pairs per AMA |
| `min_thread_comments` | 10 | Minimum comments before metadata/extraction |
| `require_question` | `True` | Require question punctuation or a recognized opener |
| `allow_heuristic_op` | `False` | Do not guess host when metadata is absent |
| `followup_min_chars` | 60 | Minimum for a follow-up dependent on the OP's prior reply |

Build 2's counts are in the README. That run stopped before a completion marker, so full source coverage cannot be inferred from its row count.

## 5. Retrieval

**Search uses two different signals, then a learned reranker.** Each stage narrows or reorders candidates; the measured timings below are from the preliminary Build 1 index only.

- **Dense embeddings.** The default model is `BAAI/bge-small-en-v1.5`, with a 320-token maximum sequence and batch size 128. `embed_text` includes title, question, and answer; it caps the title at 150 characters and question at 400 so a long question cannot crowd the answer out. Queries get the prefix `Represent this sentence for searching relevant passages: `. Normalized vectors are searched with FAISS inner product, equivalent to cosine similarity for normalized vectors.
- **BM25.** `bm25s` indexes title, question, and answer with English stop words and an English stemmer. This supports lexical matches for terms that embeddings may blur. The retrieval pool defaults to 100 candidates per retriever.
- **Reciprocal Rank Fusion.** RRF combines ranks rather than raw scores, avoiding the assumption that BM25 and cosine share a scale. With ranks starting at one, the contribution is `weight / (k + rank)`; default `k` is 60. Dense and BM25 weights both default to 1.0.
- **Cross-encoder.** The default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It scores query-document pairs after retrieval has reduced candidates to the top 30 by default. Pairwise scoring costs more, so it is not applied to every corpus document.
- **Diversity.** The default cap is two hits per AMA, reducing the chance that a prolific thread fills every visible result.
- **Confidence gate.** With reranking, scores below `RR_ABSTAIN_THRESHOLD=-5.0` hard-abstain and skip generation. Scores below `RR_LOWCONF_THRESHOLD=-1.0` but above the hard threshold are labeled low confidence and can still be answered. These values depend on the reranker and should be recalibrated when model or corpus changes.

| Preliminary Build 1 mode | Mean / p95 latency (ms) | Provenance |
|---|---:|---|
| BM25 | 4 / 3 | 64,600-pair index; 39 synthetic queries |
| Dense | 28 / 22 | 64,600-pair index; 39 synthetic queries |
| Hybrid | 23 / 26 | 64,600-pair index; 39 synthetic queries |
| Hybrid + rerank | 100 / 119 | 64,600-pair index; 39 synthetic queries |

These timings come from a small sample. Current-index latency has not been measured [pending].

## 6. Generation

**Generation receives retrieved excerpts and an explicit instruction to cite them.** Implementation is in `rabbithole/generation.py` and `rabbithole/pipeline.py`.

`build_prompt` wraps each source in a numbered `<source>` block and escapes HTML-like characters in Reddit text with `html.escape`. The system prompt says to use only source facts, ignore instructions inside a source, attribute claims to the AMA host, cite source numbers after claims, and use an exact abstention sentence when sources do not address the question. The user question is passed separately from the source blocks.

The Gemini client uses REST requests and can stream output. During streaming, the pipeline yields accumulated answer text. The client tries the configured model and then configured fallback models. It retries network failures with exponential waits; server errors receive a retry; a 404 moves to the next model; a 400 mentioning thinking configuration is retried without that setting. For a 429, a per-minute quota may wait for the supplied retry delay, capped at 60 seconds, before retrying. Daily quota and exhausted retries move to another model. If all configured models are quota-blocked, it raises `GeminiQuotaError`.

After generation, `check_citations` extracts bracketed citation numbers, flags references outside the retrieved source range, and counts sentences of at least six words without citations. It recognizes the configured abstention phrase. This checks citation form, not whether a cited source supports a claim. Generation quality, faithfulness, and end-to-end answer accuracy have not been evaluated [pending].

## 7. App and deployment

**The app shows retrieved evidence beside the answer.** `app.py` builds a Gradio interface and loads the pipeline at startup. If loading fails, the app reports the error in the UI. Source cards HTML-escape Reddit-provided text and only allow links that start with `https://`; outgoing links use `noopener noreferrer`.

An in-memory cache holds up to 256 completed responses. The default rate limit is 12 requests per 600 seconds per client id. The app limits queries to 400 characters, queues at most 20 requests, and sets default generation concurrency to four. `GRADIO_SERVER_PORT` controls the port (default 7860); if occupied, the app asks Gradio to select an available port.

Artifacts are fetched from a Hugging Face dataset repository when the local artifact directory is incomplete. The manifest records document count, embedding dimension and model, categories, thread count, build timestamp, and optional year bounds. The Gradio Space needs application dependencies, access to an artifact dataset, and `GEMINI_API_KEY` for generated answers. The deploy helper copies `app.py`, `rabbithole/`, and `requirements.txt` into the Space checkout and pushes it.

## 8. Evaluation

**The evaluation code can measure retrieval independently of generation.** It builds synthetic questions from indexed pairs, searches for the source pair, and compares result ranks.

Retrieval metrics include pair Recall@k, thread-level Recall@k, and mean reciprocal rank (MRR). `wilson_interval` gives a binomial confidence interval for recall; `_bootstrap_mean_interval` estimates an interval for MRR. Per-query rank and latency records are saved for inspection. Queries store a stable `answer_id` and thread id so they can be resolved against a rebuilt index. Older positional-only files are accepted only when the document at that position still belongs to the expected thread. Stale rows are dropped and reported; if more than half are stale, the run asks for query regeneration.

Calibration compares top reranker scores for in-domain and out-of-domain prompts. It emits pass and abstain rates across candidate thresholds and selects the highest threshold whose in-domain pass rate is at least 95%, when one exists. This is a seed-set calibration procedure, not a general proof of confidence.

The preliminary retrieval table was measured on Build 1's 64,600-pair index with 39 synthetic queries. Approximate uncertainty was ±15 percentage points. Build 2's evaluation file had positional ids generated from Build 1; every recall and MRR value was zero because positions no longer matched. Those results were discarded. Stable answer-id matching and stale-query detection now make that mismatch visible instead of silently reporting an all-zero table. Retrieval on current artifacts has not been measured [pending].

The project has not completed a generation evaluation. In the supplied Build 2 run, `ask` and `eval-generation` stopped on Gemini 429 quota errors. No generation quality, citation faithfulness, or end-to-end accuracy claim is made [pending].

## 9. Engineering lessons: problem, evidence, fix, lesson

**The most useful lessons came from failures that looked valid until checked.**

| Problem | Evidence | Fix or current handling | Lesson |
|---|---|---|---|
| Baseline guessed the AMA host from comment frequency and retained noisy text | Host-to-host replies, quote walls, HTML entities, and duplicates entered the baseline; it had 2,671 pairs from 451 threads and only the first ~500k comments | Match submission author through metadata, clean text, traverse reply trees, filter and deduplicate | Correct identity and structure matter before model choice |
| Streaming was much slower than planned | Actual speed was about 2.6k rows/s rather than the assumed 50k | Measure throughput and estimate duration from observed speed | Plan from a measured sample |
| First abstain threshold rejected a relevant nurse query | “What is the hardest part of being a nurse?” was rejected although all four hits were nurse AMAs | Use hard-abstain and low-confidence thresholds separately | A single threshold can confuse weak evidence with no evidence |
| Resume log inflated throughput | It divided cumulative scanned rows, including the resume offset, by the current session time; it showed 404,495/s early and 3,416/s later | Compute rate from consumed rows minus the session starting offset | Numerator and elapsed time must use the same time window |
| Rebuilt-index evaluation silently became all zeros | The 39-query file stored positions from Build 1 and was used against Build 2 | Store answer ids, resolve them in the current index, report stale rows, and stop when most are stale | Evaluation must use stable document identity |
| Gemini free-tier quota interrupted query generation and answer checks | Build 2 `ask` and `eval-generation` failed on 429 responses | Keep quota behavior and fallbacks; do not present missing evaluation as a quality result | Separate retrieval evidence from generation claims |
| Artifact dates were unavailable for year reporting | Artifacts built before this date field lack `created_utc` and year bounds | Add timestamps to new doc stores and manifest; keep old artifacts loadable | Store provenance needed for later analysis |
| Source count disagreed with stream | Build 2 scanned about 26.2M rows vs metadata count 25,778,822 and had no completion marker | Warn on metadata overrun; rely on actual completion, not metadata | Metadata is a hint, not proof of completion |

## 10. Design decisions and trade-offs

**Each design addresses a specific failure mode and has a cost.** These are engineering comparisons, not claims that one choice is universally best.

| Decision | Why this implementation uses it | Cost or alternative |
|---|---|---|
| RRF over weighted raw-score sum | BM25 and cosine have unrelated score scales; ranks avoid choosing normalization | Loses score magnitude; weights and `k` still affect ordering |
| bm25s over the original rank_bm25 pickle | Current sparse index is saved without Python pickle loading | Adds a dependency and token-index storage |
| FAISS flat or HNSW | Auto uses flat inner product through 200,000 documents and HNSW above that | Flat is exhaustive; HNSW is approximate |
| SQLite over loose files for working data | Comments, metadata, pairs, and progress state are queryable and transactional | Large DB needs disk, indexes, and checkpoint care |
| Thread hash sampling over first-N | Selected threads stay whole and sampling is repeatable | It still scans source rows; interrupted early scans remain time-biased |
| Cross-encoder over all-pairs scoring | Pairwise scoring reranks a manageable candidate set | Adds latency; missed candidates cannot be recovered |
| Abstain gate over LLM-only abstention | Retrieval scores can avoid generation calls for very weak evidence | Thresholds are model-specific and can suppress relevant queries |
| Keyword categories over a classifier | Fast, transparent, and useful for filters | Vocabulary gaps and overlap cause errors |
| Private DB checkpoint plus public runtime artifacts | App downloads compact runtime files while working data stays private | Checkpoint is larger and contains source text |

## 11. Limitations and next steps

**The available evidence supports a retrieval prototype, not a validated answer engine.**

- **Coverage:** Build 2 stopped around 26.2M scanned rows, beyond the split metadata count, without a completion marker. Exact year range and full timeline coverage are unconfirmed [pending]. Complete the scan and rebuild artifacts with dates.
- **Generation:** Gemini quota errors stopped answer and evaluation requests. End-to-end answer quality, faithfulness, and citation support are unmeasured [pending]. A reviewed set and usable API budget are needed.
- **Calibration:** Build 2 calibration used the old rule and a 30/12 seed split. Recalibrate on current artifacts with an expanded set [pending].
- **Synthetic evaluation:** Generated questions may be easier than real user questions and reflect generator style. A human-reviewed query set is a useful complement [pending].
- **Reranking:** The prison-painting result shows lexical overlap can still mislead the reranker. More error analysis is needed [pending].
- **Categories:** Keyword rules are coarse and overlap. Measure label agreement before using them as research labels [pending].
- **Historical period:** The source ordering is time-based and the supplied run begins in 2009. Exact corpus coverage remains unconfirmed [pending].
- **Sensitive content and privacy:** AMAs may discuss abuse, addiction, health, and other sensitive experiences. Preserve source attribution and avoid presenting individual accounts as universal or verified facts.
- **Scale:** At 100 times the current corpus, local artifacts, SQLite, index build time, and candidate quality need benchmarking. Measure bottlenecks before selecting partitioning or managed services [pending].

## 12. Interview Q&A bank

**Answer from the design and measurements that exist.** If the project has not run an experiment, state that directly.

### 1. What problem does this project solve?
It lets users search personal accounts in Reddit IAmA threads with citations to source comments. The pipeline extracts host answers from nested comment trees instead of treating the dump as flat text. It is a retrieval prototype, and the answers are anecdotes rather than verified facts.

### 2. Why use RAG instead of training a model?
The corpus is source material that retrieval can show and link to at request time. The generator receives excerpts instead of relying on facts memorized in weights. Fine-tuning would not replace retrieval or citations, and this project has not run a fine-tuning comparison.

### 3. Why is hybrid search useful?
Dense embeddings can match related meanings, while BM25 can match names and exact terms. The system retrieves with both and combines rankings with RRF. The small evaluation does not prove that hybrid always wins on real queries.

### 4. What is BM25?
BM25 is a lexical ranking method based on query-term matches, term frequency, and document length. This project uses bm25s with English stop words and stemming. It indexes each title, question, and answer.

### 5. What is dense retrieval?
Dense retrieval maps text into numeric vectors and compares query and document vectors. This project uses normalized sentence-transformer embeddings and FAISS inner-product search. The embedding model and query prefix are configured and recorded in the manifest.

### 6. What is Reciprocal Rank Fusion?
RRF adds a reciprocal contribution for each item's position in each ranked list: a weight divided by rank plus a constant. Here the default constant is 60 and both retrieval weights default to 1.0. It combines ordering without assuming cosine and BM25 raw scores have the same scale.

### 7. How do you know retrieval is good?
The evaluation code measures pair Recall@k, thread Recall@k, and MRR on synthetic queries, with Wilson intervals for recall and bootstrap intervals for MRR. The available table has 39 queries on Build 1 and is preliminary. Build 2 needs a regenerated set and fresh evaluation.

### 8. Why use a cross-encoder?
A cross-encoder reads the query and one candidate together and returns a relevance score. It can refine ordering after dense and lexical retrieval narrow the list. It costs more per candidate, so the default rerank candidate count is 30 rather than the whole corpus.

### 9. How do you reduce hallucination?
The prompt tells Gemini to use only excerpts, attribute claims, cite source numbers, and abstain when sources do not address the question. A retrieval gate can also skip generation below the hard threshold. These controls are design choices; answer faithfulness has not been measured.

### 10. How do you handle prompt injection?
The system prompt labels excerpts as data and says to ignore instructions found inside sources. The prompt builder escapes HTML-like characters and wraps excerpts in numbered source tags. This reduces risk but does not guarantee resistance to every attack.

### 11. What is an abstain threshold?
It is a reranker-score boundary below which the system returns an abstention rather than calling Gemini. A separate low-confidence boundary marks weaker retrieval while still allowing an answer. Defaults are -5.0 and -1.0 and should be calibrated for the corpus and reranker.

### 12. Why use two thresholds?
The first threshold wrongly abstained on the nurse question even though all four hits were nurse AMAs. Separating hard abstention from a low-confidence label communicates uncertainty without suppressing every borderline result. The thresholds remain empirical choices.

### 13. How does the system identify the AMA host?
It fetches submission metadata from Arctic Shift and uses the submission author as the OP. That is more direct than guessing from top-level comment frequency. A heuristic fallback exists but is disabled by default and labels its result as heuristic.

### 14. Why sample threads instead of taking the first rows?
The source is time-ordered and a thread's comments can be spread through the stream. Hash sampling gives every comment from one thread the same decision and makes resumption repeatable. A complete scan is still needed to spread the sample across the timeline.

### 15. How does resume work?
Streaming stores the consumed row offset; parquet ingest stores completed shard names and progress. Re-running uses the same database and sample rate, and a different rate is rejected when progress exists. Completion relies on the reader reaching its real end, not a metadata count.

### 16. How do you deduplicate pairs?
The code normalizes the question and answer, truncates each normalized component, and hashes the joined text with SHA-1. The pairs table makes that content hash unique. This removes normalized duplicates but not every semantically equivalent pair.

### 17. Why cap pairs per thread?
A prolific AMA could otherwise dominate both the index and search results. Extraction sorts by answer score and answer length, then applies a cap of 200. That improves balance but can omit answers from unusually large threads.

### 18. What does a category label mean?
It is a heuristic selected from the title or intro using ordered keyword rules. It supports filtering and is not a scientific content label. No label-quality benchmark has been run.

### 19. How do you evaluate generation?
The eval-generation path checks abstention, citation formatting, truncation, latency, and optionally a model-based faithfulness score. The supplied run stopped on Gemini quota errors, so there is no completed generation result. A reviewed set and a functioning quota are next steps.

### 20. What was the hardest bug?
The stale eval file looked valid but all metrics were zero because it stored document positions from an earlier index. Rebuilding changed those positions. Stable answer ids and stale-query detection now expose that mismatch.

### 21. Why can source count be stale?
Metadata advertised 25,778,822 rows, while Build 2 had scanned around 26.2 million when interrupted. The metadata may be stale or the stream may be longer; that run did not reach a completion marker. I cannot claim stream completion from the observed count alone.

### 22. How do you handle slow ingestion?
I measure rows per second and compute the rate from rows scanned during the current process session. The older log mixed a cumulative row count with one-session elapsed time and inflated the rate after resume. The observed speed was about 2,600 rows/s.

### 23. Why not use only the LLM to decide when to abstain?
A retrieval gate can avoid an unnecessary generation call when evidence scores are very low. It is visible and configurable. The downside is model-specific scores and the possibility that a bad threshold suppresses relevant results.

### 24. What are synthetic evaluation's limitations?
Generated questions may resemble source wording or favor the question generator's style. They provide known source-pair labels, but may not represent how users ask questions. I would add reviewed human-written queries before making stronger performance claims.

### 25. What might break first at 100 times the scale?
The current design uses local artifacts, SQLite, and an index built on one machine. Disk capacity, build time, model loading, and first-stage retrieval need benchmarking. I would measure bottlenecks before choosing sharding or managed infrastructure.

### 26. How do you protect sensitive content?
The app attributes claims to individual AMA hosts and states that posts are personal anecdotes, not verified facts or advice. It displays links so users can inspect source context. Sensitive-topic and privacy review remain important.

### 27. What is the manifest for?
It records index properties such as document count, vector dimension, embedding model, categories, thread count, and build time. New builds also record year bounds when timestamps are available. Optional fields allow older manifests to remain loadable.

### 28. Why validate citations after generation?
The validator flags citation numbers outside the provided source range and counts longer sentences that have no citation. The UI can report those problems while keeping the source cards visible. It does not prove semantic support.

### 29. What changed after the Build 2 evaluation failure?
Queries now store Reddit answer ids and thread ids, then resolve them against the active corpus. Legacy positional files are accepted only if the expected thread still matches. Stale questions are counted and a majority-stale set gets a regeneration error.

### 30. What would you do next?
I would complete the scan, verify its completion marker, rebuild timestamped artifacts, and inspect the year histogram. Then I would regenerate retrieval queries, expand and review calibration data, and run generation evaluation with a usable quota. Every result would name its corpus and query-set provenance.

## 13. Glossary

**BM25:** Lexical ranking function based on matching query terms.

**Dense retrieval:** Search comparing vector embeddings of queries and documents.

**Embedding:** Numeric vector representation of text.

**Cosine similarity / inner product:** Similarity comparisons; for normalized vectors, inner product equals cosine similarity.

**FAISS:** Library used here to store and search dense vectors.

**RRF:** Reciprocal Rank Fusion; combines rankings with reciprocal contributions.

**Cross-encoder:** Model that scores a query and candidate together.

**Recall@k:** Fraction of questions whose relevant answer appears within the first k results.

**MRR:** Mean Reciprocal Rank; averages one divided by the first relevant result's rank.

**Wilson interval:** Approximate confidence interval for a binomial proportion such as recall.

**Abstention:** Returning that available sources do not sufficiently answer the question.

**Prompt injection:** Instructions inside retrieved or user-provided text that try to redirect a model.

**RAG:** Retrieval-augmented generation; retrieving source text and conditioning a language model on it.

## 14. Cheat sheet

**Use each number with its run provenance.** Build 1 and Build 2 are separate runs; retrieval results below are only from Build 1.

| Measure | v1 baseline | Build 1 | Build 2 | Provenance |
|---|---:|---:|---:|---|
| Q&A pairs | 2,671 | 64,600 | 200,181 | Supplied run facts |
| Threads with pairs | 451 | 5,912 | 13,424 | Supplied run facts |
| Comments kept | first ~500k | 915,905 | 4,020,606 | Build 2 resumed Build 1 DB |
| Rows scanned | Old README's 25.8M claim was wrong | interrupted around 6.22M | interrupted around 26.2M; completion unconfirmed | Supplied run facts |
| Sample rate | — | — | 0.19396 | Build 2 run facts |
| Ingest speed | — | — | ~2,600 rows/s | Corrected per-session estimate |
| Index time | — | 76 s | 277 s | T4 measurements |
| Public artifact size | — | 284 MB | 937 MB | Dataset upload |
| Embedding dimension | — | 384 | 384 | Build report; implementation default configured in code |
| Hybrid + rerank R@5 | — | .692 | not measured | Build 1; 39 synthetic queries |
| Hybrid + rerank MRR@20 | — | .653 | not measured | Build 1; 39 synthetic queries |
| Exact current year range | — | not available | unconfirmed | Needs full scan and timestamped artifacts |
| Generation quality | not measured | not measured | not measured | Quota interrupted evaluation |

### Three-minute local demo

Run these commands from the repository root. The app needs runtime artifacts and a Gemini key for generated answers; without the key, retrieval can load but generation reports missing configuration.

```powershell
# In PowerShell:
./run_local.ps1

# Or install and launch directly:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860/>. Ask “What is the hardest part of being a nurse?” and show the answer, source cards, and Reddit links. Explain that retrieval is measurable but generation quality has not been evaluated. If the port is occupied, set `$env:GRADIO_SERVER_PORT = "7861"` before launch; Gradio chooses another available port if that is busy too.

---

**Implementation sources:** `rabbithole/config.py`, `extract.py`, `ingest.py`, `indexing.py`, `retrieval.py`, `generation.py`, `pipeline.py`, `eval.py`, `artifacts.py`, `cli.py`, and `app.py`. Corpus measurements are limited to the facts summarized in the README and this guide.

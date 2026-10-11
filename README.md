# TraceRAG

**Vectorless retrieval that shows its work.**

TraceRAG turns a document into a section tree, walks that tree with reasoning instead of
cosine similarity, and returns three things with every answer: the answer, a numbered list
of the sections it came from, and a trace of every decision the retriever made along the way.
It then re-reads its own answer sentence by sentence and flags anything the cited sections do
not actually support.

No vector database. No embedding model. No chunk-and-hope. It runs end to end with no API key
at all, and gets better when you give it one.

![TraceRAG answering a question about a refund policy](docs/screenshot.png)

## Why

Similarity is not relevance. A question about *why* refunds are slow and a paragraph about
refund SLAs share most of their vocabulary, while the paragraph that actually answers the
question may share almost none of it. Embedding search optimises for the first thing and
quietly loses the second.

The bet this project makes is that the structure of a document already encodes most of what
you need. A heading like `6.1 Data retention` is a human-written label for the question that
section answers. If retrieval reads the structure instead of shredding it, it can find the
right section even when the prose inside uses completely different words.

## What it does differently

- **An auditable retrieval trace.** Every candidate the walker considered is recorded with its
  score, the scorer that produced it, whether it was kept or pruned, and why. The trace ships
  with the answer, so a wrong answer can be debugged by reading the decisions instead of
  guessing at an embedding space.
- **A per-sentence evidence ledger.** After the answer is written, each sentence is checked
  against the text of the sections it cites. Sentences with no citation, or with too little
  overlap against the cited text, are flagged rather than trusted.
- **Hybrid routing that degrades to offline.** A pure-Python BM25 router narrows each level
  first; a model only ever sees a shortlist of candidate sections, never the corpus. With no
  API key configured the model stage is skipped entirely and the system still answers, with
  citations, deterministically.
- **Structure-aware evidence packing.** Retrieved sections are packed into a token budget, so
  the answerer receives whole sections with their headings rather than arbitrary windows.
- **A built-in evaluation harness.** The tree walker is scored against a flat fixed-window
  chunking baseline on the same corpus, with the same scorer, so the comparison is measured
  rather than asserted.
- **Token and cost accounting** per query, per call.

## Benchmark

Run with no model configured, so the numbers are free, offline and reproducible
(`tracerag eval --mode lexical`, 24 questions, 3 documents, `top_k=5`):

| metric | tree walker | naive chunking |
| --- | ---: | ---: |
| recall@5, all 24 cases | **95.8%** | 37.5% |
| recall@5, 6 multi-hop cases | **83.3%** | 33.3% |
| evidence tokens sent to the answerer | **663** | 798 |
| citation support | 100% | n/a |
| mean nodes visited | 13.0 | n/a |
| mean latency | <1 ms | n/a |

Read this table with the caveat it deserves. The corpus in `data/corpus` is synthetic and
deliberately built so that headings carry vocabulary the body does not — a real and very
common property of handbooks, filings and API references, and precisely the case flat
similarity search is worst at. This measures one failure mode carefully; it is not a claim
that tree walking wins everywhere. The baseline is not handicapped either: it keeps the
heading lines in its flattened text, and a window that straddles a boundary counts as a hit
for the section it partly covers.

`citation support` is 100% offline for an uninteresting reason: the offline answerer quotes
its evidence verbatim, so the ledger is checking quotes against their source. Configure a
model and the number starts doing real work, because the ledger is then auditing generated
prose.

## How a query works

```
                       build once                         query time
  file ──► ingest ──► section tree ──► index.json
                     (hierarchical)                     question
                                         │                 │
                                         │                 ▼
                                         │        ┌─────────────────┐
                                         │        │  BM25 router    │  per level
                                         │        └────────┬────────┘
                                         │                 ▼
                                         │        ┌─────────────────┐
                                         └───────►│  model shortlist │  optional
                                                  └────────┬────────┘
                                                           ▼
                                                  selected sections
                                                           │
                                                           ▼
                        answer + [n] markers ◄──── answerer (model or extractive)
                                                           │
                                                           ▼
                                              evidence ledger audits each sentence
```

1. **Ingest** walks the source and records which outline path was in force for each block of
   text. Markdown headings, numbered headings, Chinese chapter markers and short unpunctuated
   title lines in extracted PDF text are all recognised. Headings are consumed as structure,
   not kept as body text.
2. **Build** assembles those outline paths into a tree, assigns stable hierarchical ids
   (`n1`, `n1.2`, `n1.2.3`), and splits any section larger than 4,000 characters into child
   windows so leaf granularity stays useful.
3. **Route** ranks each level's candidates with BM25 over `title + summary`, keeps a beam, and
   optionally asks a model to drop the ones that cannot answer the question. Ancestor scores
   propagate downwards, so a strong branch keeps its weaker children alive.
4. **Answer** packs the surviving sections into a token budget and asks a model for prose with
   inline `[n]` markers, or extracts the best-matching sentences when no key is configured.
5. **Audit** splits the answer into sentences, binds each `[n]` marker to the sentence it
   supports, and scores each sentence against the text of its cited sections.

### Modes

| mode | router | answerer | needs a key |
| --- | --- | --- | --- |
| `lexical` | BM25 only | extractive | no |
| `hybrid` | BM25 + model shortlist | model, extractive fallback | no |
| `reasoning` | model sees every candidate at each level | model, extractive fallback | no |

Every mode works offline. `hybrid` and `reasoning` simply do less when no key is present.

## Quickstart

```bash
git clone https://github.com/Supereggcat/tracerag
cd tracerag
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[server,dev]"
```

Index the bundled corpus and ask a question. No API key required:

```bash
tracerag index data/corpus
tracerag docs
tracerag ask "退款到账要多久?" --doc atlas-handbook-656d6142 --trace
```

Run the evaluation harness:

```bash
tracerag eval --mode lexical
tracerag eval --mode lexical --json
```

Start the web UI:

```bash
tracerag serve --port 8011
# then open http://127.0.0.1:8011
```

### Using a model

Any provider that speaks the OpenAI `/chat/completions` shape works. Nothing is required for
the offline modes.

```bash
# OpenAI
export TRACERAG_API_KEY=sk-...
export TRACERAG_MODEL=gpt-4o-mini

# DeepSeek, Moonshot, Qwen, OpenRouter, vLLM, Ollama, ...
export TRACERAG_API_KEY=...
export TRACERAG_BASE_URL=https://api.deepseek.com/v1
export TRACERAG_MODEL=deepseek-chat

tracerag ask "What protects me from a duplicate charge?" --doc orbit-api-notes-ddf28c68 --mode reasoning
```

`OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL` are read as fallbacks. The web UI shows
the active model in the top bar and falls back to `BM25` when none is configured.

## CLI

```
tracerag index <paths...>      ingest files or directories
tracerag docs                  list indexed documents
tracerag rm <doc_id>           drop one document
tracerag ask "<question>"      answer from one document
    --doc <doc_id>             required
    --mode lexical|hybrid|reasoning
    --top-k <n>                sections to keep (default 5)
    --trace                    print the full retrieval trace
    --json                     machine-readable output
tracerag eval                  score against the flat baseline
    --cases <file>             default eval/questions.jsonl
    --mode, --top-k, --json
tracerag serve                 run the web UI
    --host, --port, --reload
```

## HTTP API

| method | path | purpose |
| --- | --- | --- |
| `GET` | `/api/health` | status, active model, document count |
| `GET` | `/api/config` | whether a model is configured, available modes |
| `GET` | `/api/docs` | list indexed documents |
| `POST` | `/api/docs` | upload and index a document (multipart) |
| `GET` | `/api/docs/{id}` | document metadata and outline |
| `GET` | `/api/docs/{id}/tree` | the full section tree |
| `DELETE` | `/api/docs/{id}` | drop a document |
| `POST` | `/api/query` | ask a question, returns answer + ledger + trace + usage |

```bash
curl -s localhost:8011/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"退款到账要多久?","doc_id":"atlas-handbook-656d6142","mode":"hybrid","top_k":5}'
```

## Python

```python
from tracerag import Corpus, RetrievalConfig, ask, index_path

index_path("data/corpus", data_dir="data/index")
doc_id = Corpus("data/index").list_docs()[0].doc_id

answer = ask(
    "退款到账要多久?",
    doc_id,
    data_dir="data/index",
    config=RetrievalConfig(mode="lexical", max_selected=5),
)

print(answer.text)
print(f"supported: {answer.supported_ratio:.0%}")
for entry in answer.ledger:
    print(entry.supported, entry.support, entry.sentence)
for step in answer.trace:
    print(step.decision, step.node_id, round(step.score, 3), step.rationale)
```

## Supported inputs

Markdown, plain text, reStructuredText, HTML and PDF. PDFs need `pypdf`, which is a core
dependency. Scanned PDFs need OCR first; this project does not do OCR.

## Tests

```bash
pytest
```

33 tests covering tokenisation, outline detection, tree construction and id stability, window
splitting, retrieval in all three modes, graceful degradation when a model is unreachable,
context budgeting, ledger auditing, corpus persistence, the HTTP API and the evaluation
harness.

## Repository layout

```
src/tracerag/
  models.py      typed records shared across modules
  text.py        CJK-aware tokeniser, sentence splitter, token estimation
  ingest.py      files -> headed blocks
  tree.py        blocks -> section tree
  lexical.py     dependency-free BM25
  llm.py         OpenAI-compatible client on urllib
  retrieve.py    the tree walker and its trace
  answer.py      synthesis and the evidence ledger
  store.py       on-disk corpus
  pipeline.py    index_file / index_path / ask
  evaluate.py    tree walker vs flat chunking
  baseline.py    the flat chunking baseline
  webapp.py      FastAPI app
  cli.py         command line entry point
web/             no-build web UI
data/corpus/     three sample documents
eval/            24 evaluation questions
```

## 中文说明

**TraceRAG 是一个"不用向量库"的检索增强生成引擎。** 它把文档解析成章节树,让模型像人翻目录
一样逐层推理找到相关段落,而不是把文档切碎后做相似度搜索。

和常见做法相比,它多了三样东西:

1. **可审计的检索轨迹** —— 每个被考虑过的章节都记录了得分、打分器和"保留/剪枝"的理由,
   答错了可以直接读决策过程,而不是去猜向量空间里发生了什么。
2. **逐句证据台账** —— 答案写完后,每个句子都会拿回它引用的原文核对,没有引用或对不上的
   句子会被明确标红,而不是默认可信。
3. **零 API Key 也能跑** —— 纯 Python 的 BM25 负责每一层的初筛,没配模型时自动走抽取式
   回答,带着引用输出,完全离线、结果确定。

```bash
pip install -e ".[server,dev]"
tracerag index data/corpus
tracerag ask "退款到账要多久?" --doc atlas-handbook-656d6142 --trace
tracerag serve --port 8011
```

想接大模型时,任何兼容 OpenAI `/chat/completions` 的服务都可以,设置
`TRACERAG_API_KEY` / `TRACERAG_BASE_URL` / `TRACERAG_MODEL` 即可。

## Credits

- The idea of replacing a similarity index with a hierarchical document tree that a model
  navigates by reasoning was popularised by **PageIndex**
  (<https://github.com/VectifyAI/PageIndex>, MIT, Copyright (c) 2025 Vectify AI). This project
  is an independent re-implementation with a different architecture, a different feature set
  and no copied source code. See [NOTICE](NOTICE).
- The web UI vendors [Lucide](https://lucide.dev) for its icon set (ISC License).

## License

MIT. See [LICENSE](LICENSE).

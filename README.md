# Documentation claim auditor

Checks documentation promises against static Python call patterns. Runs offline,
never executes inspected source, performs no network requests, writes no files,
and sends no telemetry. Python 3.9 or later is required.

## Usage

```sh
pip install retrieval-augmented-generation-claim-audit
retrieval-augmented-generation-claim-audit audit .
python3 -m retrieval_augmented_generation_claim_audit audit . --json
retrieval-augmented-generation-claim-audit audit . --fail-on medium --ignore vendor --ignore build
retrieval-augmented-generation-claim-audit --version
```

The default directory is the current directory. Each ignore option accepts one
relative path prefix. The default failure threshold is high. Exit status is zero
when the threshold is not reached, two when it is reached, and one for usage or
I/O errors. JSON includes a verdict and findings with explanatory messages.
Verbose output includes inspected file counts and evidence paths.

Root README.md and Markdown files inside a docs directory are checked. Fenced
examples are skipped; inline code counts. Paragraphs are normalized and compared
against exact vocabulary. Python files are parsed without execution. Tests count
separately; fixtures, documentation, examples, and symbolic links are excluded
from production evidence. Invalid Python supplies no evidence.

```text
documented-retrieval-augmented-generation-claim-false HIGH
 documented-retrieval-augmented-generation-claim-false MEDIUM (test evidence only)
documented-no-retrieval-augmented-generation-claim-false HIGH
```

## Differentiation

The neighboring tools check distinct operations:

```text
semantic-search-claim-audit: retrieval alone, no generation step
text-generation-claim-audit: generation alone, no retrieval step
document-question-answering-claim-audit: pipeline instance call with no retriever
```

This auditor requires the combined operation, using paired calls within one
module or an explicitly named operation.

## Honest limitations

The following matching hazards describe the accepted boundaries verbatim:

```text
- The bare word "rag" is deliberately a
  RETRIEVALAUGMENTEDGENERATION phrase: in Python project docs it
  almost always names the technique. A doc that mentions a rag
  (cloth) or uses "rag" as an unrelated acronym is a false
  positive; the same accepted shape as other families' short
  phrases.
- "chat with your documents", "answers from documents", and
  "query your documents" are deliberately phrases: they can also
  describe a manual upload-and-prompt flow with no retrieval step,
  or a plain document viewer with a chat box. The tool flags the
  claim; only the code decides.
- "retrieves documents" and "retrieves context" are deliberately
  phrases: they can describe a plain database lookup feeding a
  template, not a language model. A doc that says so while its code
  never generates from the retrieved material is exactly the false
  claim this family catches.
- "semantic retrieval" and "vector retrieval" are deliberately
  phrases: they also match retrieval-only tools (the
  semantic-search family's domain). A doc that claims retrieval
  augmented generation with retrieval-only code fails here on
  purpose: retrieval without generation is not
  retrieval-augmented generation.
- "grounded in your data" is deliberately a phrase: a doc can
  describe a dashboard that cites sources without any generation
  step.
- The manual pair's `retrieve` attribute is a common method name;
  it counts only together with a generation call in the same
  module, so a lone `cache.retrieve()` never counts.
- A hand-rolled pipeline that retrieves with differently-named
  methods (for example `search_index` plus a
  `client.chat.completions.create` call) is NOT detected: the
  manual route covers only the listed method names. The tool covers
  the LangChain / LlamaIndex / Haystack routes plus the listed
  manual pair only.
- A chain or engine that is only constructed and never invoked is
  NOT counted: building a pipeline is not running it, per the
  family rule that the construction must be paired with an
  invoke / query / run call.
- A LlamaIndex `as_retriever()` call alone is retrieval only, not
  retrieval-augmented generation: it must be paired with a query
  or chat call on a query engine.
- As with every family: static pattern matching, not semantic
  judgement; dead code counts; alias chains the tracer cannot follow
  are missed; exact phrases only, not semantics; denial handling
  depends on the denial phrase list; each pair must sit in a single
  module.

```

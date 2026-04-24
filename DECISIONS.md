# Architectural Decisions

Key trade-off decisions made in building this project, with the reasoning behind each.

---

## 1. Anthropic for chat, OpenAI for embeddings — not one provider for both

**Decision:** Use Anthropic Claude for answer generation and query rewriting. Use OpenAI `text-embedding-3-large` for vector embeddings. Do not use Anthropic for embeddings.

**Why:** Anthropic has no embeddings API. This isn't a gap that can be papered over with a shim — there's no equivalent service. OpenAI's `text-embedding-3-large` at 1024 dimensions is the right choice for this use case: it's well-benchmarked on retrieval tasks, the truncated 1024-dimension variant cuts storage cost by ~60% vs. the full 3072-dimension output with minimal quality loss, and it's the same model used in the source catalog (preserving embedding compatibility).

The split-provider setup is deliberate, not a workaround. Each provider does one thing here and does it well.

**Trade-off accepted:** Two API keys, two billing accounts. For a production deployment this would be managed via a secrets manager (AWS Secrets Manager, Azure Key Vault). The `.env` pattern here is dev-only.

---

## 2. Replaced the OpenAI Agents SDK with direct Anthropic SDK calls

**Decision:** Remove `openai-agents` (`Agent`, `Runner`, `OpenAIResponsesModel`, `function_tool`) entirely. Replace with `anthropic.AsyncAnthropic().messages.create()` and `.messages.stream()`.

**Why:** The original template used OpenAI's newer Responses API through the Agents SDK abstraction. That API has no Anthropic equivalent — `ResponseFunctionToolCall`, `ResponseInputItemParam`, and `OpenAIResponsesModel` are OpenAI-specific constructs. The Agents SDK could not be swapped to an Anthropic backend; it had to come out entirely.

More importantly: the abstraction layer was doing less than it appeared. `Runner.run()` was essentially a thin wrapper around a single API call per turn. Removing it and calling the Anthropic SDK directly made the control flow explicit, easier to debug, and easier to extend (e.g., adding streaming required one `async with messages.stream()` block, not a framework configuration change).

**Trade-off accepted:** The original template's multi-agent handoff pattern (search agent → answer agent) is no longer a distinct architectural feature — both flows now call the same `messages.create()`. For a more complex multi-agent system this would be worth re-introducing, but for a two-step RAG pipeline the separation added ceremony without adding safety.

---

## 3. Hybrid search (pgvector + full-text, fused via RRF) over vector-only

**Decision:** Run both vector similarity search and PostgreSQL full-text search in parallel and fuse results using Reciprocal Rank Fusion. Do not use vector search alone.

**Why:** Government RFQ queries fall into two patterns: semantic ("what do we have for mission-critical analytics?") and specific ("show me our NAICS 541511 capabilities"). Vector search handles the first case well but degrades on the second — exact NAICS codes and specific capability names often appear as low-similarity neighbors in embedding space even when they're an exact lexical match. Full-text search covers that gap without requiring a separate re-ranker model.

RRF was chosen over weighted score combination because it's rank-based (position matters, not raw scores) and doesn't require tuning score weights that would vary with embedding model choice. It's also simple enough to explain to a client or auditor — position 1 from either system gets the highest fusion weight.

**Trade-off accepted:** Two database queries per search request instead of one. At the catalog sizes relevant to this domain (hundreds to low thousands of capabilities) this is negligible. At very large scale (millions of rows) the full-text query would need GIN index tuning, and the RRF merge would need to happen at the database layer rather than in Python.

---

## 4. Tool use with forced tool_choice for query rewriting

**Decision:** In the advanced flow, use Claude's tool use API with `tool_choice: {type: "tool", name: "search_database"}` to extract structured search arguments from natural-language queries. Do not use few-shot prompting to produce a structured output string and parse it.

**Why:** Few-shot prompting for structured extraction is fragile — it relies on the model producing output in the exact format the parser expects, and failures are silent (a malformed filter just gets dropped). Tool use with a typed input schema (`input_schema` in Anthropic's format) makes the extraction type-constrained: the model cannot return a filter with a missing required field or an invalid comparison operator. If extraction fails, the API returns an error rather than a silently wrong answer.

`tool_choice: forced` removes the model's discretion about whether to call the tool. For query rewriting, that discretion is unwanted — every query should go through the extraction step.

**Trade-off accepted:** One additional API call per query in the advanced flow (query rewriting + answer generation vs. just answer generation in the simple flow). The simple flow exists for cases where that latency or cost is unacceptable. The UI toggle between simple and advanced flow exposes this trade-off to the user.

# MediBot Learning Notes

This document records the backend and RAG concepts explored while building MediBot. It intentionally excludes FastAPI and UI topics.

## 1. SQLite and the data-access layer

SQLite is a file-based relational database. MediBot reads the supplied `mediassist.db` through a repository layer so database code stays separate from RAG and business logic.

The SQL RAG repository accepts only validated `SELECT` statements and only the `claims` and `maintenance_tickets` tables. This is defence in depth: an LLM may propose SQL, but application code validates it before SQLite executes it.

## 2. Configuration and secrets

`backend/.env` stores local settings such as provider API keys, model IDs, the JWT signing secret, Qdrant URL, and retrieval limits. It is ignored by Git.

Environment variables supplied when the process starts take priority. `.env` supplies local defaults when those variables are absent.

## 3. Roles and access control

Roles control capabilities as well as document access. For example, only `billing_executive` and `admin` can enter SQL RAG; document chunks carry `access_roles` metadata for Qdrant filtering.

This means routing and authorization are different responsibilities. A router selects a workflow; an authorization rule decides whether the user may use it or see a document.

## 4. Python interfaces with abstract base classes

Python uses abstract base classes (for example, `LLMClient`, `RoutingStrategy`, `HybridRetriever`, and `Reranker`) as Java-like interfaces.

Unlike Java, Python does not automatically discover an interface from matching method names. A concrete class explicitly extends the abstract base class. This gives a common contract while allowing different implementations later.

## 5. Keyword routing

The first router is intentionally simple and explainable. It chooses SQL RAG only when an authorized role asks an analytical question containing a database-related term. Other questions go to Hybrid RAG.

Because callers depend on `RoutingStrategy`, a later semantic-routing or LLM-routing implementation can replace the keyword strategy without changing callers.

## 6. LLM provider and model abstraction

`LLMClient` hides provider SDK details. Configuration chooses the provider group with `REQUIRED_LLM_MODEL_GROUP` and selects separate model IDs for SQL testing, SQL generation, and answer generation.

Today the concrete provider is Groq. The interface keeps the project ready for an OpenAI or Claude adapter without changing SQL RAG or Hybrid RAG services.

## 7. SQL RAG flow

SQL RAG has three stages:

1. The LLM converts an analytical question into SQLite SQL.
2. Application code cleans and validates the generated SQL, then the repository executes a read-only query.
3. The LLM converts the returned rows into a natural-language answer.

The model is not trusted to enforce table permissions or read-only access; those guarantees belong to code.

## 8. Document ingestion

Ingestion is a separate explicit command, not a user-facing runtime action. Docling reads PDFs and preserves document structure. Its hierarchical chunker creates smaller chunks while retaining section context.

LangChain `Document` objects provide a standard container for chunk text and metadata. The ingestion process stores each chunk with document identity, collection, role access, section title, chunk type, document hash, and index version.

## 9. Embeddings, dense vectors, and sparse vectors

A dense vector represents semantic meaning as a fixed-length list of numbers. It helps match questions and text that use different words with related meaning.

A sparse vector represents important tokens and their weights. BM25-style sparse retrieval helps preserve exact terms such as `MRSA`, equipment IDs, diagnosis codes, or policy names.

Dense and sparse retrieval complement each other: semantic search handles meaning, while sparse search is strong for exact domain vocabulary.

## 10. Qdrant hybrid retrieval and RRF

Qdrant stores vectors plus payload metadata. MediBot uses one versioned collection and applies the role filter inside Qdrant before candidate chunks return to the application.

Hybrid retrieval runs dense and sparse search together. Reciprocal Rank Fusion (RRF) combines their rank positions into a `fusion_score`; it is a ranking value, not an answer-confidence percentage.

## 11. Incremental indexing

Ingestion calculates a document hash and records an index version. If the source document, expected chunk count, and index version are unchanged, the document is skipped on a later ingestion run.

Changing chunking, embedding models, or an index contract should use a new index version or an intentional reindex. This avoids silently mixing incompatible vectors in one collection.

## 12. Cross-encoder reranking

Hybrid retrieval first returns a broader candidate set (currently 10). A cross-encoder then evaluates each full `(question, chunk)` pair and keeps a smaller final context (currently 3).

The cross-encoder score is a relevance ordering value. Higher is better, and the value can be negative. It is not a probability.

## 13. Grounded answers and trusted citations

The answer model receives only role-authorized, reranked chunk text. The prompt instructs it to answer only from that context.

Source citations are not accepted from the LLM response. MediBot builds them from trusted Qdrant metadata—document name, section title, and collection—after reranking.

## 14. Authentication foundation

Demo users are provisioned explicitly in a separate authentication SQLite database. Passwords are stored as bcrypt hashes, never plaintext.

A successful login creates a signed, expiring JWT containing a verified user ID, username, and role. The signing secret protects the token signature; it is not used to encrypt passwords.

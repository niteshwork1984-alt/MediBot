# MediBot Decisions

| Decision | Why |
| --- | --- |
| Use SQLite for assignment data | The supplied data is already a SQLite database; it keeps local setup simple. |
| Keep the repository layer read-only | LLM-generated SQL must not gain write access to assignment data. |
| Allow SQL RAG only for `billing_executive` and `admin` | Claims and maintenance analytics are restricted business data. |
| Allow only `claims` and `maintenance_tickets` for SQL RAG | Limits the SQL surface area to the assignment’s approved analytical tables. |
| Start with keyword routing | It is transparent for learning and can be tested easily. |
| Define routing behind `RoutingStrategy` | A semantic or LLM router can replace keywords later without changing callers. |
| Define LLM access behind `LLMClient` | Provider SDK code stays isolated; new Groq, OpenAI, or Claude adapters can be added later. |
| Select provider/models through environment configuration | Models can change by environment without code edits. Separate SQL and answer model settings support cost/quality experiments. |
| Keep ingestion separate from runtime question answering | Indexing is a controlled batch activity; users do not trigger it from normal chat. |
| Use Docling hierarchical chunking | PDF structure and section context are retained while text is broken into retrievable pieces. |
| Use LangChain `Document` and Qdrant hybrid mode | This meets the assignment’s LangChain requirement while using Qdrant for vector storage and retrieval. |
| Use one versioned Qdrant collection with payload metadata | Collection, roles, document hash, and index version travel with every chunk and support filtering and reindexing. |
| Filter roles inside Qdrant | Unauthorized chunks do not become application retrieval candidates. |
| Use dense plus BM25 sparse search | Dense search captures meaning; sparse search preserves exact medical and operational terms. |
| Combine dense/sparse results with RRF | Rank fusion combines both retrieval signals without treating their raw scores as directly comparable. |
| Limit retrieval to 10 candidates and rerank to 3 | Keeps retrieval work bounded and sends a focused context to the answer model. |
| Use a FastEmbed cross-encoder reranker | It scores full question/chunk pairs locally and improves relevance after broad retrieval. |
| Build citations from metadata rather than LLM output | The answer model cannot invent source references. |
| Store users separately from assignment data | Authentication lifecycle and supplied clinical/billing data have distinct responsibilities. |
| Use bcrypt for demo passwords | Passwords are stored as salted hashes rather than plaintext. |
| Use signed expiring JWTs for sessions | The server can verify user identity and role without accepting a client-provided role. |

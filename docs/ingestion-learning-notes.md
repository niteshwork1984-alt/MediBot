# Ingestion learning notes

## Separate process, shared project

`python -m scripts.ingest_documents` is a standalone batch command. It is not started by FastAPI and is not called from the UI. The API will only query an already-populated Qdrant collection later.

## Chunk contract

Every Qdrant point produced by ingestion contains:

```text
id                 stable UUID point identifier
text               contextualized heading + chunk text
dense vector       semantic embedding
sparse vector      exact-term embedding
source_document    original filename
document_key       path relative to source root; used for replacement tracking
collection         assignment collection name
access_roles       roles stored for later Qdrant RBAC filtering
section_title      Docling heading path
chunk_type         text, table, heading, or code
document_hash      SHA-256 source-file fingerprint
index_version      chunking/embedding contract version
```

## Incremental indexing

The command parses a document into the expected number of chunks, then asks Qdrant how many stored points match its `document_key`, `document_hash`, and `index_version`.

| State | Result |
| --- | --- |
| Stored count equals expected count | Skip embedding and Qdrant write. |
| File content changed | Delete existing points for that file, then embed and upsert replacement points. |
| New file | Embed and upsert points. |
| `--force-reindex` | Replace every discovered file even if it appears current. |

Qdrant is therefore the persistent record of indexed chunks. A stable UUID makes repeated upserts idempotent, while deleting a changed document first removes obsolete chunks.

## Run the first index

From `MediBot/backend`:

```bash
.venv/bin/python -m scripts.ingest_documents \
  --source-root ../../doc/Medibot_Assignment_Resources/mediassist_data
```

The first run downloads the selected local embedding-model weights. It then creates `medibot_documents_v1` in local Qdrant and stores the generated points.

## Re-index after an embedding or chunking change

Use a new collection and version; do not mix two embedding models in one collection:

```bash
.venv/bin/python -m scripts.ingest_documents \
  --source-root ../../doc/Medibot_Assignment_Resources/mediassist_data \
  --index-version v2 \
  --collection medibot_documents_v2
```

After validating the new collection, set `QDRANT_COLLECTION=medibot_documents_v2` for future retrieval code.

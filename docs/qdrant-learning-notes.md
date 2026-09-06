# Qdrant learning notes

## Why Qdrant exists in MediBot

SQLite stores structured rows such as claims and maintenance tickets. Qdrant stores document chunks in a form that supports semantic and keyword retrieval. Hybrid RAG searches Qdrant; SQL RAG continues to use SQLite.

## Core concepts

| Concept | Meaning in MediBot |
| --- | --- |
| Collection | A Qdrant container, similar to a database table. We will create one `medibot_documents` collection. |
| Point | One stored document chunk, similar to one table row. |
| Point ID | A stable unique identifier for a chunk. |
| Dense vector | A list of embedding numbers representing semantic meaning. |
| Sparse vector | Term-and-weight values representing exact keyword importance for BM25-style search. |
| Payload | Filterable metadata stored next to a vector, such as `access_roles` and `source_document`. |
| Upsert | Create a new point or replace a point with the same ID. This makes re-ingestion safe. |
| Query filter | A condition evaluated by Qdrant before results are returned. We will filter `access_roles` here for RBAC. |
| Similarity metric | The mathematical measure used to rank vectors. The embedding model determines the correct metric, often cosine similarity. |

## What Qdrant will store

Each Docling/HybridChunker chunk becomes one Qdrant point with two vectors and metadata:

```text
Point
├── id: stable chunk ID
├── vectors
│   ├── dense: semantic embedding
│   └── sparse: keyword/BM25 representation
└── payload
    ├── text
    ├── source_document
    ├── collection
    ├── access_roles
    ├── section_title
    └── chunk_type
```

## Persistence

`docker-compose.yml` maps the local `data/qdrant_storage/` directory to `/qdrant/storage` inside the container. Qdrant writes collections and indexes there. Stopping, restarting, or recreating the container keeps the data as long as the local directory remains.

## Local commands

From the project root:

```bash
docker compose up -d qdrant
docker compose ps
docker compose down
```

If Docker Desktop provides the legacy Compose command instead, replace `docker compose` with `docker-compose`.

Qdrant's local REST endpoint and dashboard use `http://localhost:6333`. The backend will later use this URL through `QDRANT_URL` configuration.

## Security rule to remember

The embedding text helps find relevant chunks, but it does not enforce permissions. The Qdrant query must include a payload filter for the authenticated role. Therefore, chunks for a billing document cannot be returned at all to a nurse query.

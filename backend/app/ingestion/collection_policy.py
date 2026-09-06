"""Explicit mapping between supplied document folders and retrieval permissions."""

from pathlib import Path


# This constant lists every role defined by the MediBot assignment.
ALL_ROLES = frozenset({"doctor", "nurse", "billing_executive", "technician", "admin"})

# This mapping defines collection access without relying on user-provided document content.
COLLECTION_ACCESS_ROLES: dict[str, frozenset[str]] = {
    "general": ALL_ROLES,
    "clinical": frozenset({"doctor", "admin"}),
    "nursing": frozenset({"doctor", "nurse", "admin"}),
    "billing": frozenset({"billing_executive", "admin"}),
    "equipment": frozenset({"technician", "admin"}),
}


# This function reads the collection from a file's first directory below the source root.
def collection_for_source(source_root: Path, source_path: Path) -> str:
    """Return the approved collection owning a source document."""
    try:
        relative_path = source_path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(f"Source file is outside source root: {source_path}") from error

    if len(relative_path.parts) < 2:
        raise ValueError(
            "Each source document must be inside an approved collection directory."
        )

    collection = relative_path.parts[0]
    if collection not in COLLECTION_ACCESS_ROLES:
        raise ValueError(f"Unsupported document collection: {collection}")
    return collection


# This function returns the roles Qdrant must store in every point payload for a collection.
def access_roles_for_collection(collection: str) -> list[str]:
    """Return a stable, JSON-friendly role list for one approved collection."""
    try:
        return sorted(COLLECTION_ACCESS_ROLES[collection])
    except KeyError as error:
        raise ValueError(f"Unsupported document collection: {collection}") from error

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def _infer_text_metadata(path: Path, content: str) -> dict[str, str]:
    metadata: dict[str, str] = {"source": str(path), "extension": path.suffix.lower()}
    stem = path.stem.lower()

    if stem == "python_intro":
        metadata.update({"lang": "en", "topic": "python"})
    elif stem == "vector_store_notes":
        metadata.update({"lang": "en", "topic": "vector-store"})
    elif stem == "rag_system_design":
        metadata.update({"lang": "en", "topic": "rag"})
    elif stem == "customer_support_playbook":
        metadata.update({"lang": "en", "topic": "support"})
    elif stem == "chunking_experiment_report":
        metadata.update({"lang": "en", "topic": "chunking"})
    elif stem == "vi_retrieval_notes":
        metadata.update({"lang": "vi", "topic": "retrieval"})

    if stem.startswith("fed_fomc_statement_"):
        metadata.update(
            {
                "lang": "en",
                "topic": "monetary-policy",
                "category": "monetary_policy",
                "document_type": "government_press_release",
                "type": "Statement",
                "publisher": "Federal Reserve",
            }
        )
        date_match = re.search(r"(\d{4})_(\d{2})_(\d{2})$", stem)
        if date_match:
            metadata["date"] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

    if stem.startswith("bls_cpi_"):
        metadata.update(
            {
                "lang": "en",
                "topic": "inflation",
                "category": "inflation",
                "document_type": "government_press_release",
                "event_type": "CPI Report",
                "publisher": "Bureau of Labor Statistics",
            }
        )

    date_match = re.search(r"^Date:\s*(.+)$", content, flags=re.MULTILINE)
    if date_match:
        metadata["date"] = date_match.group(1).strip()

    reference_match = re.search(r"^Reference Period:\s*(.+)$", content, flags=re.MULTILINE)
    if reference_match:
        metadata["reference_period"] = reference_match.group(1).strip()

    rate_match = re.search(r"^Interest Rate:\s*(.+)$", content, flags=re.MULTILINE)
    if rate_match:
        metadata["interest_rate"] = rate_match.group(1).strip()

    return metadata


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt", ".csv"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt, .csv)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader):
                    cleaned_row = {
                        key.strip(): str(value).strip()
                        for key, value in row.items()
                        if key and value and str(value).strip()
                    }
                    if not cleaned_row:
                        continue

                    metadata = {
                        "source": str(path),
                        "extension": path.suffix.lower(),
                        "row_index": index,
                    }
                    for field_name in ("Date", "Type", "Reference_Period", "Event_Type"):
                        if field_name in cleaned_row:
                            metadata[field_name.lower()] = cleaned_row[field_name]

                    content = " | ".join(f"{key}: {value}" for key, value in cleaned_row.items())
                    documents.append(
                        Document(
                            id=f"{path.stem}_{index}",
                            content=content,
                            metadata=metadata,
                        )
                    )
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata=_infer_text_metadata(path, content),
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt, .csv")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())

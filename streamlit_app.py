from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from main import SAMPLE_FILES, demo_llm, load_documents_from_files
from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)

st.set_page_config(page_title="Demo Lab 7", layout="wide")

FINANCIAL_DATA_FILES = [
    "data/bls_cpi_december_2025.txt",
    "data/fed_fomc_statement_2025_07_30.txt",
    "data/fed_fomc_statement_2025_09_17.txt",
    "data/fed_fomc_statement_2025_10_29.txt",
    "data/fed_fomc_statement_2025_12_10.txt",
]

DATA_DIR = Path("data")
SUMMARY_FILE = DATA_DIR / "final_documents_metadata.json"
SCHEMA_FILE = DATA_DIR / "final_financial_metadata.json"

STRATEGIES = [
    "Raw documents",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
]

STRATEGY_LABELS = {
    "Raw documents": "Không chia nhỏ",
    "FixedSizeChunker": "Chia đoạn cố định",
    "SentenceChunker": "Chia theo câu",
    "RecursiveChunker": "Chia đệ quy",
}

QUERY_OPTIONS = [
    "What was the CPI year-over-year change in December 2025?",
    "What was the federal funds rate decision in July 2025?",
    "What drove the monthly CPI increase in December 2025?",
    "How did the Fed describe the labor market in its 2025 statements?",
    "What interest rate did the Fed set in December 2025?",
]

FILTER_PRESETS: dict[str, dict[str, Any] | None] = {
    "Không dùng bộ lọc": None,
    "Lạm phát (CPI)": {"category": "inflation"},
    "Chính sách tiền tệ": {"category": "monetary_policy"},
    "CPI tháng 12/2025": {"category": "inflation", "reference_period": "DECEMBER 2025"},
    "FOMC tháng 7/2025": {"category": "monetary_policy", "date": "2025-07-30"},
    "FOMC tháng 12/2025": {"category": "monetary_policy", "date": "2025-12-10"},
}

SAMPLE_BENCHMARKS = [
    {
        "query": "What is a vector store?",
        "expected_phrases": ["storage layer", "semantic search", "embeddings"],
        "metadata_filter": None,
    },
    {
        "query": "Why is recursive chunking often useful?",
        "expected_phrases": ["preserve", "separator", "context"],
        "metadata_filter": None,
    },
    {
        "query": "When should a support assistant escalate?",
        "expected_phrases": ["escalate", "support"],
        "metadata_filter": {"topic": "support"},
    },
    {
        "query": "Trong tài liệu tiếng Việt, metadata giúp gì cho retrieval?",
        "expected_phrases": ["metadata", "ngôn ngữ", "lọc"],
        "metadata_filter": {"lang": "vi"},
    },
    {
        "query": "Python is commonly used for what kinds of tasks?",
        "expected_phrases": ["automation", "machine learning", "data analysis"],
        "metadata_filter": {"topic": "python"},
    },
]

FINANCIAL_BENCHMARKS = [
    {
        "query": "What interest rate range did the Fed set on December 10, 2025?",
        "expected_phrases": ["3-1/2 to 3-3/4 percent", "2025-12-10"],
        "metadata_filter": {"date": "2025-12-10"},
    },
    {
        "query": "What was the Year-over-Year CPI change in December 2025?",
        "expected_phrases": ["2.7%", "DECEMBER 2025"],
        "metadata_filter": {"reference_period": "DECEMBER 2025"},
    },
]


@st.cache_data(show_spinner=False)
def get_available_files() -> list[str]:
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return [
        path.as_posix()
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv"}
    ]


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_selected_documents(file_paths: tuple[str, ...]) -> list[Document]:
    return load_documents_from_files(list(file_paths))


@st.cache_data(show_spinner=False)
def prepare_documents(
    file_paths: tuple[str, ...],
    strategy_name: str,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
) -> list[Document]:
    documents = load_selected_documents(file_paths)
    if strategy_name == "Raw documents":
        return documents

    if strategy_name == "FixedSizeChunker":
        chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy_name == "SentenceChunker":
        chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size)

    chunked_docs: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        extension = str(metadata.get("extension", "")).lower()

        # CSV rows are already good retrieval units, so keep them intact.
        if extension == ".csv":
            metadata.update({"parent_id": doc.id, "chunk_index": 0, "strategy": strategy_name})
            chunked_docs.append(Document(id=f"{doc.id}__row", content=doc.content, metadata=metadata))
            continue

        pieces = chunker.chunk(doc.content)
        if not pieces:
            continue

        for index, piece in enumerate(pieces):
            piece_metadata = dict(metadata)
            piece_metadata.update({"parent_id": doc.id, "chunk_index": index, "strategy": strategy_name})
            chunked_docs.append(
                Document(
                    id=f"{doc.id}__{strategy_name.lower()}__{index}",
                    content=piece,
                    metadata=piece_metadata,
                )
            )
    return chunked_docs


def build_store(documents: list[Document], strategy_name: str, use_chroma: bool) -> EmbeddingStore:
    store = EmbeddingStore(
        collection_name=f"streamlit_{strategy_name.lower().replace(' ', '_')}_{len(documents)}",
        prefer_chroma=use_chroma,
        persist_directory=".chroma",
    )
    store.add_documents(documents)
    return store


def run_query(
    file_paths: tuple[str, ...],
    strategy_name: str,
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
    use_chroma: bool,
) -> tuple[list[Document], list[dict[str, Any]], str, str]:
    prepared_docs = prepare_documents(file_paths, strategy_name, chunk_size, overlap, max_sentences)
    store = build_store(prepared_docs, strategy_name, use_chroma=use_chroma)
    if metadata_filter:
        results = store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
    else:
        results = store.search(query, top_k=top_k)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    answer = agent.answer(query, top_k=top_k)
    return prepared_docs, results, answer, store.backend_name


def benchmark_strategy(
    file_paths: tuple[str, ...],
    strategy_name: str,
    cases: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    max_sentences: int,
    top_k: int,
    use_chroma: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_docs = prepare_documents(file_paths, strategy_name, chunk_size, overlap, max_sentences)
    store = build_store(prepared_docs, strategy_name, use_chroma=use_chroma)

    detail_rows: list[dict[str, Any]] = []
    passed = 0
    total_top_score = 0.0

    for case in cases:
        metadata_filter = case.get("metadata_filter")
        if metadata_filter:
            results = store.search_with_filter(case["query"], top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = store.search(case["query"], top_k=top_k)

        joined_content = " ".join(item["content"] for item in results).lower()
        matched = any(phrase.lower() in joined_content for phrase in case["expected_phrases"])
        passed += int(matched)

        top_result = results[0] if results else {}
        total_top_score += float(top_result.get("score", 0.0))
        metadata = top_result.get("metadata", {}) if top_result else {}

        detail_rows.append(
            {
                "Câu hỏi": case["query"],
                "Bộ lọc": json.dumps(metadata_filter, ensure_ascii=False) if metadata_filter else "-",
                "Kết quả": "✅" if matched else "❌",
                "Điểm top-1": round(float(top_result.get("score", 0.0)), 4),
                "Nguồn top-1": metadata.get("source", "-"),
            }
        )

    summary = {
        "Cách làm": STRATEGY_LABELS.get(strategy_name, strategy_name),
        "Backend": store.backend_name,
        "Số tài liệu/chunks": len(prepared_docs),
        "Số câu khớp": f"{passed}/{len(cases)}",
        "Tỉ lệ đúng (%)": round((passed / len(cases)) * 100, 1) if cases else 0.0,
        "Điểm TB top-1": round(total_top_score / len(cases), 4) if cases else 0.0,
    }
    return summary, detail_rows


def preview(text: str, limit: int = 220) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}..."


def parse_filter_text(raw_text: str) -> dict[str, Any] | None:
    if not raw_text.strip():
        return None
    return json.loads(raw_text)


st.title("🎯 Demo Lab 7 — Hệ thống Retrieval & Benchmark")
st.caption(
    "Giao diện rút gọn bằng tiếng Việt để phục vụ thuyết trình: giới thiệu dữ liệu, so sánh cách làm, tra cứu nhanh và benchmark."
)

available_files = get_available_files()
default_files = [path for path in FINANCIAL_DATA_FILES if path in available_files] or [path for path in SAMPLE_FILES if path in available_files] or available_files[:3]

with st.sidebar:
    st.header("⚙️ Tùy chỉnh nhanh")
    st.caption("Dùng các lựa chọn dưới đây để chạy demo nhanh khi thuyết trình.")
    selected_files = st.multiselect(
        "Chọn file dữ liệu",
        options=available_files,
        default=default_files,
    )
    chunk_size = st.slider("Kích thước chunk", min_value=120, max_value=600, value=220, step=20)
    overlap = st.slider("Độ chồng lặp", min_value=0, max_value=120, value=50, step=10)
    max_sentences = st.slider("Số câu tối đa / chunk", min_value=1, max_value=8, value=3)
    top_k = st.slider("Số kết quả hiển thị", min_value=1, max_value=5, value=3)
    use_chroma = st.checkbox("Dùng ChromaDB nếu có", value=True)
    st.info(
        "Ứng dụng có thể so sánh `in-memory` và `ChromaDB`. Hiện embedding mặc định là `_mock_embed`, "
        "nên phần benchmark phù hợp nhất để minh họa cách chia nhỏ và metadata filtering."
    )

if not selected_files:
    st.warning("Hãy chọn ít nhất một file trong sidebar để bắt đầu.")
    st.stop()

file_tuple = tuple(selected_files)
documents = load_selected_documents(file_tuple)
financial_preview_files = tuple(path for path in FINANCIAL_DATA_FILES if path in available_files)
financial_documents = load_selected_documents(financial_preview_files) if financial_preview_files else []
summary_data = load_json_file(str(SUMMARY_FILE))
schema_data = load_json_file(str(SCHEMA_FILE))

st.markdown("---")
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("Số tài liệu đã nạp", len(documents))
metrics_col2.metric("Số file đã chọn", len(selected_files))
metrics_col3.metric("Tổng số ký tự", sum(len(doc.content) for doc in documents))


data_intro_tab, overview_tab, chunk_tab, search_tab, benchmark_tab = st.tabs(
    ["Giới thiệu dữ liệu", "Tổng quan", "So sánh chunking", "Tra cứu nhanh", "Benchmark"]
)

with data_intro_tab:
    st.subheader("Giới thiệu tổng quát về bộ dữ liệu")

    intro_col1, intro_col2, intro_col3, intro_col4 = st.columns(4)
    intro_col1.metric("Domain", summary_data.get("domain", "Financial Data"))
    intro_col2.metric("Số tài liệu", summary_data.get("total_documents", len(financial_documents)))
    intro_col3.metric(
        "Tổng ký tự",
        f"{summary_data.get('total_characters', sum(len(doc.content) for doc in financial_documents)):,}",
    )
    intro_col4.metric("Nguồn dữ liệu", len(summary_data.get("sources", [])) or 2)

    desc_col1, desc_col2 = st.columns([1.1, 1])
    with desc_col1:
        st.markdown(
            """
Bộ dữ liệu này tập trung vào **thông tin tài chính và kinh tế vĩ mô của Mỹ** từ các nguồn công khai, đáng tin cậy.

Nó phù hợp để minh họa:
- dữ liệu **lạm phát (CPI)** từ BLS
- các **thông cáo FOMC** từ Federal Reserve
- câu hỏi cần vừa tra cứu **số liệu chính xác** vừa cần **ngữ cảnh giải thích**
            """
        )
    with desc_col2:
        st.markdown(
            """
**Vì sao bộ dữ liệu này phù hợp cho bài lab?**
- Dễ so sánh các **chiến lược chunking**
- Dễ thử **metadata filtering** như `date`, `category`, `reference_period`
- Có thể demo rõ sự khác biệt giữa **in-memory** và **ChromaDB**
- Thuận tiện để thuyết trình vì dữ liệu thực tế và dễ kiểm chứng
            """
        )

    sources = summary_data.get("sources", [])
    categories = summary_data.get("categories", [])
    if sources or categories:
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown("**Nguồn dữ liệu**")
            for source in sources:
                st.markdown(f"- {source}")
        with meta_col2:
            st.markdown("**Nhóm chủ đề**")
            for category in categories:
                st.markdown(f"- `{category}`")

    document_rows = [
        {
            "Tên file": item.get("filename", "-"),
            "Nguồn": item.get("source", "-"),
            "Danh mục": item.get("category", "-"),
            "Số ký tự": item.get("chars", 0),
            "Chỉ số chính": ", ".join(item.get("indicators", [])),
        }
        for item in summary_data.get("documents", [])
    ]
    if document_rows:
        st.markdown("**Danh sách tài liệu**")
        st.dataframe(document_rows, width="stretch")

    metadata_schema = schema_data.get("metadata_schema", {})
    if metadata_schema:
        req_col, opt_col = st.columns(2)
        with req_col:
            st.markdown("**Metadata bắt buộc**")
            for field in metadata_schema.get("required_fields", []):
                st.markdown(f"- `{field}`")
        with opt_col:
            st.markdown("**Metadata tùy chọn**")
            for field in metadata_schema.get("optional_fields", []):
                st.markdown(f"- `{field}`")

    st.markdown("**Câu hỏi demo gợi ý**")
    st.markdown(
        """
1. `What was the CPI year-over-year change in December 2025?`
2. `What was the federal funds rate decision in July 2025?`
3. `What drove the monthly CPI increase in December 2025?`
4. `How did the Fed describe the labor market in its 2025 statements?`
5. `What interest rate did the Fed set in December 2025?`
        """
    )

    if financial_documents:
        selected_intro_doc_id = st.selectbox(
            "Chọn tài liệu để xem nhanh",
            options=[doc.id for doc in financial_documents],
            key="data_intro_doc",
        )
        selected_intro_doc = next(doc for doc in financial_documents if doc.id == selected_intro_doc_id)
        preview_col1, preview_col2 = st.columns([1.2, 0.8])
        with preview_col1:
            st.markdown(f"**Nguồn:** `{selected_intro_doc.metadata.get('source', '-')}`")
            st.write(preview(selected_intro_doc.content, limit=1500))
        with preview_col2:
            st.markdown("**Metadata**")
            st.json(selected_intro_doc.metadata, expanded=True)

with overview_tab:
    st.subheader("Tổng quan các file đang chọn")
    st.dataframe(
        [
            {
                "Mã": doc.id,
                "Nguồn": doc.metadata.get("source", "-"),
                "Loại file": doc.metadata.get("extension", "-"),
                "Số ký tự": len(doc.content),
            }
            for doc in documents[:200]
        ],
        width="stretch",
    )
    with st.expander("Xem nhanh tài liệu đầu tiên"):
        first_doc = documents[0]
        st.markdown(f"**Nguồn:** `{first_doc.metadata.get('source', '-')}`")
        st.write(preview(first_doc.content, limit=1000))

with chunk_tab:
    st.subheader("So sánh các cách chia nhỏ văn bản")
    text_docs = [doc for doc in documents if doc.metadata.get("extension") in {".txt", ".md"}]
    if not text_docs:
        st.info("Không có `.txt` hoặc `.md` trong lựa chọn hiện tại để so sánh chunking.")
    else:
        selected_doc_id = st.selectbox("Chọn tài liệu", options=[doc.id for doc in text_docs])
        selected_doc = next(doc for doc in text_docs if doc.id == selected_doc_id)
        comparison = ChunkingStrategyComparator().compare(selected_doc.content, chunk_size=chunk_size)

        summary_rows = []
        for strategy_name, stats in comparison.items():
            summary_rows.append(
                {
                    "Cách làm": STRATEGY_LABELS.get(strategy_name, strategy_name),
                    "Số chunk": stats["count"],
                    "Độ dài TB": round(stats["avg_length"], 2),
                    "Ví dụ chunk đầu": preview(stats["chunks"][0] if stats["chunks"] else "", limit=120),
                }
            )
        st.dataframe(summary_rows, width="stretch")

        chunk_preview_tabs = st.tabs([STRATEGY_LABELS.get(name, name) for name in comparison.keys()])
        for tab, strategy_name in zip(chunk_preview_tabs, comparison.keys()):
            with tab:
                for index, piece in enumerate(comparison[strategy_name]["chunks"][:3], start=1):
                    st.markdown(f"**Đoạn {index}**")
                    st.write(piece)

with search_tab:
    st.subheader("Tra cứu nhanh để demo")
    st.caption("Chọn một câu hỏi mẫu, chọn bộ lọc gợi ý, rồi bấm chạy để so sánh kết quả.")

    selected_query = st.selectbox("Câu hỏi mẫu", options=QUERY_OPTIONS, index=0)
    query = st.text_input("Hoặc nhập câu hỏi khác", value=selected_query)
    preset_name = st.selectbox("Bộ lọc gợi ý", options=list(FILTER_PRESETS.keys()), index=0)
    preset_filter = FILTER_PRESETS[preset_name]

    with st.expander("Tùy chỉnh nâng cao (nếu cần)", expanded=False):
        default_filter_text = "" if preset_filter is None else json.dumps(preset_filter, ensure_ascii=False)
        filter_text = st.text_input(
            "Bộ lọc dạng JSON",
            value=default_filter_text,
            help='Ví dụ: {"category": "inflation"} hoặc {"date": "2025-12-10"}',
        )

    selected_strategies = st.multiselect(
        "Chọn cách làm muốn so sánh",
        options=STRATEGIES,
        default=["Raw documents", "RecursiveChunker"],
        format_func=lambda name: STRATEGY_LABELS.get(name, name),
    )

    if st.button("Chạy tra cứu", type="primary"):
        try:
            metadata_filter = parse_filter_text(filter_text) if filter_text.strip() else preset_filter
        except json.JSONDecodeError as exc:
            st.error(f"Bộ lọc JSON không hợp lệ: {exc}")
        else:
            if not selected_strategies:
                st.warning("Hãy chọn ít nhất một cách làm để so sánh.")
            else:
                columns = st.columns(len(selected_strategies))
                for column, strategy_name in zip(columns, selected_strategies):
                    prepared_docs, results, answer, backend_name = run_query(
                        file_tuple,
                        strategy_name,
                        query,
                        top_k,
                        metadata_filter,
                        chunk_size,
                        overlap,
                        max_sentences,
                        use_chroma,
                    )
                    with column:
                        st.markdown(f"### {STRATEGY_LABELS.get(strategy_name, strategy_name)}")
                        st.caption(f"Số tài liệu/chunks: {len(prepared_docs)} · Backend: {backend_name}")
                        if not results:
                            st.warning("Không có kết quả phù hợp.")
                            continue
                        for index, result in enumerate(results, start=1):
                            st.markdown(
                                f"**#{index} · score={result['score']:.4f}**  \\nNguồn: `{result['metadata'].get('source', '-')}`"
                            )
                            st.write(preview(result["content"]))
                        with st.expander("Xem metadata và câu trả lời mẫu"):
                            st.json(result.get("metadata", {}), expanded=False)
                            st.code(answer)

with benchmark_tab:
    st.subheader("Kết quả benchmark")
    suite_name = st.radio(
        "Chọn bộ benchmark",
        options=["Bộ dữ liệu mẫu", "Bộ dữ liệu tài chính"],
        horizontal=True,
    )

    if suite_name == "Bộ dữ liệu mẫu":
        benchmark_files = tuple(path for path in SAMPLE_FILES if path in available_files)
        benchmark_cases = SAMPLE_BENCHMARKS
    else:
        benchmark_files = tuple(path for path in FINANCIAL_DATA_FILES if path in available_files)
        benchmark_cases = FINANCIAL_BENCHMARKS

    st.write("**Các file đang dùng để benchmark:**")
    for path in benchmark_files:
        st.markdown(f"- `{path}`")

    if st.button("Chạy benchmark", key="run_benchmark"):
        if not benchmark_files:
            st.warning("Không tìm thấy file phù hợp trong thư mục `data/` để chạy benchmark.")
        else:
            summary_rows = []
            detail_map: dict[str, list[dict[str, Any]]] = {}
            for strategy_name in STRATEGIES:
                summary, detail_rows = benchmark_strategy(
                    benchmark_files,
                    strategy_name,
                    benchmark_cases,
                    chunk_size,
                    overlap,
                    max_sentences,
                    top_k,
                    use_chroma,
                )
                summary_rows.append(summary)
                detail_map[strategy_name] = detail_rows

            st.dataframe(summary_rows, width="stretch")
            st.caption("`Tỉ lệ đúng (%)` là phần trăm câu hỏi mà top-k results chứa được cụm từ mong đợi trong benchmark.")

            detail_tabs = st.tabs([STRATEGY_LABELS.get(name, name) for name in STRATEGIES])
            for tab, strategy_name in zip(detail_tabs, STRATEGIES):
                with tab:
                    st.dataframe(detail_map[strategy_name], width="stretch")

st.markdown("---")
st.markdown(
    "**Cách chạy:** `streamlit run streamlit_app.py`  \\n"
    "Gợi ý khi thuyết trình: mở lần lượt tab **Giới thiệu dữ liệu** → **So sánh chunking** → **Tra cứu nhanh** → **Benchmark**."
)

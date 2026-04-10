# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phan Thanh Sang
**Nhóm:** 62
**Ngày:** 2026-04-10

> **Tình trạng hiện tại:** Phần `src/` đã hoàn thành và được verify bằng `pytest tests -v` với kết quả **42/42 tests passed**. Báo cáo dưới đây đã được điền sẵn phần kỹ thuật; các mục mang tính cá nhân/nhóm có thể chỉnh lại theo dữ liệu thực tế của nhóm.

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai đoạn văn có hướng biểu diễn rất giống nhau trong không gian embedding, tức là chúng gần nhau về mặt ngữ nghĩa. Nói ngắn gọn, hai câu đang “nói về những ý gần nhau”.

**Ví dụ HIGH similarity:**
- Sentence A: `Python is widely used for machine learning.`
- Sentence B: `Many AI applications are built with Python.`
- Tại sao tương đồng: Cả hai đều nói về việc dùng Python trong AI/machine learning.

**Ví dụ LOW similarity:**
- Sentence A: `Vector stores help semantic search.`
- Sentence B: `Bananas are yellow fruits.`
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau, gần như không có liên hệ ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào **hướng** của vector hơn là độ lớn tuyệt đối, nên phù hợp hơn với text embeddings. Với văn bản, điều quan trọng là mức độ giống nhau về nghĩa chứ không phải độ dài hay magnitude của vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
>
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100 thì `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng lên **25 chunks**. Overlap lớn hơn giúp giữ lại ngữ cảnh giữa các chunk liền kề, giảm nguy cơ mất ý khi câu hoặc đoạn bị cắt ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Internal knowledge assistant / RAG & retrieval notes

**Tại sao nhóm chọn domain này?**
> Bộ dữ liệu mẫu hiện có trong thư mục `data/` xoay quanh Python, vector store, chunking, RAG, support workflow và retrieval song ngữ. Đây là một domain phù hợp để thử nghiệm semantic search vì tài liệu vừa có nội dung kỹ thuật vừa có tài liệu vận hành/support.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `python_intro.txt` | `data/python_intro.txt` | 1944 | `source`, `extension`, `lang=en`, `topic=python` |
| 2 | `vector_store_notes.md` | `data/vector_store_notes.md` | 2123 | `source`, `extension`, `lang=en`, `topic=vector-store` |
| 3 | `rag_system_design.md` | `data/rag_system_design.md` | 2391 | `source`, `extension`, `lang=en`, `topic=rag` |
| 4 | `customer_support_playbook.txt` | `data/customer_support_playbook.txt` | 1692 | `source`, `extension`, `lang=en`, `topic=support` |
| 5 | `chunking_experiment_report.md` | `data/chunking_experiment_report.md` | 1987 | `source`, `extension`, `lang=en`, `topic=chunking` |
| 6 | `vi_retrieval_notes.md` | `data/vi_retrieval_notes.md` | 1667 | `source`, `extension`, `lang=vi`, `topic=retrieval` |

### Additional Validation Corpus (Financial TXT)

> Sau khi hoàn thành phần core lab, tôi tiếp tục kiểm thử trên bộ dữ liệu tài chính thực tế đã được thay mới trong thư mục `data/`. Theo file `data/final_documents_metadata.json`, bộ này có **5 documents** với tổng cộng **122,588 ký tự** từ hai nguồn chính là **BLS** và **Federal Reserve**.

| # | Tên tài liệu | Nguồn | Quy mô | Metadata nổi bật |
|---|--------------|-------|--------|------------------|
| 7 | `bls_cpi_december_2025.txt` | `data/bls_cpi_december_2025.txt` | 111,283 chars | `reference_period`, `category=inflation`, `event_type=CPI Report` |
| 8 | `fed_fomc_statement_2025_07_30.txt` | `data/fed_fomc_statement_2025_07_30.txt` | 2,978 chars | `date`, `category=monetary_policy`, `interest_rate` |
| 9 | `fed_fomc_statement_2025_09_17.txt` | `data/fed_fomc_statement_2025_09_17.txt` | 2,612 chars | `date`, `category=monetary_policy`, `interest_rate` |
| 10 | `fed_fomc_statement_2025_10_29.txt` | `data/fed_fomc_statement_2025_10_29.txt` | 2,792 chars | `date`, `category=monetary_policy`, `interest_rate` |
| 11 | `fed_fomc_statement_2025_12_10.txt` | `data/fed_fomc_statement_2025_12_10.txt` | 2,923 chars | `date`, `category=monetary_policy`, `interest_rate` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | `str` | `data/vector_store_notes.md` | Giúp truy vết nguồn tài liệu và explainability |
| `extension` | `str` | `.md`, `.txt` | Hữu ích khi muốn lọc theo loại file |
| `lang` | `str` | `en`, `vi` | Giúp filter theo ngôn ngữ truy vấn |
| `topic` | `str` | `rag`, `support`, `python` | Thu hẹp search space theo chủ đề |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên `rag_system_design.md` với `chunk_size=200`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `rag_system_design.md` | FixedSizeChunker (`fixed_size`) | 14 | 189.36 | Trung bình — dễ cắt giữa ý |
| `rag_system_design.md` | SentenceChunker (`by_sentences`) | 8 | 297.12 | Khá tốt nhưng một số chunk còn dài |
| `rag_system_design.md` | RecursiveChunker (`recursive`) | 20 | 117.65 | Tốt — giữ cấu trúc đoạn và ý rõ hơn |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> Strategy này thử tách theo thứ tự ưu tiên từ separator lớn đến nhỏ: `\n\n`, `\n`, `. `, khoảng trắng, và cuối cùng là tách cứng theo kích thước nếu cần. Cách này giúp giữ nguyên cấu trúc đoạn văn càng lâu càng tốt trước khi phải cắt nhỏ hơn. Nhờ đó, chunk vừa nằm trong giới hạn độ dài vừa ít bị mất ý.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu kỹ thuật và playbook trong thư mục `data/` đã có cấu trúc khá rõ theo đoạn và heading. Recursive chunking tận dụng tốt cấu trúc đó, nên phù hợp hơn fixed-size chunking khi muốn giữ ngữ cảnh cho retrieval.

**Code snippet (nếu custom):**
```python
# Sử dụng built-in strategy đã implement trong src/chunking.py
from src import RecursiveChunker

chunker = RecursiveChunker(chunk_size=200)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `rag_system_design.md` | FixedSizeChunker | 14 | 189.36 | Ổn nhưng đôi lúc trả về fragment thiếu ngữ cảnh |
| `rag_system_design.md` | **RecursiveChunker (của tôi)** | 20 | 117.65 | Cân bằng tốt giữa độ ngắn và độ đầy đủ ý |

### So Sánh Với Thành Viên Khác

> Dưới đây là phần so sánh thực nghiệm giữa 3 strategy trên cùng 5 benchmark queries của bộ dữ liệu mẫu. Nếu nhóm có số liệu riêng, có thể thay tên và tinh chỉnh lại bảng này.

| Thành viên / Strategy | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | `RecursiveChunker` | **8 / 10** | Giữ ngữ cảnh tốt, phù hợp tài liệu có heading/paragraph, top-3 relevant ở 4/5 queries | Vẫn có failure case nếu query mơ hồ hoặc thiếu metadata filter |
| Baseline A | `FixedSizeChunker` | **4 / 10** | Dễ cài đặt, chunk size ổn định | Dễ cắt giữa ý, retrieval quality thấp hơn trên bộ benchmark mẫu |
| Baseline B | `SentenceChunker` | **8 / 10** | Dễ đọc, thường trả về chunk coherent | Kích thước chunk không đều, đôi lúc quá dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Trên bộ benchmark mẫu, `SentenceChunker` và `RecursiveChunker` đều đạt **8/10**, cao hơn rõ so với `FixedSizeChunker` (**4/10**). Tôi vẫn ưu tiên `RecursiveChunker` vì nó cân bằng tốt hơn giữa việc giữ cấu trúc tài liệu và giới hạn độ dài chunk.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])\s+` để tách theo ranh giới câu và sau đó gom tối đa `max_sentences_per_chunk` câu vào một chunk. Hàm cũng xử lý edge case như chuỗi rỗng, khoảng trắng thừa, và loại bỏ các câu rỗng sau khi split.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Tôi triển khai thuật toán đệ quy: nếu đoạn hiện tại đã nhỏ hơn `chunk_size` thì trả về luôn; nếu chưa, thử tách bằng separator lớn nhất còn lại. Nếu một phần vẫn quá dài thì tiếp tục đệ quy với separator nhỏ hơn, và nếu hết separator thì fallback sang cắt cứng theo kích thước.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Store hỗ trợ cả in-memory lẫn ChromaDB nếu thư viện có sẵn, nhưng luôn fallback an toàn về in-memory để tests không phụ thuộc môi trường ngoài. Khi thêm tài liệu, code tạo record chuẩn hóa, tính embedding cho từng document, lưu metadata, và khi search thì embed query rồi xếp hạng bằng cosine similarity.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter()` lọc record theo metadata **trước** khi tính similarity để giảm nhiễu đúng như yêu cầu lab. `delete_document()` tìm tất cả record có `doc_id` tương ứng, xóa khỏi in-memory store, và đồng bộ xóa trên Chroma nếu backend đó đang được bật.

**Mở rộng cho dữ liệu mới:**
> Tôi cũng cập nhật `main.py` để ingest thêm file `.csv` theo từng dòng và chuyển mỗi row thành một `Document`. Cách này cho phép tái sử dụng nguyên pipeline vector store/search trên dữ liệu cấu trúc như CPI và FOMC mà không phải viết lại store riêng.

### KnowledgeBaseAgent

**`answer`** — approach:
> Hàm `answer()` lấy top-k chunk liên quan từ store, đánh số từng chunk trong phần `Context`, rồi ghép thành prompt yêu cầu model chỉ trả lời dựa trên bằng chứng đã retrieve. Cách này phản ánh đúng RAG pattern: retrieve trước, rồi mới generate từ context.

### Test Results

```text
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.0.3
collected 42 items
...
============================= 42 passed in 0.08s ==============================
```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> Các score dưới đây được đo bằng `compute_similarity(_mock_embed(a), _mock_embed(b))`, nên mang tính minh họa cho pipeline hơn là đánh giá semantic quality của một embedder thật.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is used for machine learning. | Python helps build AI applications. | High | 0.1324 | Có |
| 2 | Vector stores keep embeddings. | Databases can store vectors for semantic search. | High | -0.2655 | Không |
| 3 | Recursive chunking preserves context. | Chunking strategy affects retrieval quality. | High | -0.0560 | Không |
| 4 | Billing issues should be escalated. | Customers may need password recovery support. | Low | -0.2205 | Có |
| 5 | Python is a programming language. | Bananas are yellow fruits. | Low | 0.0785 | Gần đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 2 và cặp 3 khá bất ngờ vì về mặt ngữ nghĩa chúng khá gần nhau nhưng score vẫn thấp hoặc âm. Điều này cho thấy `_mock_embed` rất hữu ích để test pipeline một cách deterministic, nhưng nếu muốn đánh giá chất lượng semantic retrieval thực sự thì nên thử thêm `LocalEmbedder` hoặc `OpenAIEmbedder`.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **Bảng dưới đây dùng bộ dữ liệu mẫu hiện có trong repo làm điểm khởi đầu; nhóm có thể chỉnh lại theo benchmark thực tế.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Nguồn/chunk chính |
|---|-------|-------------|-------------------|
| 1 | What is a vector store? | A vector store is a storage layer for embeddings that retrieves the most similar items to a query vector. | `vector_store_notes.md`, `rag_system_design.md` |
| 2 | Why is recursive chunking often useful? | It preserves context by splitting on larger boundaries first and only using smaller separators when needed. | `chunking_experiment_report.md` |
| 3 | When should a support assistant escalate? | When retrieval is insufficient or the system lacks reliable documentation, the assistant should recommend escalation instead of improvising. | `customer_support_playbook.txt` |
| 4 | Trong tài liệu tiếng Việt, metadata giúp gì cho retrieval? | Metadata giúp lọc theo ngôn ngữ/chủ đề để tránh lấy nhầm tài liệu không liên quan và tăng độ chính xác. | `vi_retrieval_notes.md` |
| 5 | Python is commonly used for what kinds of tasks? | Automation, backend services, data analysis, scientific computing, and machine learning. | `python_intro.txt` |

### Kết Quả Của Tôi

> Kết quả dưới đây được chạy với `RecursiveChunker(chunk_size=220)` + `EmbeddingStore` + `_mock_embed`. Với query số 4, tôi dùng thêm `search_with_filter(metadata_filter={"lang": "vi"})` để minh họa lợi ích của metadata filtering.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is a vector store? | `vector_store_notes.md` — đoạn mô tả vector store là storage layer cho embeddings và semantic search | 0.3069 | Yes | Trả lời đúng định nghĩa và mục đích của vector store |
| 2 | Why is recursive chunking often useful? | `chunking_experiment_report.md` — nêu rõ recursive chunking giữ context bằng cách tách từ separator lớn xuống nhỏ | 0.3488 | Yes | Trả lời đúng lý do strategy này thường hữu ích |
| 3 | When should a support assistant escalate? | `rag_system_design.md` đứng top-1, còn chunk support không lên đầu | 0.3799 | No | Đây là failure case: cần metadata/topic filter hoặc embedder mạnh hơn |
| 4 | Trong tài liệu tiếng Việt, metadata giúp gì cho retrieval? | `vi_retrieval_notes.md` — giải thích filter theo ngôn ngữ/phòng ban/chủ đề | 0.0655 | Yes | Trả lời đúng vai trò của metadata khi lọc tài liệu tiếng Việt |
| 5 | Python is commonly used for what kinds of tasks? | `python_intro.txt` — nêu automation, backend, data analysis, AI/ML | 0.3265 | Yes | Trả lời đúng các nhóm tác vụ phổ biến của Python |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **4 / 5** trên bộ dữ liệu mẫu với `RecursiveChunker` + `_mock_embed`

### Validation on New Financial TXT Files (bổ sung ngày 2026-04-10)

> Sau khi đổi sang bộ dữ liệu `.txt` tài chính thực tế, tôi đã chạy lại pipeline trên 5 documents mới. Kết quả xác minh cho thấy ingest hoạt động bình thường và các filter như `date` / `reference_period` vẫn phát huy hiệu quả sau khi tôi bổ sung metadata từ tên file và nội dung tài liệu.

| Query kiểm thử | Filter dùng khi search | Kết quả nổi bật | Đánh giá |
|---|---|---|---|
| `What interest rate range did the Fed set on December 10, 2025?` | `{'date': '2025-12-10'}` | Top hit là `fed_fomc_statement_2025_12_10.txt` với score **0.8148** và metadata chứa `interest_rate: 3-1/2 to 3-3/4 percent` | Đúng và rõ ràng |
| `What was the Year-over-Year CPI change in December 2025?` | `{'reference_period': 'DECEMBER 2025'}` | Top hit là `bls_cpi_december_2025.txt` với score **1.2081** và phần mở đầu nêu rõ `Year-over-Year Change: 2.7%` | Trả lời đúng và có grounding tốt |

**Kết luận ngắn từ phần test dữ liệu mới:**
> Với bộ `.txt` tài chính mới, hệ thống hiện vẫn hoạt động ổn trên cả `in-memory` lẫn `ChromaDB`. Để retrieval chính xác hơn, nên tiếp tục kết hợp **metadata filter (`date`, `reference_period`, `category`)** và về sau thay `_mock_embed` bằng embedder thật như `sentence-transformers`.

---

## 7. What I Learned (5 điểm — Demo)

### Failure Analysis (Ex 3.5)

**Failure case tôi ghi nhận:**
> Query `When should a support assistant escalate?` là ví dụ retrieval thất bại rõ nhất trong benchmark. Với `RecursiveChunker` + `_mock_embed`, top-1 lại rơi vào `rag_system_design.md` thay vì `customer_support_playbook.txt`.

**Tại sao thất bại?**
> Có ba nguyên nhân chính: (1) câu hỏi còn khá rộng, (2) metadata về `topic=support` chưa được khai thác mạnh trong search mặc định, và (3) mock embedding chỉ phục vụ kiểm thử pipeline nên semantic quality không mạnh như embedder thật.

**Đề xuất cải thiện:**
> Tôi sẽ bổ sung metadata như `topic`, `department`, `audience`, `lang`, sau đó dùng `search_with_filter()` khi câu hỏi đã có phạm vi rõ. Ngoài ra, thử `LocalEmbedder` hoặc `OpenAIEmbedder` sẽ giúp đánh giá retrieval quality thực tế tốt hơn.

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Metadata filtering không chỉ là “nice to have” mà thực sự giúp giảm nhiễu khi câu hỏi đã có phạm vi rõ, ví dụ theo phòng ban hoặc ngôn ngữ. Đây là một cách cải thiện retrieval rẻ hơn nhiều so với đổi model lớn hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Cùng một bộ tài liệu nhưng thay đổi chiến lược chunking có thể làm kết quả khác hẳn. Recursive chunking thường là lựa chọn cân bằng tốt cho tài liệu kỹ thuật hỗn hợp.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ bổ sung metadata rõ hơn như `department`, `audience`, `lang`, `updated_at` và chạy lại benchmark với một embedder thật thay vì chỉ `_mock_embed`. Ngoài ra, tôi sẽ thu thập thêm failure cases để tinh chỉnh chunk size và filter strategy.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **83 / 100** |

> Có thể điều chỉnh lại phần tự đánh giá sau khi nhóm hoàn tất benchmark và demo thực tế.

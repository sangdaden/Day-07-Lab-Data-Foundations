# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phan Thanh Sang
**Nhóm:** 62
**Ngày:** 10/04/2026

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(chunk_size=500)` trên 2 tài liệu thật:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `fed_fomc_statement_2025_07_30.txt` (2,943 chars) | FixedSizeChunker | 7 | 463.3 | Trung bình — cắt ngang giữa câu |
| `fed_fomc_statement_2025_07_30.txt` | SentenceChunker | 10 | 292.7 | Tốt — giữ ranh giới câu |
| `fed_fomc_statement_2025_07_30.txt` | RecursiveChunker | 9 | 325.2 | Tốt — tách theo paragraph |
| `bls_cpi_december_2025.txt` (111,283 chars) | FixedSizeChunker | 248 | 498.5 | Kém — cắt ngang bảng số liệu |
| `bls_cpi_december_2025.txt` | SentenceChunker | 323 | 343.3 | Kém — bảng số liệu không có dấu câu |
| `bls_cpi_december_2025.txt` | RecursiveChunker | 263 | 421.4 | Tốt nhất — tách theo `\n\n` trước |

### Strategy Của Tôi

**Loại:** `RecursiveChunker` (chunk_size=500)

**Mô tả cách hoạt động:**

RecursiveChunker thử tách text theo danh sách separator ưu tiên: `["\n\n", "\n", ". ", " ", ""]`. Với tài liệu tài chính, separator `"\n\n"` tách theo paragraph, giữ mỗi đoạn văn nguyên vẹn. Nếu một paragraph quá dài, tiếp tục tách bằng `"\n"` rồi `". "`. Kết quả: chunk có ranh giới tự nhiên hơn FixedSizeChunker và ít bị cắt ngang ý hơn.

**Tại sao chọn strategy này cho domain financial news?**

Tài liệu tài chính (FOMC statements, BLS press releases) được viết theo cấu trúc paragraph rõ ràng — mỗi paragraph nêu một luận điểm (quyết định lãi suất, mô tả thị trường lao động, lạm phát...). RecursiveChunker giữ nguyên cấu trúc đó. Với file BLS lớn (111k chars), việc tách theo `\n\n` giúp tránh bị cắt ngang bảng số liệu.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|-----------|----------|-------------|------------|-------------------|
| FOMC statement (2,943 chars) | SentenceChunker (baseline tốt nhất) | 10 | 292.7 | Tốt — giữ ranh giới câu |
| FOMC statement | **RecursiveChunker (của tôi)** | **9** | **325.2** | **Tốt — ít chunks hơn, avg length lớn hơn → context đủ rộng** |
| BLS CPI (111,283 chars) | FixedSizeChunker (baseline) | 248 | 498.5 | Kém — cắt ngang bảng |
| BLS CPI | **RecursiveChunker (của tôi)** | **263** | **421.4** | **Tốt nhất trong 3 — tách theo paragraph** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Queries Relevant | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------|-----------|----------|
| Tôi | RecursiveChunker (chunk_size=500) | 3/5 | Filter theo date hoạt động đúng | Q3 thất bại do top-1 là raw data chunk |
| Phan Thanh Sang | RawDocumentChunker | 5/5 | Score cao nhất (1.22), không bị cắt nhầm vào raw data | Không scale được với document lớn + embedder thật |
| Đỗ Minh Khiêm | RecursiveChunker (chunk_size=500) | 5/5 | Chunk đầu file relevant, agent answer đầy đủ | Score âm ở Q2/Q3 (mock embedder không semantic) |
| Trần Đình Minh Vương | SentenceChunker (max=3) | 2-3/5 | Score cao ở Q1 (0.69) | Retrieve nhầm tháng ở Q2 — thiếu date filter |
| Trần Tiến Dũng | SentenceChunker (max=5) | ~4/5 | Gold answer chi tiết (kể cả dissenters FOMC vote) | Dùng queries riêng, khó so sánh trực tiếp |

**Strategy nào tốt nhất cho domain này? Tại sao?**

Kết quả thú vị: `RawDocumentChunker` (Sang) đạt score cao nhất với mock embedder vì không bị rơi vào chunk raw data — nhưng đây là artifact của mock embedder, không phải ưu điểm thật. Với embedder thật và document 111k chars, raw embedding sẽ bị "loãng" và kém hơn chunking. `RecursiveChunker` (Khiêm, và tôi) là strategy cân bằng nhất cho production. Bài học quan trọng: **metadata filter theo `date`** là yếu tố quyết định với FOMC statements — TV3 (Vương) thất bại Q2 vì thiếu date filter, retrieve nhầm tháng 7 thay vì tháng 12.
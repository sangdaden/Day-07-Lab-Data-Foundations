# Kịch bản demo 3–4 phút

## Mục tiêu
Giới thiệu ngắn gọn bộ dữ liệu, cách hệ thống retrieval hoạt động, và minh họa benchmark trên giao diện Streamlit.

---

## 0:00 – 0:30 | Mở đầu
**Nói:**

> Xin chào thầy/cô và các bạn. Đây là demo cho Lab 7 về **Embedding, Vector Store và Retrieval**.  
> Nhóm em xây dựng một giao diện Streamlit để trực quan hóa dữ liệu, so sánh các chiến lược chunking, và chạy benchmark trên bộ dữ liệu tài chính thực tế.

---

## 0:30 – 1:15 | Tab `Giới thiệu dữ liệu`
**Thao tác:** mở tab `Giới thiệu dữ liệu`

**Nói:**

> Ở đây là phần tổng quan về bộ dữ liệu.  
> Nhóm em sử dụng dữ liệu tài chính thực từ **Bureau of Labor Statistics** và **Federal Reserve**.  
> Bộ dữ liệu gồm các báo cáo CPI và các thông cáo FOMC, rất phù hợp để kiểm tra retrieval vì vừa có số liệu cụ thể, vừa có phần giải thích bằng ngôn ngữ tự nhiên.

> Ngoài ra, mỗi tài liệu đều có metadata như `category`, `date`, `reference_period`, giúp hệ thống lọc và truy xuất chính xác hơn.

---

## 1:15 – 2:00 | Tab `So sánh chunking`
**Thao tác:** mở tab `So sánh chunking`, chọn một tài liệu FOMC hoặc CPI

**Nói:**

> Ở tab này, em so sánh 3 cách chia nhỏ văn bản: **chia đoạn cố định**, **chia theo câu**, và **chia đệ quy**.  
> Mỗi cách tạo ra số lượng chunk và độ dài trung bình khác nhau.

> Qua quan sát, `RecursiveChunker` thường giữ ngữ cảnh tốt hơn, nên phù hợp hơn cho retrieval trên tài liệu dài và có cấu trúc như báo cáo kinh tế hoặc thông cáo Fed.

---

## 2:00 – 3:00 | Tab `Tra cứu nhanh`
**Thao tác:** mở tab `Tra cứu nhanh`

### Demo câu hỏi 1
Chọn query:
- `What was the CPI year-over-year change in December 2025?`
- preset filter: `CPI tháng 12/2025`

**Nói:**

> Với câu hỏi này, hệ thống truy xuất đúng tài liệu `bls_cpi_december_2025.txt` và trả ra kết quả **2.7% year-over-year**.

### Demo câu hỏi 2
Chọn query:
- `What interest rate did the Fed set in December 2025?`
- preset filter: `FOMC tháng 12/2025`

**Nói:**

> Với câu hỏi về lãi suất tháng 12/2025, hệ thống trả về đúng file FOMC và kết quả là **3-1/2 to 3-3/4 percent**.

> Phần này cũng cho thấy metadata filtering rất quan trọng, vì nếu lọc theo `date` hoặc `category` thì kết quả ổn định hơn nhiều.

---

## 3:00 – 3:40 | Tab `Benchmark`
**Thao tác:** mở tab `Benchmark`, bấm `Chạy benchmark`

**Nói:**

> Cuối cùng là phần benchmark.  
> Ở đây em so sánh các cách làm trên cùng một bộ câu hỏi để xem cách nào cho retrieval tốt hơn.

> Kết quả này giúp nhóm đánh giá được chiến lược chunking nào phù hợp nhất với domain dữ liệu tài chính, đồng thời minh họa vai trò của metadata filtering và vector store trong một hệ thống RAG đơn giản.

---

## 3:40 – 4:00 | Kết luận
**Nói:**

> Tóm lại, bài demo này cho thấy ba điểm chính:  
> 1. Cần chunking hợp lý để giữ ngữ cảnh,  
> 2. Metadata filtering giúp tăng độ chính xác,  
> 3. Giao diện Streamlit giúp trực quan hóa toàn bộ quá trình retrieval và benchmark.  
> Em xin hết, cảm ơn thầy/cô và các bạn.

---

## Gợi ý thao tác nhanh khi demo
1. Mở `streamlit run streamlit_app.py`
2. Vào tab `Giới thiệu dữ liệu`
3. Sang tab `So sánh chunking`
4. Sang tab `Tra cứu nhanh` và chạy 2 câu hỏi mẫu
5. Kết thúc ở tab `Benchmark`

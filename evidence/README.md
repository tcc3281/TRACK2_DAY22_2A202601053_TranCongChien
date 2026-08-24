# Báo Cáo Đánh Giá & Bằng Chứng Thực Nghiệm — Day 22 Lab

## 1. Danh sách các tệp bằng chứng (Evidence Files)

| STT | Tên tệp | Mô tả | Trạng thái |
|:---|:---|:---|:---|
| 1 | `01_langsmith_traces.png` | Ảnh chụp màn hình giao diện LangSmith với ≥ 50 traces cho RAG pipeline (`rag-query`) | ✅ Đã hoàn thành |
| 2 | `02_prompt_hub.png` | Ảnh chụp màn hình giao diện LangSmith Prompt Hub hiển thị 2 phiên bản prompt (`tran-cong-chien-rag-prompt-v1` và `tran-cong-chien-rag-prompt-v2`) | ✅ Đã hoàn thành |
| 3 | `02_ab_routing_log.txt` | Log console của quá trình A/B routing 50 câu hỏi tất định theo MD5 hash | ✅ Đã hoàn thành |
| 4 | `03_ragas_scores.png` | Ảnh chụp terminal hiển thị bảng so sánh điểm 4 chỉ số RAGAS giữa V1 và V2 | ✅ Đã hoàn thành |
| 5 | `03_ragas_report.json` | Bản sao tệp JSON chứa kết quả chi tiết đánh giá RAGAS cho V1 và V2 | ✅ Đã hoàn thành |
| 6 | `04_pii_demo_log.txt` | Log console của bộ validator PIIDetector phát hiện và che giấu các thông tin PII nhạy cảm | ✅ Đã hoàn thành |
| 7 | `04_json_demo_log.txt` | Log console của bộ validator JSONFormatter tự động sửa lỗi và chuẩn hoá cú pháp JSON | ✅ Đã hoàn thành |

---

## 2. Phân tích kết quả thực nghiệm RAGAS (V1 vs V2)

### Bảng kết quả định lượng:

| Chỉ số (Metric) | Prompt V1 (Ngắn gọn & Thân thiện) | Prompt V2 (Chuyên gia & Cấu trúc) | Phiên bản vượt trội (Winner) |
|:---|:---:|:---:|:---:|
| **Faithfulness** | **0.9755** ⭐ | 0.8203 ⭐ | **V1** |
| **Answer Relevancy** | **0.9118** | 0.8947 | **V1** |
| **Context Recall** | **1.0000** | **1.0000** | **Hòa (Cả 2 đều đạt tối đa)** |
| **Context Precision** | 0.9417 | **0.9450** | **V2** |

*(Cả 2 phiên bản đều đạt mục tiêu `faithfulness ≥ 0.8`, trong đó V1 đạt tới `0.9755`)*

---

### Phân tích chuyên sâu:

1. **Vì sao Faithfulness của V1 (0.9755) cao hơn đáng kể so với V2 (0.8203)?**
   - **Prompt V1** được thiết kế ngắn gọn, súc tích (2–4 câu), yêu cầu LLM trả lời trực tiếp và **chỉ dựa trên context**, từ chối suy diễn nếu thiếu thông tin. Khi câu trả lời ngắn và tập trung, xác suất mô hình sinh thêm thông tin ngoài lề (hallucination) là cực kỳ thấp, do đó độ trung thực với ngữ cảnh (*Faithfulness*) đạt gần như tuyệt đối (97.55%).
   - **Prompt V2** yêu cầu phong cách chuyên gia với 3 bước phân tích có cấu trúc (tóm tắt, trích dẫn số liệu/nguồn, đánh giá mức độ chắc chắn). Do yêu cầu giải thích và suy luận cấu trúc dài hơn, LLM có xu hướng sử dụng thêm các từ nối logic hoặc diễn giải ngữ nghĩa mở rộng, dẫn đến việc thuật toán evaluator của RAGAS đánh giá một số luận điểm là không được hỗ trợ hoàn toàn 100% từ context gốc.

2. **Về Answer Relevancy (Độ liên quan của câu trả lời):**
   - **V1 (0.9118)** nhỉnh hơn V2 (0.8947) vì V1 trả lời thẳng vào trọng tâm câu hỏi của người dùng mà không thêm các phần trình bày quy trình hay nhận định mức độ chắc chắn như V2.

3. **Về Context Recall và Context Precision:**
   - Cả 2 phiên bản đều đạt **Context Recall = 1.0000**, chứng tỏ bộ retriever với FAISS (k=3) đã truy xuất đầy đủ tất cả các thông tin cần thiết để trả lời 50 câu hỏi chuẩn.
   - **Context Precision của V2 (0.9450)** cao hơn nhẹ so với V1 (0.9417), cho thấy khi có chỉ dẫn cấu trúc chi tiết, các đoạn trích dẫn quan trọng được tận dụng chặt chẽ ở các vị trí đầu câu trả lời.

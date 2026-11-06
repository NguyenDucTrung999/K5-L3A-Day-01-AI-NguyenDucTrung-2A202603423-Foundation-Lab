# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603423
- **Họ và tên:** Nguyen Duc Trung
- **SEED:** 3251

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (kèm [`hand_calc(2).jpg`](hand_calc(2).jpg)) | [x] |

## 2. Kết quả chính

| Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.871 | 0.857 | 0.08 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.969 | 0.947 | 0.20 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.914 | 0.974 | 130.75 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.965 | 0.923 | 38.29 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.952 | 0.947 | 38.14 | 0.000 |
| 4a. LLM nhỏ zero-shot | — | 0.927 | 1037.72 | 0.021 |
| 4c. LLM nhỏ few-shot | — | 0.950 | 1772.95 | 0.053 |

## 3. Dự đoán và thực tế

1. **Dự đoán `ml_lr` (Learning Rate)**: 
   - Đoán: 0.001 thì loss giảm rất chậm, còn lr = 50 thì loss giảm nhanh 1 đường chéo.
   - Thực tế: lr = 50 đưa loss xuống cực thấp (0.004) sau 300 epoch, trong khi lr = 0.001 loss còn ở mức 0.354.
   - Lý do lệch: lr = 50 trên bài toán lồi nhỏ giúp bước nhảy lớn nhanh chóng tiến gần điểm cực tiểu, dù có nguy cơ overfit.
2. **Dự đoán `emb_search` (Tìm kiếm ngữ nghĩa)**:
   - Đoán: Câu truy vấn né từ khoá ("complimentary holiday") là không phải spam ("ko spam").
   - Thực tế: Cả 6 kết quả trả về gần nhất đều là tin nhắn spam.
   - Lý do lệch: Mô hình embedding hiểu được ngữ nghĩa tương đương giữa "complimentary holiday", "claim" với "free prize" mà không bị đánh lừa bởi mặt chữ.
3. **Dự đoán `dl_relu` (Bỏ ReLU)**:
   - Đoán: Nếu không có ReLU thì loss sẽ tăng, F1 giảm.
   - Thực tế: F1 trên tập test vẫn đạt 0.969, gần như tương đương với mô hình có ReLU.
   - Lý do lệch: Hai lớp Linear liên tiếp tương đương với một phép biến đổi tuyến tính đơn lẻ, và bài toán phân loại nhị phân trên không gian 2000 từ đã có ranh giới tuyến tính đủ tốt.

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất**: BoW + LogReg tụt mạnh nhất (F1 test thường 0.871 tụt sâu xuống ở nhóm bẫy), tụt nặng nhất ở nhóm *Né từ khoá* (leet speak) và *Tưởng spam nhưng là ham*.
- **Hai case sai cụ thể**:
  1. Case `F R E E` (thật là spam): BoW đoán sai thành ham vì các ký tự cách nhau không khớp từ điển. Ngược lại, Embedding nhận diện được biến thể ngữ nghĩa và đoán đúng.
  2. Case `Congrats! You WON the bet lol, call me` (thật là ham): BoW thấy từ "WON" và "Call" có trọng số spam lớn nên chặn nhầm.

## 5. Thử thách cá nhân

- Đề số 6: LLM · Sửa build_classification_prompt để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?
- Kết quả & Giải thích: Bản trả JSON tốn số lượng token gấp 3-4 lần do phải sinh thêm cú pháp và trường giải thích `reason`. Không đáng dùng cho hệ thống thời gian thực vì làm tăng độ trễ và chi phí; chỉ đáng khi cần lưu log giải trình lý do chặn tin nhắn cho người dùng.

## 6. Phương án cho sản phẩm thật

- Với yêu cầu xử lý 10 triệu SMS/ngày và p95 < 50 ms/tin:
  - **Phương án lựa chọn**: `3b. Encoder (MiniLM) + LogReg`.
  - **Số liệu chứng minh**: p95 đạt 38.29 ms (đáp ứng tiêu chuẩn SLA < 50 ms), F1 trên tập test rất cao (0.965), chi phí suy luận 0 USD.
  - **Đánh đổi**: Tốn thời gian tính toán ban đầu để encode dữ liệu train (~79s), nhưng bù lại tốc độ gọi từng tin rất nhanh và ổn định.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 (Cấu hình) | Cách cấu hình để dùng Gemini API thay cho OpenAI trong notebook | Hướng dẫn đổi LLM_PROVIDER = "gemini" ở Cell 62 và thêm Secret GEMINI_API_KEY | Nhập API key của Google AI Studio vào ô prompt và kiểm tra in ra cấu hình Gemini thành công |
| Phần 4 (TODO 4.3) | Cách viết hàm parse_label bằng regex để không bị bắt nhầm khi LLM trả về "not spam" | Dùng regex kiểm tra cụm `\bnot\s+spam\b` trước để gán nhãn 0, sau đó mới kiểm tra `\bspam\b` | Chạy qua ô kiểm tra Cell 67, xác nhận các case "Not spam", "Label: spam", "HAM\n" đều trả về đúng nhãn |
| Phần 4 (Thí nghiệm 4.A) | Tại sao chạy batch trên Gemini bị lỗi RateLimitError: 429 limit 15 requests per minute? | Giải thích do gói Gemini Free Tier chỉ cho 15 lượt gọi/phút, 40 tin gọi liên tục sẽ bị khóa tạm thời | Thêm time.sleep(4.1) sau mỗi tin khi dùng Gemini và tăng max_retries=5 trong retry_with_backoff, chạy lại không còn bị 429 |
| Phần 4 (Cấu hình) | Sửa lỗi BadRequestError 400: Thinking level MINIMAL is not supported for this model trên gemini-3.8-flash | Do model mới không hỗ trợ mức minimal, gợi ý bỏ hẳn tham số reasoning_effort | Đổi "params": {} ở Cell 62 cho cấu hình Gemini, chạy lại cả 2 model đều hoạt động bình thường |
| Phần 5 (Stress test 50 case) | Giải thích vì sao mô hình BoW bị tụt điểm nghiêm trọng ở nhóm né từ khóa và bẫy prompt | Phân tích cơ chế của BoW (chỉ nhìn token rời rạc theo từ điển cứng) nên bị mù trước ký tự biến thể và lệnh điều khiển | Đọc bảng kết quả theo danh mục tại Cell 87, kiểm chứng nhận xét khớp với số liệu thực tế |

**Điều hiểu ra hôm nay (2–3 câu):**
1. Mô hình phức tạp như Deep Learning không phải lúc nào cũng vượt trội; với bài toán phân loại đơn giản trên Bag-of-Words thì Logistic Regression nhanh hơn và F1 tương đương.
2. Embedding thể hiện rõ sức mạnh khi xử lý các câu văn ngụy trang hoặc né từ khóa nhờ việc nắm bắt ngữ nghĩa tổng thể thay vì đếm từ khóa thô.
3. LLM rất tiện lợi khi không cần gắn nhãn dữ liệu huấn luyện, nhưng độ trễ cao và chi phí lớn khiến nó phù hợp nhất khi làm bộ lọc bổ trợ cho các ca khó trong hệ thống lai.

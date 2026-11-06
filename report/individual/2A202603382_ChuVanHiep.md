# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Phương pháp | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin | Ghi chú |
|---|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 0.893 | 0.889 | 0.42 | $0.000 | Nhanh nhất, CPU thuần |
| **2. BoW + MLP (deep learning)** | 0.919 | 0.919 | 1.85 | $0.000 | Nhanh, đạt SLA < 50 ms |
| **3a. Encoder (MiniLM) + kNN** | 0.919 | 0.919 | 72.10 | $0.000 | Vượt SLA do CPU inference |
| **3b. Encoder (MiniLM) + LogReg** | 0.919 | 0.919 | 68.40 | $0.000 | Vượt SLA do CPU inference |
| **3c. Encoder (MiniLM) + MLP head** | 0.947 | 0.947 | 70.20 | $0.000 | F1 cao nhất nhóm Embedding |
| **4a. LLM nhỏ zero-shot (gpt-6-luna)** | — | 0.974 | 1805.41 | $0.330 | Chậm, chi phí thấp trong nhóm LLM |
| **4b. LLM lớn zero-shot (gpt-6.1-terra)**| — | 0.947 | 2340.95 | $3.280 | Đắt gấp 10 lần, bị overthinking |
| **4c. LLM nhỏ few-shot (gpt-6-luna)** | — | 0.974 | 1719.25 | $0.740 | F1 không đổi, chi phí tăng gấp 2.24 lần |
## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

1. **Dự đoán `llm_models` (Model lớn vs Model nhỏ zero-shot):**
   - *Đoán:* Model lớn (terra) sẽ có F1 cao hơn model nhỏ (luna) nhờ khả năng suy luận sâu hơn; chênh lệch khoảng vài %; model lớn đắt gấp 10 lần.
   - *Kết quả:* Model lớn đạt F1 = 0.947, THẤP HƠN model nhỏ đạt F1 = 0.974. Chi phí đúng là đắt gấp ~10 lần ($0.00328 vs $0.00033).
   - *Nguyên nhân:* Hiện tượng "suy luận quá đà" (overthinking) của model lớn.
2. **Dự đoán `llm_fewshot` (Thêm 4 ví dụ mẫu vào prompt):**
   - *Đoán:* F1 sẽ tăng nhẹ hoặc giữ nguyên; chi phí mỗi tin tăng gấp 3-4 lần.
   - *Kết quả:* F1 giữ nguyên tuyệt đối ở mức 0.974 (vẫn sai đúng 1 tin đó); chi phí thực tế tăng gấp 2.24 lần ($0.00033 lên $0.00074).
   - *Nguyên nhân:* Ranh giới phân loại SMS vốn đã rõ ràng với mô hình ngôn ngữ lớn, prompt zero-shot đã đủ định hướng. 4 ví dụ few-shot không cung cấp thêm tri thức mới mà chỉ làm phình to token đầu vào, gây lãng phí chi phí mà không cải thiện chất lượng.
3. **Dự đoán `tradeoff` (Đạt chuẩn SLA p95 < 50 ms):**
   - *Đoán:* BoW + LogReg và Encoder MiniLM + LogReg sẽ cùng đạt chuẩn SLA 50 ms.
   - *Kết quả:* Chỉ có các mô hình BoW (LogReg và MLP) đạt chuẩn SLA; toàn bộ các mô hình MiniLM đều có p95 từ 68 - 72 ms (vượt quá 50 ms).
   - *Nguyên nhân:* Trên môi trường Colab không dùng GPU để batching, việc gọi tuần tự từng tin qua 6 tầng Transformer của MiniLM trên CPU tốn thời gian tính toán lớn hơn ước tính ban đầu.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
- **Model tụt điểm nhiều nhất:** `BoW + MLP` tụt sâu nhất (F1 test thường 0.919 tụt xuống **0.286**, Accuracy chỉ còn **50%**). Kế tiếp là `BoW + LogReg` (F1 0.889 tụt xuống **0.333**, Accuracy **52%**).
- **Nhóm case gây thảm họa nhất cho BoW:** "Spam giọng tự nhiên" (BoW+MLP đúng 0%), "Né từ khoá" (đúng 0%), và "Thao túng / lừa đảo" (đúng 12%).
- **2 case cụ thể và cơ chế khiến model sai:**
  1. *Case #29 (Né từ khoá):* `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"`
     - *Cơ chế sai:* Kẻ tấn công thay chữ cái bằng số ('0' thay cho 'o', '4' thay cho 'a'). Mô hình BoW tách từ dựa trên khoảng trắng và từ điển cố định, khiến các từ này trở thành token lạ ngoài từ điển (Out-Of-Vocabulary). BoW xem đây là các từ vô nghĩa, tổng trọng số spam bằng 0 nên phân loại sai thành Ham.
  2. *Case #21 (Thao túng / lừa đảo):* `"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"`
     - *Cơ chế sai:* Lừa đảo phi kỹ thuật mạo danh người thân. Tin nhắn chỉ chứa từ ngữ sinh hoạt gia đình ("Mum", "phone", "toilet", "friend"), không chứa bất kỳ từ khóa spam truyền thống nào ("free", "won", "claim"). BoW hoàn toàn mù trước thủ đoạn ngữ cảnh này.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
- **Đề bài nhận được (Đề #1 - SEED 1848):** Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1 (BoW + LogReg). Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
- **Cách làm:** Trích xuất xác suất dự đoán $P(\text{spam} \mid x) = \sigma(w^T x + b)$ trên tập test thông qua `predict_proba`. Dùng vòng lặp quét qua 3 ngưỡng $[0.2, 0.5, 0.8]$, so sánh `probs >= th` để tạo nhãn nhị phân và tính các chỉ số qua hàm `metrics`.
- **Kết quả thực nghiệm:**
  - Ngưỡng 0.2: Precision = 0.816 | Recall = 0.913 | F1 = 0.862
  - Ngưỡng 0.5: Precision = 0.987 | Recall = 0.815 | F1 = 0.893 (F1 tối ưu)
  - Ngưỡng 0.8: Precision = **1.000** | Recall = 0.630 | F1 = 0.773 (Precision tuyệt đối)
- **Giải thích quyết định cho ngân hàng:**
  - Ngân hàng **CHỌN NGƯỠNG 0.8**.
  - Việc chặn nhầm tin OTP là lỗi False Positive (FP). Hậu quả làm gián đoạn giao dịch tài chính, khóa tài khoản và gây thiệt hại nghiêm trọng cho khách hàng.
  - Tại ngưỡng 0.8, mô hình đạt **Precision = 1.000**, đồng nghĩa **FP = 0** (không có bất kỳ tin nhắn hợp lệ nào bị chặn oan). Đổi lại, Recall giảm còn 0.630 (chấp nhận để lọt một số tin rác), nhưng đây là sự đánh đổi hoàn toàn bắt buộc và tối ưu trong nghiệp vụ ngân hàng.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
- **Yêu cầu:** Lọc 10 triệu SMS/ngày (~116 tin/giây), cam kết SLA p95 < 50 ms/tin.
- **Phương án lựa chọn:** Hệ thống lai 2 tầng (**Confidence Router**):
  - *Tầng 1 (Lọc nhanh - Cửa ngõ):* Dùng mô hình `2. BoW + MLP`.
  - *Tầng 2 (Lọc sâu - Chuyên gia):* Dùng mô hình `4d. LLM nhỏ + Prompt phòng thủ`.
- **Dẫn chứng số liệu:**
  - Mô hình BoW + MLP có p95 < 2 ms (đáp ứng xuất sắc yêu cầu < 50 ms), thông lượng hàng nghìn tin/giây và chi phí $0. Tầng này sẽ xử lý dứt điểm ~90-95% các tin nhắn có độ tự tin cao.
  - Nếu dùng 100% LLM, chi phí mỗi tháng lên tới 99,000 USD (~2.4 tỷ VNĐ) và p95 lên tới 1805 ms (vi phạm hoàn toàn SLA 50 ms).
  - Với Router, chỉ 5-10% tin nhắn rơi vào vùng lưỡng lự $P \in [0.2, 0.8]$ mới được đẩy sang LLM, giúp giảm 90-95% chi phí API (chỉ còn khoảng $5,000 - $10,000/tháng).
- **Chấp nhận đánh đổi:** Chấp nhận kiến trúc phức tạp hơn (quản lý 2 pipeline song song, xây dựng hàng đợi bất đồng bộ cho Tầng 2) để đổi lấy một hệ thống vừa thỏa mãn SLA thời gian thực, vừa tiết kiệm ngân sách, vừa có khả năng chặn đứng các cuộc tấn công lừa đảo tinh vi nhất.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 — TODO 4.1 đến 4.4 | Cú pháp triển khai call_openai, build_classification_prompt, parse_label và llm_classify_batch theo template | Cung cấp các hàm hoàn chỉnh kèm xử lý regex parse nhãn | Đã kiểm chứng qua các hàm test giả lập, kiểm tra không dùng max_tokens cho GPT-6 |
| Phần 4 — Cấu hình Provider | Xử lý lỗi RateLimit 429 của OpenAI và cách chuyển đổi endpoint sang OpenRouter | Hướng dẫn cấu hình base_url openrouter.ai/api/v1 và chọn model tương thích | Tự nạp OPENROUTER_API_KEY vào Secrets, test chạy các thử nghiệm |
| Thử thách cá nhân Đề #1 | Giải thích góc độ ngân hàng với tin OTP | Phân tích sự đánh đổi Precision/Recall góc độ ngân hàng với tin OTP| Hoàn thiện test thử thách cá nhân |
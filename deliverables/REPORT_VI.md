# Quy luật mở rộng (Scaling Laws) cho truy hồi dày (Dense Retrieval): thí nghiệm minh họa thu nhỏ, kèm kiểm toán

*Báo cáo đồng hành cho bài SIGIR '24 "Scaling Laws For Dense Retrieval" (Fang, Zhan, Ai, Mao, Su, Chen, Liu; arXiv 2403.18684v2). Mọi con số trong báo cáo này được sinh tự động bằng script từ `data/results/*.json`, `data/results.csv` và `data/results_summary.csv`. Dấu thập phân dùng dấu chấm, giống bản tiếng Anh.*

## 0. Hộp kết quả kiểm toán

| Nhóm | Phán quyết |
|---|---|
| A. Đề môn học | PASS CÓ LƯU Ý |
| B. Lần giao việc đầu tiên | PASS CÓ LƯU Ý |
| C. Độ trung thực với bài báo | PASS CÓ LƯU Ý: định nghĩa khớp, độ dài huấn luyện thì không |
| D. Vệ sinh thực nghiệm | PASS CÓ LƯU Ý |

Ma trận tuân thủ: 42 mục = 24 MATCH, 12 DEVIATION-JUSTIFIED, 3 DEVIATION-IMPACTFUL, 3 UNVERIFIED (bảng đầy đủ: `AUDIT_COMPLIANCE_MATRIX.md`).

* **DEVIATION-IMPACTFUL (lệch có ảnh hưởng):** B8 (results.csv, SUMMARY.md, CHANGES.md); C7 (Optimisation length); C9 (Metric: contrastive entropy = NLL of the positive among 1+256 candidates).
* **UNVERIFIED (chưa xác minh được):** A4 (Group of 2–3 students); C18 (Mixed precision); D6 (Train loss vs eval loss comparable (to discuss over-fitting)?).

**Nên viết:** "trong một thiết lập thu nhỏ mạnh, loss biến đổi cùng chiều với bài báo". **Không được viết** rằng đã tái hiện bài báo.

## 1. Tóm tắt điều hành

**Bài báo nói** (Sec. 1, Sec. 4.1–4.2, Fig. 3–4): loss kiểm thử của mô hình truy hồi dày, đo bằng "contrastive entropy" (entropy đối sánh) liên tục, tuân theo quy luật lũy thừa theo kích thước mô hình và số cặp được gán nhãn.

**Thí nghiệm của chúng ta cho thấy** (Sec. 6 bên dưới): với 5 cỡ BERT (N từ 512,128 đến 86,235,648 tham số non-embedding), D = 10,000 cặp huấn luyện và 3 seed, loss trung bình giảm từ 0.7265 xuống 0.1581 nat. Với BERT-Small và D từ 2,500 đến 40,000 cặp, loss giảm từ 0.3658 xuống 0.1596. Quy luật lũy thừa 2 tham số khớp tốt cả hai chuỗi (R² trên thang log là 0.986 và 0.989; số mũ 0.299 và 0.285), và số mũ ổn định khi bỏ từng điểm (0.259–0.320 và 0.267–0.294).

**Lưu ý quan trọng:** (1) các lần chạy của chúng ta kéo dài 60–939 bước tối ưu, còn bài báo cố định 10,000 bước (Sec. 3.4); (2) số mũ 3 tham số của chuỗi dữ liệu không ổn định (0.260–0.770 khi bỏ từng điểm); (3) hai cặp điểm liền kề (Small với Medium, D = 10,000 với 20,000) không tách biệt khi so với độ dao động giữa các seed; (4) thước đo là negative log-likelihood (NLL, đơn vị nat), **không phải** perplexity: nhãn trục "Contrastive Perplexity" [OLD-LABEL] trong biểu đồ gốc là sai và đã được sửa ở đây; đoán ngẫu nhiên cho loss ln(257) = 5.549.

## 2. Kiến thức nền cho người mới

**Truy hồi thông tin (Information Retrieval, IR)** là tìm các tài liệu (ở đây: đoạn văn ngắn) trả lời một truy vấn (query). **Truy hồi dày (dense retrieval)** làm việc đó bằng mạng nơ-ron: một *bộ mã hóa (encoder)* biến mỗi truy vấn và mỗi đoạn văn thành một dãy số (vector, còn gọi là *embedding*). Đoạn văn được xem là liên quan khi vector của nó hướng gần giống vector truy vấn; điểm số là tích vô hướng (dot product). Vì vector của đoạn văn có thể tính trước một lần, việc tìm trong hàng triệu đoạn rất nhanh.

*Ví dụ.* Truy vấn "capital of California" và đoạn "Sacramento is the capital of California" phải có điểm cao; đoạn nói về bóng đá phải có điểm thấp (Hình F2).

**Huấn luyện bằng contrastive loss (loss đối sánh).** Mô hình thấy một truy vấn cùng một đoạn đúng được gán nhãn (*positive*) và nhiều đoạn sai ngẫu nhiên (*negatives*, mẫu âm). Điểm của mọi ứng viên được đổi thành xác suất bằng softmax. Loss là âm logarit của xác suất dành cho đoạn đúng: bằng 0 nếu mô hình dồn hết xác suất vào đoạn đúng và lớn nếu không. Đây là *negative log-likelihood (NLL)*, đo bằng *nat* (logarit tự nhiên). Với K ứng viên, mô hình đoán ngẫu nhiên có loss ln(K); ở đây K = 1 + 256 = 257, nên ln(K) = 5.549 (Hình F3, F10).

**Quy luật mở rộng (scaling law).** Ở nhiều mạng nơ-ron, loss giảm theo luật lũy thừa trơn, L ≈ a·X^(−α), khi kích thước mô hình N hoặc lượng dữ liệu D tăng. Trên trục log–log đó là đường thẳng có độ dốc −α. Nếu luật đúng, các thí nghiệm nhỏ, rẻ có thể dự đoán thí nghiệm lớn, nên chủ đề này quan trọng; bài báo cho rằng nó còn giúp phân bổ ngân sách giữa gán nhãn và tính toán (Sec. 5).

## 3. Bài báo nói gì

*(Mọi phát biểu đều dẫn mục, hình, bảng hoặc phương trình của arXiv 2403.18684v2. Metadata được đối chiếu với Crossref và arXiv API: SIGIR '24, Washington DC, 14–18 tháng Bảy 2024, trang 1339–1349, DOI 10.1145/3626772.3657743.)*

**Bài báo nói:** các câu hỏi nghiên cứu là kích thước mô hình, lượng dữ liệu gán nhãn và chất lượng nhãn ảnh hưởng thế nào tới truy hồi dày (Sec. 1, phần mở đầu Sec. 4).

**Bài báo nói:** bộ mã hóa là một Transformer tiền huấn luyện có thêm lớp chiếu (projection layer), dùng chung cho truy vấn và đoạn văn, chấm điểm bằng tích vô hướng (Sec. 3.1–3.2, Eq. 1–2, 4); huấn luyện dùng mẫu âm ngẫu nhiên và mẫu âm trong batch, 256 mẫu âm mỗi bước, cố định 10,000 bước (Sec. 3.4, Eq. 3). Tốc độ học, kích thước batch và temperature không được nêu trong văn bản (UNVERIFIED trong bài báo).

**Bài báo nói:** thước đo là "contrastive entropy", tức NLL của Eq. 5 với 256 mẫu âm ngẫu nhiên cho mỗi cặp kiểm thử; một tập con 100,000 đoạn được dùng trong phần nghiên cứu thước đo; nó tương quan với MAP@10, NDCG@10 và Recall@1000 (Sec. 3.5, Fig. 2), với một bước nhảy "emergent" có thể có gần 0.25.

**Bài báo nói:** N là số tham số non-embedding, từ 0.5M (BERT-Tiny) đến 82M (BERT-Base) (Sec. 3.2 và chú thích 2; chú giải Fig. 6); phần Giới thiệu viết "0.5 to 87 million" (Sec. 1), một mâu thuẫn nội bộ mà chúng tôi không giải quyết được. D là số cặp truy vấn–đoạn văn được gán nhãn (Sec. 3.1, 4.2).

**Bài báo nói:** luật theo kích thước mô hình L(N) = (A/N)^α + δ (Eq. 6) với MS MARCO: A = 3.22e4, α = 0.53, δ = 0.04, R² = 0.991 (Table 1, Fig. 3). Luật theo dữ liệu L(D) = (B/D)^β + δ (Eq. 7) với B = 3.49e3, β = 1.05, δ = 0.05, R² = 0.954 (Table 2, Fig. 4), khớp trên BERT-Base với lượng nhãn tăng tới toàn bộ. Luật kết hợp (Eq. 8) có A ≈ 3.6e4, B ≈ 7.1e3, α ≈ 0.56, β ≈ 1.31, δ ≈ 0.03 (Sec. 4.4, Eq. 9–10, Fig. 6). Hằng số δ được hiểu là loss không thể giảm thêm (Sec. 4.1).

**Bài báo nói:** quy luật đúng với nhãn người và với ba loại nhãn sinh tự động (ICT, docT5query, ChatGLM3); ICT có độ dốc nhỏ nhất, và chú thích Fig. 5 nói ChatGLM3 dốc nhất và vượt nhãn người ở 500K trong khi văn bản nói nhãn người cho mô hình tốt nhất (Sec. 4.3, Fig. 5). Với luật kết hợp và mô hình chi phí, ngân sách $20,000 ưu tiên mô hình khoảng 13 tỷ tham số nếu bỏ qua chi phí suy luận, và mô hình cỡ triệu tham số nếu tính cả suy luận (Sec. 5, Fig. 7–8).

**Bài báo nói (hạn chế, Sec. 6):** chỉ đánh giá trong cùng miền, không có hard negatives hay distillation, chỉ bộ truy hồi một vector. Danh mục arXiv ghi rằng v2 "fixes a bug in the experiments".

## 4. Thí nghiệm của chúng ta

**Mục tiêu.** Một minh họa rẻ cho hai xu hướng cơ bản (loss theo N, loss theo D) kèm ước lượng sai số trung thực. Đây **không** phải là tái hiện.

| Aspect | Paper (section / figure) | Our experiment |
|---|---|---|
| Task / data | MS MARCO passage (8.8M passages, 0.5M train queries) and T2Ranking (Sec. 3.3) | MS MARCO only; passage file truncated to 8,326,970 passages (94.18%) |
| Models | 24 Google BERT checkpoints 0.5M–82M non-emb. params; ERNIE for T2Ranking (Sec. 3.2) | 5 BERT checkpoints: Tiny, Mini, Small, Medium, Base (same family as the paper's red stars, Fig. 3) |
| N definition | non-embedding parameters (Sec. 3.2, footnote 2) | non-embedding incl. projection layer and unused BERT pooler (= the paper's Fig. 6 legend sizes when divided by 2^20) |
| D definition | number of annotated query–passage pairs (Sec. 3.1, 4.2) | number of (qid, pid) training pairs |
| Architecture | shared encoder + projection to 768; CLS or mean pooling (Sec. 3.2, Eq. 2) | shared BERT + mean pooling + Linear(H→768); no normalisation |
| Training loss | softmax cross-entropy, random + in-batch negatives (Sec. 3.4, Eq. 3) | same: 1 positive + 256 random negatives shared by the batch; mean CE; no temperature |
| Optimisation length | fixed 10,000 steps (Sec. 3.4) | 3 epochs → 60–939 steps (depends on D); 237 steps for the model sweep |
| lr / batch / optimiser | not stated in the paper (UNVERIFIED) | lr 1e-4, batch 128, AdamW, linear decay, warmup 10 steps, bf16 |
| Metric | contrastive entropy = −log softmax of the positive among 1+256 candidates (Sec. 3.5, Eq. 5) | same quantity (mean NLL, nats), K = 257 |
| Evaluation set | dev queries (~7,000, Sec. 3.3); 100,000-passage sub-corpus used for the metric study (Sec. 3.5) | first 1,000 dev.small queries by qid (1,086 pairs); 100,000-passage pool; 5 repeats; fixed eval seed 1 |
| Checkpoint reported | best test loss during training, no early stopping (Sec. 4.1) | final checkpoint after epoch 3 |
| Model sweep | full training set, many models (Sec. 4.1, Fig. 3) | D = 10,000 pairs, 5 models |
| Data sweep | BERT-Base, up to all ~500K annotations (Sec. 4.2, Fig. 4) | BERT-Small, D = 2,500 … 40,000 |
| Fit form | L = (A/N)^α + δ (Eq. 6); L = (B/D)^β + δ (Eq. 7); joint Eq. 8 | 3-param a·X^-α + c (same family; a = A^α) and a 2-param a·X^-α (extra, main) |
| Seeds | not stated (UNVERIFIED) | 3 training seeds 42/43/44; evaluation seed fixed |
| Annotation quality, budget allocation | Sec. 4.3 (ICT, docT5query, ChatGLM3), Sec. 5 | not reproduced |

**Dữ liệu.** MS MARCO passage ranking. Tệp corpus có 8,326,970 đoạn (94.18% bộ đầy đủ; tệp bị cắt ở đuôi, id 0…8,326,969). Trong 532,751 cặp huấn luyện duy nhất, 3,311 cặp (0.62%) trỏ tới đoạn bị thiếu nên bị loại, còn 529,440 cặp trên 499,872 truy vấn. D cặp là D cặp đầu của một phép xáo trộn có seed.

**Mô hình.** Năm mô hình BERT thu nhỏ của Google, số tham số ở Bảng A1 phụ lục; một encoder dùng chung, mean pooling, Linear(H→768), điểm tích vô hướng, không chuẩn hóa, không temperature (Hình F2).

**Huấn luyện.** 3 epoch, lr 1e-4, batch 128, AdamW (weight decay 0.001, β2 0.999), giảm tuyến tính, 10 bước khởi động, cắt gradient 1.0, bf16, gradient checkpointing. Mỗi bước, 256 đoạn ngẫu nhiên của corpus được cả batch dùng chung làm mẫu âm (K = 257). Thiết lập giống hệt nhau cho cả 27 lần chạy (đọc từ `training_args.bin` của từng lần chạy).

**Đánh giá.** 1,000 truy vấn dev.small có qid nhỏ nhất (1,086 cặp truy vấn–positive); một pool cố định 100,000 đoạn lấy từ 8,325,884 đoạn không phải positive của các truy vấn đó; với mỗi cặp, 256 mẫu âm lấy từ pool, lặp 5 lần (5,430 lần lấy mẫu cho mỗi mô hình); seed đánh giá 1 cho mọi mô hình; trung bình theo cặp là loss được báo cáo.

**Seed.** Seed huấn luyện 42, 43, 44 làm đổi khởi tạo lớp chiếu, tập con và thứ tự dữ liệu, và các mẫu âm khi huấn luyện; seed đánh giá không đổi.

**Phần cứng và thời gian.** Một NVIDIA RTX 5880 Ada Generation (49,140 MiB; driver 580.178.04, nvidia-smi on 2026-10-29); Intel Core Ultra 9 285K (24 logical CPUs), 93 GiB RAM; PyTorch 2.11.0+cu128, Transformers 4.52.4; bộ nhớ GPU đỉnh (allocated) 4645.6 MiB. Tổng trên 27 lần chạy: huấn luyện 1966 s, đánh giá 1368 s, tổng 3495 s (58.2 phút). Phiên điều phối cuối (seed 43/44 và các cấu hình mới) kéo dài 51.1 phút.

![F1. Quy trình](figures/F1_workflow.png)

## 5. Kết quả kiểm toán

### 5.1 Tóm tắt tuân thủ

Phán quyết và số lượng nằm trong hộp ở Mục 0; ma trận đầy đủ kèm bằng chứng là `AUDIT_COMPLIANCE_MATRIX.md`. Nhóm C là nhóm giới hạn những gì được phép khẳng định.

### 5.2 Phát hiện về code

**Code làm:** kiến trúc trong `modeling.py` không đổi so với repo gốc (tệp giống hệt từng byte). Mean pooling, Linear(H→768), tích vô hướng thuần, không temperature, cross-entropy trên 1 positive và 256 mẫu âm ngẫu nhiên dùng chung, trung bình theo batch (`modeling.py` dòng 47–72, 112–118).

**Code làm:** `contrastive_perplexity` tính, cho mỗi cặp (truy vấn, positive), cross-entropy trên 1 + 256 ứng viên rồi lấy trung bình các loss theo cặp và 5 lần lặp. Không có `exp`, nên giá trị là NLL tính bằng nat, không phải perplexity. Mức đoán ngẫu nhiên là ln(257) = 5.549; mô hình tệ nhất của chúng ta bằng 0.131 mức đó và mô hình tốt nhất bằng 0.028. Chỉ để tham khảo, exp của trung bình là 2.068 cho mô hình tệ nhất và 1.171 cho mô hình tốt nhất.

**Code làm:** huấn luyện và đánh giá đều dùng K = 257 và cùng công thức, nhưng loss huấn luyện trong log là trung bình chạy của cả quá trình, tính ở chế độ train (dropout) với bf16 trên các cặp đã thấy. Loss dev cuối tính trên mô hình cuối ở chế độ eval với fp32 trên truy vấn giữ lại. Hai đại lượng này **không so sánh được**; đường cong huấn luyện (F8) cho thấy loss huấn luyện vẫn đang giảm ở cuối, và loss huấn luyện ở cửa sổ cuối của BERT-Small (0.0555) thấp hơn nhiều so với loss dev (0.2345), nhưng không thể gọi đó là overfitting nếu thiếu phép đo like-for-like (ví dụ đánh giá mô hình cuối ở chế độ eval trên truy vấn huấn luyện được giữ lại, chưa làm ở đây).

**Code làm:** thay đổi so với repo gốc chỉ gồm đường dẫn, đọc tệp bền hơn, sửa tham số thiếu, lưu checkpoint, nạp sẵn dữ liệu tùy chọn, giới hạn 1,000 truy vấn đánh giá và bộ điều phối nhiều seed (xem `CHANGES.md`). Không thay đổi nào làm đổi loss, mô hình hay định nghĩa thước đo; giới hạn số truy vấn và kích thước pool là thu nhỏ có chủ ý.

**Kiểm toán cho thấy:** không có truy vấn nào rò từ huấn luyện sang đánh giá (qid train ∩ qid dev = 0, ∩ qid đánh giá = 0, văn bản truy vấn trùng hệt = 0). Các tập con D lồng nhau trong cùng một seed và độc lập giữa các seed (trùng 10,000 cặp là 203–207 so với 189 kỳ vọng ngẫu nhiên). Pool đánh giá giống hệt nhau cho mọi mô hình (kiểm tra dựng lại, tiền tố sha1 f965e175b62f). Tập con 1,000 truy vấn đánh giá hơi kém đại diện: độ dài truy vấn trung bình 5.30 từ so với 6.03 ở các truy vấn dev còn lại.

**Kiểm toán cho thấy:** 5.8% corpus bị thiếu là các id đoạn cao nhất. Các cặp huấn luyện bị mất thuộc về truy vấn hơi ngắn hơn (5.74 so với 5.97 từ); dev không mất gì. Ảnh hưởng được đánh giá là thấp, dù việc id có mang thứ tự chủ đề hay không là UNVERIFIED. Tệp cặp huấn luyện có 10 dòng trùng (532,761 dòng, 532,751 cặp duy nhất).

**Kiểm toán cho thấy:** cách đếm non-embedding dùng cho N bao gồm cả BERT pooler không dùng đến và lớp chiếu. Theo đơn vị 2^20, năm cỡ là 0.5, 3.3, 12.7, 24.7, 82.2 M, đúng bằng nhãn trong chú giải Fig. 6 của bài báo, nên quy ước đếm khớp với bài báo. Trục x của chúng ta dùng đơn vị 10^6.

### 5.3 Tính lại độc lập (từ 27 tệp JSON)

Trung bình và độ lệch chuẩn (SD, ddof = 1) lấy từ các tệp JSON, khớp với `results.csv` (chênh lệch tối đa 0.0e+00) và `results_summary.csv` (chỉ khác do làm tròn). Có 9 cấu hình khác nhau (không phải 10: BERT-Small với D = 10,000 thuộc cả hai chuỗi). Khớp đường: cả sáu giá trị trong SUMMARY được tính lại khớp đến độ chính xác đã in.

| Sweep | Fit | a | exponent | c | R² (log) | R² (linear) | notes |
|---|---|---|---|---|---|---|---|
| Model size N | 2-param  L = a·X^-α (OLS in log–log) | 34.79 | 0.299 | — | 0.9858 | 0.9867 | main fit |
| Model size N | 3-param  L = a·X^-α + c (curve_fit) | 206.29 | 0.440 | 0.093 | — | 0.9991 | bound flags: none; 5 points, 3 parameters |
| Data size D | 2-param  L = a·X^-α (OLS in log–log) | 3.32 | 0.285 | — | 0.9887 | 0.9882 | main fit |
| Data size D | 3-param  L = a·X^-α + c (curve_fit) | 8.97 | 0.444 | 0.084 | — | 0.9918 | bound flags: none; 5 points, 3 parameters |

Các đường khớp 3 tham số đạt cùng một cực trị từ cả 27 điểm khởi tạo (mô hình) và 27 (dữ liệu), có hoặc không có biên, và không chạm biên nào. **Tuy nhiên** chúng chỉ có 2 bậc tự do: bỏ từng điểm cho số mũ 3 tham số 0.426–0.492 (mô hình) và 0.260–0.770 (dữ liệu, trong đó 1 lần khớp chạm biên c ≥ 0); khớp riêng từng seed cho 0.367–0.489 và 0.247–0.669. Số mũ 2 tham số ổn định hơn nhiều: bỏ từng điểm 0.259–0.320 và 0.267–0.294, theo từng seed 0.292–0.307 và 0.258–0.310 (Hình F9, Bảng A2–A3). Nếu đếm tham số không kể lớp đầu ra thì số mũ của mô hình là 0.288 thay vì 0.299.

**Các điểm liền kề** (chỉ mô tả; n = 3 cho mỗi điểm):

| Sweep | From | To | Mean ± SD (from) | Mean ± SD (to) | Difference | ±SD intervals overlap | Seed min–max ranges overlap | Seeds (of 3) where larger setting is better |
|---|---|---|---|---|---|---|---|---|
| model | BERT-Tiny D=10000 | BERT-Mini D=10000 | 0.7265 ± 0.0145 | 0.3707 ± 0.0220 | 0.3558 | no | no | 3 |
| model | BERT-Mini D=10000 | BERT-Small D=10000 | 0.3707 ± 0.0220 | 0.2345 ± 0.0129 | 0.1362 | no | no | 3 |
| model | BERT-Small D=10000 | BERT-Medium D=10000 | 0.2345 ± 0.0129 | 0.2149 ± 0.0155 | 0.0196 | yes | yes | 3 |
| model | BERT-Medium D=10000 | BERT-Base D=10000 | 0.2149 ± 0.0155 | 0.1581 ± 0.0177 | 0.0568 | no | no | 3 |
| data | BERT-Small D=2500 | BERT-Small D=5000 | 0.3658 ± 0.0187 | 0.2830 ± 0.0063 | 0.0828 | no | no | 3 |
| data | BERT-Small D=5000 | BERT-Small D=10000 | 0.2830 ± 0.0063 | 0.2345 ± 0.0129 | 0.0485 | no | no | 3 |
| data | BERT-Small D=10000 | BERT-Small D=20000 | 0.2345 ± 0.0129 | 0.2059 ± 0.0217 | 0.0286 | yes | yes | 3 |
| data | BERT-Small D=20000 | BERT-Small D=40000 | 0.2059 ± 0.0217 | 0.1596 ± 0.0069 | 0.0463 | no | no | 3 |

Trong 8 cặp liền kề, 2 cặp có khoảng ±SD chồng nhau: Small với Medium (chênh 0.0196) và D = 10,000 với 20,000 (chênh 0.0286). Thiết lập lớn hơn tốt hơn ở cả 3 seed cho mọi cặp, nhưng với n = 3 điều này chỉ mang tính mô tả.

**So sánh với số mũ của bài báo** (các đại lượng cùng một họ công thức; chế độ thực nghiệm thì không giống nhau):

| Quantity | Paper (MS MARCO; Tables 1–2) | Ours | Comparable? |
|---|---|---|---|
| Model-size exponent α (3-param family; Table 1, Sec. 4.1) | 0.53 | 0.440 | No: see caveats |
| Model-size constant δ_N / c (Table 1) | 0.04 | 0.093 | No: see caveats |
| Model-size fit R² (Table 1) | 0.991 | 0.9991 | Descriptive only (5 points, 3 parameters) |
| Data-size exponent β (3-param family; Table 2, Sec. 4.2) | 1.05 | 0.444 | No: see caveats |
| Data-size constant δ_D / c (Table 2) | 0.05 | 0.084 | No: see caveats |
| Data-size fit R² (Table 2) | 0.954 | 0.9918 | Descriptive only |
| 2-param exponents (no constant) | not reported | 0.299 (model), 0.285 (data) | No: the paper has no such fit |

Vì sao không so sánh được like-for-like: (i) 237 bước so với 10,000 (Sec. 3.4); (ii) D từ 2,500 đến 40,000 trên BERT-Small so với tới khoảng 500K trên BERT-Base (Sec. 4.2, Fig. 4); (iii) checkpoint cuối so với loss kiểm thử tốt nhất (Sec. 4.1); (iv) tập đánh giá nhỏ hơn; (v) khớp 5 điểm với 3 tham số tự do. Số mũ của chúng ta thấp hơn bài báo, và cả hai hằng số (0.093, 0.084) lớn hơn của bài báo (0.04, 0.05). Chúng tôi không rút kết luận nào từ các khác biệt này.

## 6. Kết quả thực nghiệm

| Sweep | Model | N (non-emb., incl. fc) | D (pairs) | Steps | Seeds | Mean ± SD (nats) | Seed 42 / 43 / 44 | exp(mean) (supplementary) | Train time (s) | Eval time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| model sweep | BERT-Tiny | 512,128 | 10,000 | 237 | 3 | 0.7265 ± 0.0145 | 0.7424 / 0.7232 / 0.7140 | 2.068 | 21.3 | 17.3 |
| model sweep | BERT-Mini | 3,422,208 | 10,000 | 237 | 3 | 0.3707 ± 0.0220 | 0.3759 / 0.3466 / 0.3896 | 1.449 | 34.7 | 22.5 |
| model sweep | BERT-Small | 13,266,176 | 10,000 | 237 | 3 | 0.2345 ± 0.0129 | 0.2281 / 0.2493 / 0.2261 | 1.264 | 41.4 | 32.0 |
| model sweep | BERT-Medium | 25,875,712 | 10,000 | 237 | 3 | 0.2149 ± 0.0155 | 0.1977 / 0.2278 / 0.2191 | 1.240 | 75.8 | 60.2 |
| model sweep | BERT-Base | 86,235,648 | 10,000 | 237 | 3 | 0.1581 ± 0.0177 | 0.1785 / 0.1466 / 0.1492 | 1.171 | 201.2 | 193.7 |
| data sweep | BERT-Small | 13,266,176 | 2,500 | 60 | 3 | 0.3658 ± 0.0187 | 0.3618 / 0.3494 / 0.3862 | 1.442 | 11.2 | 31.6 |
| data sweep | BERT-Small | 13,266,176 | 5,000 | 120 | 3 | 0.2830 ± 0.0063 | 0.2831 / 0.2892 / 0.2766 | 1.327 | 21.4 | 32.0 |
| data sweep | BERT-Small | 13,266,176 | 10,000 | 237 | 3 | 0.2345 ± 0.0129 | 0.2281 / 0.2493 / 0.2261 | 1.264 | 41.4 | 32.0 |
| data sweep | BERT-Small | 13,266,176 | 20,000 | 471 | 3 | 0.2059 ± 0.0217 | 0.1834 / 0.2266 / 0.2078 | 1.229 | 82.8 | 33.4 |
| data sweep | BERT-Small | 13,266,176 | 40,000 | 939 | 3 | 0.1596 ± 0.0069 | 0.1656 / 0.1612 / 0.1520 | 1.173 | 165.6 | 33.4 |

Bảng: trung bình ± SD qua 3 seed của contrastive entropy (NLL, nat, K = 257, càng thấp càng tốt). `exp(mean)` chỉ để bổ sung. Hàng BERT-Small với D = 10,000 xuất hiện ở cả hai chuỗi; đó là cùng 3 lần chạy.

![F4. Quy luật theo kích thước mô hình](figures/F4_model_scaling.png)

**Hình F4.** Loss theo số tham số non-embedding (log–log). Xanh: trung bình ± SD qua 3 seed; xám: từng seed; nét liền: khớp 2 tham số (chính); nét đứt: khớp 3 tham số.

![F5. Quy luật theo lượng dữ liệu](figures/F5_data_scaling.png)

**Hình F5.** Loss theo số cặp huấn luyện với BERT-Small. Lưu ý số bước tăng theo D (60 đến 939).

**Diễn giải.** Loss giảm theo cả N và D. Từ Tiny đến Base loss trung bình giảm 4.6 lần; từ D = 2,500 đến 40,000 giảm 2.29 lần. SD giữa các seed chiếm từ 2.0% đến 11.2% giá trị trung bình, nên nhận định "độ lệch nhỏ" trong SUMMARY gốc là quá mạnh với các điểm sát nhau. Mọi mô hình đều thấp hơn xa mức đoán ngẫu nhiên (F10).

![F6. Phần dư](figures/F6_residuals.png)

**Hình F6.** Phần dư của hai đường khớp. Đường 2 tham số dự đoán thấp hơn thực tế ở mô hình nhỏ nhất; đường 3 tham số bám sát mọi điểm hơn nhưng có nhiều tự do hơn.

![F7. Bảng kết quả](figures/F7_results_table.png)

![F8. Đường cong huấn luyện](figures/F8_training_curves.png)

**Hình F8.** Loss huấn luyện theo cửa sổ 10 bước và loss dev cuối (hình thoi). Dữ liệu, chế độ và cách lấy trung bình khác nhau: không so sánh được, không kết luận overfitting.

![F9. Độ ổn định khi bỏ từng điểm](figures/F9_loo_stability.png)

![F10. Mức đoán ngẫu nhiên](figures/F10_chance_level.png)

![F11. Thời gian huấn luyện](figures/F11_training_time.png)

![F2. DREncoder](figures/F2_architecture.png)

![F3. Minh họa contrastive loss](figures/F3_contrastive_loss.png)

**Hình F3** dùng điểm số tự đặt để minh họa phép tính: p(positive) = 0.9127 cho loss 0.0914 nat.

## 7. Thảo luận

**Có cùng chiều với bài báo không?** Trong phạm vi nhỏ này, có về chiều: loss giảm khi N và D tăng (bài báo: Fig. 3, Fig. 4). Không được nói mạnh hơn: số mũ, hằng số và độ khớp không so sánh được với Tables 1–2 (Mục 5.3), và không có gì ở đây kiểm tra chất lượng nhãn (Sec. 4.3) hay phân bổ ngân sách (Sec. 5).

**Khác biệt so với bài báo.** Lịch huấn luyện ngắn (237 bước cho chuỗi mô hình), báo cáo checkpoint cuối chứ không phải tốt nhất, đánh giá nhỏ hơn, BERT-Small cho chuỗi dữ liệu, một bộ dữ liệu. Ở các lần chạy ngắn như vậy mô hình lớn học nhanh hơn trên mỗi bước (F8), nên đường theo kích thước mô hình trộn lẫn dung lượng và tốc độ học; chuỗi dữ liệu trộn lẫn dữ liệu và số bước.

**Hạn chế.** Năm điểm mỗi chuỗi, ba seed; khớp 3 tham số chỉ có 2 bậc tự do; không kiểm tra ngoại suy; không có hard negatives; corpus bị cắt; tập đánh giá là 1,000 qid nhỏ nhất; tốc độ học được chọn từ hai giá trị với mô hình gần như chưa học (loss 5.4256 và 4.9708 so với mức ngẫu nhiên 5.549).

**Các mối đe dọa tới tính hợp lệ.** *Nội tại:* độ dài huấn luyện khác nhau theo D; seed đánh giá cố định (SD không gồm nhiễu lấy mẫu đánh giá); các lần chạy seed 42 đầu tiên được tạo trước khi bộ điều phối được viết lại (cấu hình của chúng đã được xác minh từ `training_args.bin`). *Ngoại tại:* một bộ dữ liệu, một họ mô hình, chỉ trong cùng miền. *Cấu trúc:* contrastive entropy với 256 mẫu âm ngẫu nhiên đo khả năng phân biệt với mẫu âm ngẫu nhiên, không phải chất lượng xếp hạng (bài báo cũng lưu ý hai thước đo không tương đương, Sec. 6).

## 8. Cách tái lập

```
# huấn luyện + đánh giá (bộ điều phối gốc, chạy ngầm)
export MSMARCO_DIR=<đường dẫn tới các tệp MS MARCO>
cd drscale_exp && nohup python3 run_all.py > logs/run.log 2>&1 &
# kiểm toán và đóng gói (từ deliverables/scripts)
python3 -I audit_data.py; python3 -I audit_model.py; python3 -I audit_config.py
python3 -I recompute.py; python3 -I make_figures.py; python3 -I build_docs.py; python3 -I check_numbers.py
```

Môi trường: Python 3.11.7, PyTorch 2.11.0+cu128, Transformers 4.52.4, NumPy 1.26.4, SciPy 1.16.3, pandas 2.3.3, Matplotlib 3.10.6; repo gốc `jingtaozhan/DRScale`, commit `81453c3b67486641ad282d58f819392a08561e77`. Huấn luyện trên GPU không tái lập từng bit, nên chạy lại sẽ cho loss hơi khác.

## 9. Phụ lục

**A1. Số tham số** (đã kiểm tra bằng cách dựng từng mô hình trên CPU)

| Model | Layers L | Hidden H | Transformer layers only | + BERT pooler (unused) | = encoder non-emb. ("no fc") | + output_fc | = non-emb. incl. fc (used on x-axis) | same in units of 2^20 | Total params |
|---|---|---|---|---|---|---|---|---|---|
| BERT-Tiny | 2 | 128 | 396,544 | 16,512 | 413,056 | 99,072 | 512,128 | 0.5 | 4,484,992 |
| BERT-Mini | 4 | 256 | 3,159,040 | 65,792 | 3,224,832 | 197,376 | 3,422,208 | 3.3 | 11,367,936 |
| BERT-Small | 4 | 512 | 12,609,536 | 262,656 | 12,872,192 | 393,984 | 13,266,176 | 12.7 | 29,157,632 |
| BERT-Medium | 8 | 512 | 25,219,072 | 262,656 | 25,481,728 | 393,984 | 25,875,712 | 24.7 | 41,767,168 |
| BERT-Base | 12 | 768 | 85,054,464 | 590,592 | 85,645,056 | 590,592 | 86,235,648 | 82.2 | 110,072,832 |

**A2. Khớp bỏ từng điểm (leave-one-out)**

| Sweep | Point left out | 2-param exponent | R² (log) | 3-param exponent | 3-param c | bound flag |
|---|---|---|---|---|---|---|
| Model size N | 512,128 | 0.259 | 0.981 | 0.492 | 0.108 | — |
| Model size N | 3,422,208 | 0.303 | 0.987 | 0.468 | 0.106 | — |
| Model size N | 13,266,176 | 0.296 | 0.994 | 0.426 | 0.090 | — |
| Model size N | 25,875,712 | 0.300 | 0.985 | 0.437 | 0.087 | — |
| Model size N | 86,235,648 | 0.320 | 0.989 | 0.452 | 0.101 | — |
| Data size D | 2,500 | 0.267 | 0.984 | 0.260 | 0.000 | c at bound |
| Data size D | 5,000 | 0.292 | 0.992 | 0.353 | 0.042 | — |
| Data size D | 10,000 | 0.285 | 0.990 | 0.432 | 0.081 | — |
| Data size D | 20,000 | 0.294 | 0.995 | 0.465 | 0.083 | — |
| Data size D | 40,000 | 0.276 | 0.978 | 0.770 | 0.166 | — |

**A3. Khớp riêng từng seed**

| Sweep | Seed | 2-param exponent | R² (log) | 3-param exponent | 3-param c | R² (3-param) | bound flag |
|---|---|---|---|---|---|---|---|
| model | 42 | 0.292 | 0.955 | 0.489 | 0.115 | 0.997 | — |
| data | 42 | 0.288 | 0.983 | 0.548 | 0.107 | 0.998 | — |
| model | 43 | 0.299 | 0.984 | 0.469 | 0.109 | 0.995 | — |
| data | 43 | 0.258 | 0.958 | 0.247 | 0.000 | 0.972 | c at bound |
| model | 44 | 0.307 | 0.987 | 0.367 | 0.047 | 0.996 | — |
| data | 44 | 0.310 | 0.966 | 0.669 | 0.124 | 0.979 | — |

**A4. Thuật ngữ.** *NLL / contrastive entropy*: âm logarit của xác suất softmax dành cho positive. *Nat*: đơn vị của NLL khi dùng logarit tự nhiên. *Non-embedding parameters*: tham số không kể embedding từ/vị trí. *SD*: độ lệch chuẩn mẫu với n − 1. *R²*: tỷ lệ phương sai được đường khớp giải thích. *Seed*: số cố định bộ sinh ngẫu nhiên. *bf16*: định dạng số thực 16 bit dùng khi huấn luyện.

**A5. Dữ liệu thô.** `data/results/*.json` (mỗi lần chạy một tệp), `data/results.csv`, `data/results_summary.csv`, `data/figure_data/*.csv`, `data/run.log`, `data/status.json`, `tables/`, `audit/`.

**A6. Nguồn.** Bài báo: arXiv 2403.18684v2 (đọc PDF cục bộ; trích văn bản bằng PyMuPDF). Metadata: Crossref API và arXiv API, lấy ngày 2026-10-29. Xếp hạng: cổng CORE 2023 (SIGIR = A*); danh mục CCF 2026 xem qua một bản sao của bên thứ ba (chưa truy cập trang chính thức). Commit của repo 81453c3b67486641ad282d58f819392a08561e77.

*Ghi chú: tên cột bảng, danh sách mục kiểm toán và bảng thiết lập giữ nguyên tiếng Anh để khớp với các tệp CSV và ma trận tuân thủ.*

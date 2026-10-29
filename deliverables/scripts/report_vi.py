VI = r"""# Quy luật mở rộng (Scaling Laws) cho truy hồi dày (Dense Retrieval): thí nghiệm minh họa thu nhỏ, kèm kiểm toán

*Báo cáo đồng hành cho bài SIGIR '24 "Scaling Laws For Dense Retrieval" (Fang, Zhan, Ai, Mao, Su, Chen, Liu; arXiv 2403.18684v2). Mọi con số trong báo cáo này được sinh tự động bằng script từ `data/results/*.json`, `data/results.csv` và `data/results_summary.csv`. Dấu thập phân dùng dấu chấm, giống bản tiếng Anh.*

## 0. Hộp kết quả kiểm toán

| Nhóm | Phán quyết |
|---|---|
| A. Đề môn học | <<verd_A>> |
| B. Lần giao việc đầu tiên | <<verd_B>> |
| C. Độ trung thực với bài báo | <<verd_C>> |
| D. Vệ sinh thực nghiệm | <<verd_D>> |

Ma trận tuân thủ: <<n_rows>> mục = <<cnt_MATCH>> MATCH, <<cnt_DEVIATION_JUSTIFIED>> DEVIATION-JUSTIFIED, <<cnt_DEVIATION_IMPACTFUL>> DEVIATION-IMPACTFUL, <<cnt_UNVERIFIED>> UNVERIFIED (bảng đầy đủ: `AUDIT_COMPLIANCE_MATRIX.md`).

* **DEVIATION-IMPACTFUL (lệch có ảnh hưởng):** <<impactful_list>>.
* **UNVERIFIED (chưa xác minh được):** <<unverified_list>>.

**Nên viết:** "trong một thiết lập thu nhỏ mạnh, loss biến đổi cùng chiều với bài báo". **Không được viết** rằng đã tái hiện bài báo.

## 1. Tóm tắt điều hành

**Bài báo nói** (Sec. 1, Sec. 4.1–4.2, Fig. 3–4): loss kiểm thử của mô hình truy hồi dày, đo bằng "contrastive entropy" (entropy đối sánh) liên tục, tuân theo quy luật lũy thừa theo kích thước mô hình và số cặp được gán nhãn.

**Thí nghiệm của chúng ta cho thấy** (Sec. 6 bên dưới): với 5 cỡ BERT (N từ <<p_tiny_with_fc>> đến <<p_base_with_fc>> tham số non-embedding), D = 10,000 cặp huấn luyện và 3 seed, loss trung bình giảm từ <<mean_tiny>> xuống <<mean_base>> nat. Với BERT-Small và D từ 2,500 đến 40,000 cặp, loss giảm từ <<mean_d2500>> xuống <<mean_d40000>>. Quy luật lũy thừa 2 tham số khớp tốt cả hai chuỗi (R² trên thang log là <<m_r2log>> và <<d_r2log>>; số mũ <<m_al2>> và <<d_al2>>), và số mũ ổn định khi bỏ từng điểm (<<m_loo2_lo>>–<<m_loo2_hi>> và <<d_loo2_lo>>–<<d_loo2_hi>>).

**Lưu ý quan trọng:** (1) các lần chạy của chúng ta kéo dài <<steps_2500>>–<<steps_40000>> bước tối ưu, còn bài báo cố định <<paper_steps>> bước (Sec. 3.4); (2) số mũ 3 tham số của chuỗi dữ liệu không ổn định (<<d_loo3_lo>>–<<d_loo3_hi>> khi bỏ từng điểm); (3) hai cặp điểm liền kề (Small với Medium, D = 10,000 với 20,000) không tách biệt khi so với độ dao động giữa các seed; (4) thước đo là negative log-likelihood (NLL, đơn vị nat), **không phải** perplexity: nhãn trục "Contrastive Perplexity" [OLD-LABEL] trong biểu đồ gốc là sai và đã được sửa ở đây; đoán ngẫu nhiên cho loss ln(257) = <<lnK>>.

## 2. Kiến thức nền cho người mới

**Truy hồi thông tin (Information Retrieval, IR)** là tìm các tài liệu (ở đây: đoạn văn ngắn) trả lời một truy vấn (query). **Truy hồi dày (dense retrieval)** làm việc đó bằng mạng nơ-ron: một *bộ mã hóa (encoder)* biến mỗi truy vấn và mỗi đoạn văn thành một dãy số (vector, còn gọi là *embedding*). Đoạn văn được xem là liên quan khi vector của nó hướng gần giống vector truy vấn; điểm số là tích vô hướng (dot product). Vì vector của đoạn văn có thể tính trước một lần, việc tìm trong hàng triệu đoạn rất nhanh.

*Ví dụ.* Truy vấn "capital of California" và đoạn "Sacramento is the capital of California" phải có điểm cao; đoạn nói về bóng đá phải có điểm thấp (Hình F2).

**Huấn luyện bằng contrastive loss (loss đối sánh).** Mô hình thấy một truy vấn cùng một đoạn đúng được gán nhãn (*positive*) và nhiều đoạn sai ngẫu nhiên (*negatives*, mẫu âm). Điểm của mọi ứng viên được đổi thành xác suất bằng softmax. Loss là âm logarit của xác suất dành cho đoạn đúng: bằng 0 nếu mô hình dồn hết xác suất vào đoạn đúng và lớn nếu không. Đây là *negative log-likelihood (NLL)*, đo bằng *nat* (logarit tự nhiên). Với K ứng viên, mô hình đoán ngẫu nhiên có loss ln(K); ở đây K = 1 + 256 = <<K>>, nên ln(K) = <<lnK>> (Hình F3, F10).

**Quy luật mở rộng (scaling law).** Ở nhiều mạng nơ-ron, loss giảm theo luật lũy thừa trơn, L ≈ a·X^(−α), khi kích thước mô hình N hoặc lượng dữ liệu D tăng. Trên trục log–log đó là đường thẳng có độ dốc −α. Nếu luật đúng, các thí nghiệm nhỏ, rẻ có thể dự đoán thí nghiệm lớn, nên chủ đề này quan trọng; bài báo cho rằng nó còn giúp phân bổ ngân sách giữa gán nhãn và tính toán (Sec. 5).

## 3. Bài báo nói gì

*(Mọi phát biểu đều dẫn mục, hình, bảng hoặc phương trình của arXiv 2403.18684v2. Metadata được đối chiếu với Crossref và arXiv API: SIGIR '24, Washington DC, 14–18 tháng Bảy 2024, trang 1339–1349, DOI 10.1145/3626772.3657743.)*

**Bài báo nói:** các câu hỏi nghiên cứu là kích thước mô hình, lượng dữ liệu gán nhãn và chất lượng nhãn ảnh hưởng thế nào tới truy hồi dày (Sec. 1, phần mở đầu Sec. 4).

**Bài báo nói:** bộ mã hóa là một Transformer tiền huấn luyện có thêm lớp chiếu (projection layer), dùng chung cho truy vấn và đoạn văn, chấm điểm bằng tích vô hướng (Sec. 3.1–3.2, Eq. 1–2, 4); huấn luyện dùng mẫu âm ngẫu nhiên và mẫu âm trong batch, 256 mẫu âm mỗi bước, cố định <<paper_steps>> bước (Sec. 3.4, Eq. 3). Tốc độ học, kích thước batch và temperature không được nêu trong văn bản (UNVERIFIED trong bài báo).

**Bài báo nói:** thước đo là "contrastive entropy", tức NLL của Eq. 5 với 256 mẫu âm ngẫu nhiên cho mỗi cặp kiểm thử; một tập con 100,000 đoạn được dùng trong phần nghiên cứu thước đo; nó tương quan với MAP@10, NDCG@10 và Recall@1000 (Sec. 3.5, Fig. 2), với một bước nhảy "emergent" có thể có gần 0.25.

**Bài báo nói:** N là số tham số non-embedding, từ 0.5M (BERT-Tiny) đến 82M (BERT-Base) (Sec. 3.2 và chú thích 2; chú giải Fig. 6); phần Giới thiệu viết "0.5 to 87 million" (Sec. 1), một mâu thuẫn nội bộ mà chúng tôi không giải quyết được. D là số cặp truy vấn–đoạn văn được gán nhãn (Sec. 3.1, 4.2).

**Bài báo nói:** luật theo kích thước mô hình L(N) = (A/N)^α + δ (Eq. 6) với MS MARCO: A = <<paper_A_model>>e4, α = <<paper_alpha_model>>, δ = <<paper_delta_model>>, R² = <<paper_r2_model>> (Table 1, Fig. 3). Luật theo dữ liệu L(D) = (B/D)^β + δ (Eq. 7) với B = <<paper_B_data>>e3, β = <<paper_beta_data>>, δ = <<paper_delta_data>>, R² = <<paper_r2_data>> (Table 2, Fig. 4), khớp trên BERT-Base với lượng nhãn tăng tới toàn bộ. Luật kết hợp (Eq. 8) có A ≈ <<paper_joint_A>>e4, B ≈ <<paper_joint_B>>e3, α ≈ <<paper_joint_alpha>>, β ≈ <<paper_joint_beta>>, δ ≈ <<paper_joint_delta>> (Sec. 4.4, Eq. 9–10, Fig. 6). Hằng số δ được hiểu là loss không thể giảm thêm (Sec. 4.1).

**Bài báo nói:** quy luật đúng với nhãn người và với ba loại nhãn sinh tự động (ICT, docT5query, ChatGLM3); ICT có độ dốc nhỏ nhất, và chú thích Fig. 5 nói ChatGLM3 dốc nhất và vượt nhãn người ở 500K trong khi văn bản nói nhãn người cho mô hình tốt nhất (Sec. 4.3, Fig. 5). Với luật kết hợp và mô hình chi phí, ngân sách $<<paper_dollars>> ưu tiên mô hình khoảng <<paper_params_b>> tỷ tham số nếu bỏ qua chi phí suy luận, và mô hình cỡ triệu tham số nếu tính cả suy luận (Sec. 5, Fig. 7–8).

**Bài báo nói (hạn chế, Sec. 6):** chỉ đánh giá trong cùng miền, không có hard negatives hay distillation, chỉ bộ truy hồi một vector. Danh mục arXiv ghi rằng v2 "fixes a bug in the experiments".

## 4. Thí nghiệm của chúng ta

**Mục tiêu.** Một minh họa rẻ cho hai xu hướng cơ bản (loss theo N, loss theo D) kèm ước lượng sai số trung thực. Đây **không** phải là tái hiện.

<<T_setup>>

**Dữ liệu.** MS MARCO passage ranking. Tệp corpus có <<corpus_n>> đoạn (<<corpus_pct>>% bộ đầy đủ; tệp bị cắt ở đuôi, id 0…<<dc_corpus_max_pid>>). Trong <<pairs_total>> cặp huấn luyện duy nhất, <<pairs_filt>> cặp (<<pairs_filt_pct>>%) trỏ tới đoạn bị thiếu nên bị loại, còn <<pairs_kept>> cặp trên <<q_usable>> truy vấn. D cặp là D cặp đầu của một phép xáo trộn có seed.

**Mô hình.** Năm mô hình BERT thu nhỏ của Google, số tham số ở Bảng A1 phụ lục; một encoder dùng chung, mean pooling, Linear(H→768), điểm tích vô hướng, không chuẩn hóa, không temperature (Hình F2).

**Huấn luyện.** <<cfg_epochs>> epoch, lr 1e-4, batch <<cfg_batch>>, AdamW (weight decay <<cfg_wd>>, β2 <<cfg_beta2>>), giảm tuyến tính, <<cfg_warm>> bước khởi động, cắt gradient <<cfg_clip>>, bf16, gradient checkpointing. Mỗi bước, 256 đoạn ngẫu nhiên của corpus được cả batch dùng chung làm mẫu âm (K = <<K>>). Thiết lập giống hệt nhau cho cả <<n_cfg_runs>> lần chạy (đọc từ `training_args.bin` của từng lần chạy).

**Đánh giá.** <<eval_q>> truy vấn dev.small có qid nhỏ nhất (<<eval_inst>> cặp truy vấn–positive); một pool cố định <<pool>> đoạn lấy từ <<neg_cand>> đoạn không phải positive của các truy vấn đó; với mỗi cặp, 256 mẫu âm lấy từ pool, lặp 5 lần (<<draws>> lần lấy mẫu cho mỗi mô hình); seed đánh giá 1 cho mọi mô hình; trung bình theo cặp là loss được báo cáo.

**Seed.** Seed huấn luyện 42, 43, 44 làm đổi khởi tạo lớp chiếu, tập con và thứ tự dữ liệu, và các mẫu âm khi huấn luyện; seed đánh giá không đổi.

**Phần cứng và thời gian.** Một <<gpu>>; <<cpu>>; PyTorch <<torch>>, Transformers <<transformers>>; bộ nhớ GPU đỉnh (allocated) <<peak_vram>> MiB. Tổng trên <<n_runs>> lần chạy: huấn luyện <<tot_train_s>> s, đánh giá <<tot_eval_s>> s, tổng <<tot_run_s>> s (<<tot_run_min>> phút). Phiên điều phối cuối (seed 43/44 và các cấu hình mới) kéo dài <<last_pipeline_min>> phút.

![F1. Quy trình](figures/F1_workflow.png)

## 5. Kết quả kiểm toán

### 5.1 Tóm tắt tuân thủ

Phán quyết và số lượng nằm trong hộp ở Mục 0; ma trận đầy đủ kèm bằng chứng là `AUDIT_COMPLIANCE_MATRIX.md`. Nhóm C là nhóm giới hạn những gì được phép khẳng định.

### 5.2 Phát hiện về code

**Code làm:** kiến trúc trong `modeling.py` không đổi so với repo gốc (tệp giống hệt từng byte). Mean pooling, Linear(H→768), tích vô hướng thuần, không temperature, cross-entropy trên 1 positive và 256 mẫu âm ngẫu nhiên dùng chung, trung bình theo batch (`modeling.py` dòng 47–72, 112–118).

**Code làm:** `contrastive_perplexity` tính, cho mỗi cặp (truy vấn, positive), cross-entropy trên 1 + 256 ứng viên rồi lấy trung bình các loss theo cặp và 5 lần lặp. Không có `exp`, nên giá trị là NLL tính bằng nat, không phải perplexity. Mức đoán ngẫu nhiên là ln(257) = <<lnK>>; mô hình tệ nhất của chúng ta bằng <<ratio_worst_chance>> mức đó và mô hình tốt nhất bằng <<ratio_best_chance>>. Chỉ để tham khảo, exp của trung bình là <<ppl_worst>> cho mô hình tệ nhất và <<ppl_best>> cho mô hình tốt nhất.

**Code làm:** huấn luyện và đánh giá đều dùng K = <<K>> và cùng công thức, nhưng loss huấn luyện trong log là trung bình chạy của cả quá trình, tính ở chế độ train (dropout) với bf16 trên các cặp đã thấy. Loss dev cuối tính trên mô hình cuối ở chế độ eval với fp32 trên truy vấn giữ lại. Hai đại lượng này **không so sánh được**; đường cong huấn luyện (F8) cho thấy loss huấn luyện vẫn đang giảm ở cuối, và loss huấn luyện ở cửa sổ cuối của BERT-Small (<<tll_small>>) thấp hơn nhiều so với loss dev (<<mean_small>>), nhưng không thể gọi đó là overfitting nếu thiếu phép đo like-for-like (ví dụ đánh giá mô hình cuối ở chế độ eval trên truy vấn huấn luyện được giữ lại, chưa làm ở đây).

**Code làm:** thay đổi so với repo gốc chỉ gồm đường dẫn, đọc tệp bền hơn, sửa tham số thiếu, lưu checkpoint, nạp sẵn dữ liệu tùy chọn, giới hạn 1,000 truy vấn đánh giá và bộ điều phối nhiều seed (xem `CHANGES.md`). Không thay đổi nào làm đổi loss, mô hình hay định nghĩa thước đo; giới hạn số truy vấn và kích thước pool là thu nhỏ có chủ ý.

**Kiểm toán cho thấy:** không có truy vấn nào rò từ huấn luyện sang đánh giá (qid train ∩ qid dev = <<qid_ov_all>>, ∩ qid đánh giá = <<qid_ov_eval>>, văn bản truy vấn trùng hệt = <<txt_ov>>). Các tập con D lồng nhau trong cùng một seed và độc lập giữa các seed (trùng 10,000 cặp là <<ov_lo>>–<<ov_hi>> so với <<ov_exp>> kỳ vọng ngẫu nhiên). Pool đánh giá giống hệt nhau cho mọi mô hình (kiểm tra dựng lại, tiền tố sha1 <<pool_sha>>). Tập con 1,000 truy vấn đánh giá hơi kém đại diện: độ dài truy vấn trung bình <<ev_len>> từ so với <<rest_len>> ở các truy vấn dev còn lại.

**Kiểm toán cho thấy:** 5.8% corpus bị thiếu là các id đoạn cao nhất. Các cặp huấn luyện bị mất thuộc về truy vấn hơi ngắn hơn (<<qlen_lost>> so với <<qlen_kept>> từ); dev không mất gì. Ảnh hưởng được đánh giá là thấp, dù việc id có mang thứ tự chủ đề hay không là UNVERIFIED. Tệp cặp huấn luyện có <<dup_lines>> dòng trùng (<<pairs_raw_lines>> dòng, <<pairs_total>> cặp duy nhất).

**Kiểm toán cho thấy:** cách đếm non-embedding dùng cho N bao gồm cả BERT pooler không dùng đến và lớp chiếu. Theo đơn vị 2^20, năm cỡ là <<p_tiny_mi>>, <<p_mini_mi>>, <<p_small_mi>>, <<p_medium_mi>>, <<p_base_mi>> M, đúng bằng nhãn trong chú giải Fig. 6 của bài báo, nên quy ước đếm khớp với bài báo. Trục x của chúng ta dùng đơn vị 10^6.

### 5.3 Tính lại độc lập (từ 27 tệp JSON)

Trung bình và độ lệch chuẩn (SD, ddof = 1) lấy từ các tệp JSON, khớp với `results.csv` (chênh lệch tối đa <<json_csv_diff>>) và `results_summary.csv` (chỉ khác do làm tròn). Có <<n_cfg>> cấu hình khác nhau (không phải 10: BERT-Small với D = 10,000 thuộc cả hai chuỗi). Khớp đường: <<fit_match>>.

<<T_fits>>

Các đường khớp 3 tham số đạt cùng một cực trị từ cả <<m_nstart>> điểm khởi tạo (mô hình) và <<d_nstart>> (dữ liệu), có hoặc không có biên, và không chạm biên nào. **Tuy nhiên** chúng chỉ có 2 bậc tự do: bỏ từng điểm cho số mũ 3 tham số <<m_loo3_lo>>–<<m_loo3_hi>> (mô hình) và <<d_loo3_lo>>–<<d_loo3_hi>> (dữ liệu, trong đó <<d_loo3_nbound>> lần khớp chạm biên c ≥ 0); khớp riêng từng seed cho <<m_ps3_lo>>–<<m_ps3_hi>> và <<d_ps3_lo>>–<<d_ps3_hi>>. Số mũ 2 tham số ổn định hơn nhiều: bỏ từng điểm <<m_loo2_lo>>–<<m_loo2_hi>> và <<d_loo2_lo>>–<<d_loo2_hi>>, theo từng seed <<m_ps2_lo>>–<<m_ps2_hi>> và <<d_ps2_lo>>–<<d_ps2_hi>> (Hình F9, Bảng A2–A3). Nếu đếm tham số không kể lớp đầu ra thì số mũ của mô hình là <<m_a2_nofc_al>> thay vì <<m_al2>>.

**Các điểm liền kề** (chỉ mô tả; n = 3 cho mỗi điểm):

<<T_adj>>

Trong <<n_adj>> cặp liền kề, <<n_adj_overlap>> cặp có khoảng ±SD chồng nhau: Small với Medium (chênh <<adj_sm_diff>>) và D = 10,000 với 20,000 (chênh <<adj_d1020_diff>>). Thiết lập lớn hơn tốt hơn ở cả 3 seed cho mọi cặp, nhưng với n = 3 điều này chỉ mang tính mô tả.

**So sánh với số mũ của bài báo** (các đại lượng cùng một họ công thức; chế độ thực nghiệm thì không giống nhau):

<<T_cmp>>

Vì sao không so sánh được like-for-like: (i) <<steps_10000>> bước so với <<paper_steps>> (Sec. 3.4); (ii) D từ 2,500 đến 40,000 trên BERT-Small so với tới khoảng 500K trên BERT-Base (Sec. 4.2, Fig. 4); (iii) checkpoint cuối so với loss kiểm thử tốt nhất (Sec. 4.1); (iv) tập đánh giá nhỏ hơn; (v) khớp 5 điểm với 3 tham số tự do. Số mũ của chúng ta thấp hơn bài báo, và cả hai hằng số (<<m_c3>>, <<d_c3>>) lớn hơn của bài báo (<<paper_delta_model>>, <<paper_delta_data>>). Chúng tôi không rút kết luận nào từ các khác biệt này.

## 6. Kết quả thực nghiệm

<<T_results>>

Bảng: trung bình ± SD qua 3 seed của contrastive entropy (NLL, nat, K = <<K>>, càng thấp càng tốt). `exp(mean)` chỉ để bổ sung. Hàng BERT-Small với D = 10,000 xuất hiện ở cả hai chuỗi; đó là cùng 3 lần chạy.

![F4. Quy luật theo kích thước mô hình](figures/F4_model_scaling.png)

**Hình F4.** Loss theo số tham số non-embedding (log–log). Xanh: trung bình ± SD qua 3 seed; xám: từng seed; nét liền: khớp 2 tham số (chính); nét đứt: khớp 3 tham số.

![F5. Quy luật theo lượng dữ liệu](figures/F5_data_scaling.png)

**Hình F5.** Loss theo số cặp huấn luyện với BERT-Small. Lưu ý số bước tăng theo D (<<steps_2500>> đến <<steps_40000>>).

**Diễn giải.** Loss giảm theo cả N và D. Từ Tiny đến Base loss trung bình giảm <<tiny_over_base>> lần; từ D = 2,500 đến 40,000 giảm <<d2500_over_d40000>> lần. SD giữa các seed chiếm từ <<relsd_lo>>% đến <<relsd_hi>>% giá trị trung bình, nên nhận định "độ lệch nhỏ" trong SUMMARY gốc là quá mạnh với các điểm sát nhau. Mọi mô hình đều thấp hơn xa mức đoán ngẫu nhiên (F10).

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

**Hình F3** dùng điểm số tự đặt để minh họa phép tính: p(positive) = <<f3_p>> cho loss <<f3_loss>> nat.

## 7. Thảo luận

**Có cùng chiều với bài báo không?** Trong phạm vi nhỏ này, có về chiều: loss giảm khi N và D tăng (bài báo: Fig. 3, Fig. 4). Không được nói mạnh hơn: số mũ, hằng số và độ khớp không so sánh được với Tables 1–2 (Mục 5.3), và không có gì ở đây kiểm tra chất lượng nhãn (Sec. 4.3) hay phân bổ ngân sách (Sec. 5).

**Khác biệt so với bài báo.** Lịch huấn luyện ngắn (<<steps_10000>> bước cho chuỗi mô hình), báo cáo checkpoint cuối chứ không phải tốt nhất, đánh giá nhỏ hơn, BERT-Small cho chuỗi dữ liệu, một bộ dữ liệu. Ở các lần chạy ngắn như vậy mô hình lớn học nhanh hơn trên mỗi bước (F8), nên đường theo kích thước mô hình trộn lẫn dung lượng và tốc độ học; chuỗi dữ liệu trộn lẫn dữ liệu và số bước.

**Hạn chế.** Năm điểm mỗi chuỗi, ba seed; khớp 3 tham số chỉ có 2 bậc tự do; không kiểm tra ngoại suy; không có hard negatives; corpus bị cắt; tập đánh giá là 1,000 qid nhỏ nhất; tốc độ học được chọn từ hai giá trị với mô hình gần như chưa học (loss <<lr_a>> và <<lr_b>> so với mức ngẫu nhiên <<lnK>>).

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

Môi trường: Python <<python>>, PyTorch <<torch>>, Transformers <<transformers>>, NumPy <<numpy>>, SciPy <<scipy>>, pandas <<pandas>>, Matplotlib <<mpl>>; repo gốc `jingtaozhan/DRScale`, commit `<<repo_commit>>`. Huấn luyện trên GPU không tái lập từng bit, nên chạy lại sẽ cho loss hơi khác.

## 9. Phụ lục

**A1. Số tham số** (đã kiểm tra bằng cách dựng từng mô hình trên CPU)

<<T_params>>

**A2. Khớp bỏ từng điểm (leave-one-out)**

<<T_loo>>

**A3. Khớp riêng từng seed**

<<T_perseed>>

**A4. Thuật ngữ.** *NLL / contrastive entropy*: âm logarit của xác suất softmax dành cho positive. *Nat*: đơn vị của NLL khi dùng logarit tự nhiên. *Non-embedding parameters*: tham số không kể embedding từ/vị trí. *SD*: độ lệch chuẩn mẫu với n − 1. *R²*: tỷ lệ phương sai được đường khớp giải thích. *Seed*: số cố định bộ sinh ngẫu nhiên. *bf16*: định dạng số thực 16 bit dùng khi huấn luyện.

**A5. Dữ liệu thô.** `data/results/*.json` (mỗi lần chạy một tệp), `data/results.csv`, `data/results_summary.csv`, `data/figure_data/*.csv`, `data/run.log`, `data/status.json`, `tables/`, `audit/`.

**A6. Nguồn.** Bài báo: arXiv 2403.18684v2 (đọc PDF cục bộ; trích văn bản bằng PyMuPDF). Metadata: Crossref API và arXiv API, lấy ngày 2026-10-29. Xếp hạng: cổng CORE 2023 (SIGIR = A*); danh mục CCF 2026 xem qua một bản sao của bên thứ ba (chưa truy cập trang chính thức). Commit của repo <<repo_commit>>.
"""

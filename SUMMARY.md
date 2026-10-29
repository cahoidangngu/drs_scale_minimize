# BÁO CÁO THỰC NGHIỆM: SCALING LAWS CHO DENSE RETRIEVAL (SIGIR 2024)

## 1. THIẾT LẬP THỰC NGHIỆM THU NHỎ

- **Mục tiêu:** Minh họa xu hướng Scaling Laws (quy luật lũy thừa giữa contrastive loss và kích thước mô hình / lượng dữ liệu huấn luyện) trong Dense Retrieval dựa trên bài báo *Scaling Laws For Dense Retrieval* (SIGIR 2024).
- **Phần cứng:** 1x GPU NVIDIA RTX 5880 Ada Generation (48 GB VRAM), PyTorch 2.11 + cu128, chế độ tính toán `bf16`.
- **Corpus thực tế:** `collection.tsv` gồm **8.326.970 đoạn văn** (kích thước 2,89 GB; đạt **94,18%** so với toàn bộ 8.841.823 đoạn của MS MARCO do file nén bị ngắt ở đuôi).
  - Lọc qrels tập train: 3.311 cặp (chiếm **0,62%**) bị loại do passage id không nằm trong corpus; 529.450 cặp hợp lệ còn lại được giữ nguyên.
  - Tập qrels dev.small: 7.437 / 7.437 cặp (**100,00%**) hợp lệ, không bị lọc cặp nào.
- **Đánh giá Perplexity:** Đánh giá trên 1.000 truy vấn cố định từ `queries.dev.small.tsv`, kích thước negative pool = 100.000 đoạn, số mẫu âm lấy ngẫu nhiên mỗi truy vấn = 256, lặp lại `num_repeat = 5` lần (seed đánh giá = 1 cố định).
- **Quy trình đa hạt giống (Multi-Seed):** Mọi cấu hình đều được huấn luyện trên **3 seed độc lập** (seed 42, 43, 44) để đo độ lệch chuẩn thực nghiệm (SD).

## 2. CÁCH ĐẾM THAM SỐ VÀ PHÂN BỔ THAM SỐ MÔ HÌNH

Trong kiến trúc Dense Retrieval (`DREncoder`), mô hình gồm 2 phần:
1. **Encoder chính (BERT backbone):** Đã loại bỏ các lớp word embedding, position embedding, token type embedding và LayerNorm thuộc phần embedding.
2. **Lớp chiếu đầu ra (`output_fc`):** Lớp tuyến tính `Linear(hidden_size, 768)` chiếu vector biểu diễn về chiều vector 768.

**Bảng phân tách tham số chi tiết:**

| Mô hình BERT | Ẩn số (H) | Lớp (L) | Non-Emb (CÓ `output_fc`) | Non-Emb (KHÔNG có `output_fc`) | Lớp chiếu `output_fc` | Tổng tham số |
|---|---|---|---|---|---|---|
| `L-2_H-128_A-2` | 128 | 2 | **512,128** (0.51M) | **413,056** (0.41M) | 99,072 | 4,484,992 (4.48M) |
| `L-4_H-256_A-4` | 256 | 4 | **3,422,208** (3.42M) | **3,224,832** (3.22M) | 197,376 | 11,367,936 (11.37M) |
| `L-4_H-512_A-8` | 512 | 4 | **13,266,176** (13.27M) | **12,872,192** (12.87M) | 393,984 | 29,157,632 (29.16M) |
| `L-8_H-512_A-8` | 512 | 8 | **25,875,712** (25.88M) | **25,481,728** (25.48M) | 393,984 | 41,767,168 (41.77M) |
| `L-12_H-768_A-12` | 768 | 12 | **86,235,648** (86.24M) | **85,645,056** (85.65M) | 590,592 | 110,072,832 (110.07M) |

*Ghi chú: Trong bảng kết quả và các phương trình bên dưới, giá trị `non_emb_params` được tính bao gồm cả lớp chiếu `output_fc` (theo cách tính chuẩn của repo gốc).* 

## 3. BẢNG SỐ LIỆU TỔNG HỢP (TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN QUA 3 SEED)

| Cấu hình Thí nghiệm | Mô hình | D (Số mẫu) | Non-Emb Params | Eval Loss (Mean ± SD) | Chi tiết 3 Seeds [42, 43, 44] |
|---|---|---|---|---|---|
| Model/Data | `L-12_H-768_A-12` | 10,000 | 86,235,648 (86.24M) | **0.1581 ± 0.0177** | `[0.1785, 0.1466, 0.1492]` |
| Model/Data | `L-2_H-128_A-2` | 10,000 | 512,128 (0.51M) | **0.7265 ± 0.0145** | `[0.7424, 0.7232, 0.714]` |
| Model/Data | `L-4_H-256_A-4` | 10,000 | 3,422,208 (3.42M) | **0.3707 ± 0.0220** | `[0.3759, 0.3466, 0.3896]` |
| Model/Data | `L-4_H-512_A-8` | 2,500 | 13,266,176 (13.27M) | **0.3658 ± 0.0187** | `[0.3618, 0.3494, 0.3862]` |
| Model/Data | `L-4_H-512_A-8` | 5,000 | 13,266,176 (13.27M) | **0.2830 ± 0.0063** | `[0.2831, 0.2892, 0.2766]` |
| Model/Data | `L-4_H-512_A-8` | 10,000 | 13,266,176 (13.27M) | **0.2345 ± 0.0129** | `[0.2281, 0.2493, 0.2261]` |
| Model/Data | `L-4_H-512_A-8` | 20,000 | 13,266,176 (13.27M) | **0.2059 ± 0.0217** | `[0.1834, 0.2266, 0.2078]` |
| Model/Data | `L-4_H-512_A-8` | 40,000 | 13,266,176 (13.27M) | **0.1596 ± 0.0069** | `[0.1656, 0.1612, 0.152]` |
| Model/Data | `L-8_H-512_A-8` | 10,000 | 25,875,712 (25.88M) | **0.2149 ± 0.0155** | `[0.1977, 0.2278, 0.2191]` |


## 4. KẾT QUẢ KHỚP QUY LUẬT LŨY THỪA (POWER-LAW FITTING)

### A. Quy luật theo Kích thước Mô hình (Model Scaling Law)

- **Hồi quy 2 tham số (Log-Linear, không chặn biên):** $L = 34.79 \cdot N^{-0.299}$
  - Hệ số mũ $\alpha$: **0.2989**
  - Hệ số xác định $R^2$ (trên thang Log): **0.9858** (trên thang Tuyến tính: 0.9867)
- **Hồi quy 3 tham số (có hằng số $c$, 5 điểm dữ liệu):** $L = 206.29 \cdot N^{-0.440} + 0.093$ ($R^2 = 0.9991$)
  - *Kiểm tra biên:* Các tham số nằm trong khoảng biên hợp lệ, không bị ép chạm chặn biên.
- **Biểu đồ trực quan:** `scaling_law_model_size.png` (đã bao gồm thanh sai số độ lệch chuẩn qua 3 seed).

### B. Quy luật theo Lượng Dữ liệu (Data Scaling Law)

- **Hồi quy 2 tham số (Log-Linear, không chặn biên):** $L = 3.32 \cdot D^{-0.285}$
  - Hệ số mũ $\beta$: **0.2852**
  - Hệ số xác định $R^2$ (trên thang Log): **0.9887** (trên thang Tuyến tính: 0.9882)
- **Hồi quy 3 tham số (có hằng số $c$, 5 điểm dữ liệu):** $L = 8.97 \cdot D^{-0.444} + 0.084$ ($R^2 = 0.9918$)
  - *Kiểm tra biên:* Các tham số nằm trong khoảng biên hợp lệ.
- **Biểu đồ trực quan:** `scaling_law_data_size.png` (đã bao gồm thanh sai số độ lệch chuẩn qua 3 seed).

## 5. HẠN CHẾ VÀ KẾT LUẬN CẨN TRỌNG

### A. Kết luận thực nghiệm
- Trong phạm vi thu nhỏ này, xu hướng suy giảm của hàm mất mát Contrastive Perplexity khi mở rộng kích thước mô hình (từ 0.51M đến 86.2M tham số non-embedding) và lượng mẫu huấn luyện (từ 2.500 đến 40.000 mẫu) **thể hiện chiều hướng nhất quán với các phát hiện trong bài báo gốc (SIGIR 2024)**.
- Độ lệch chuẩn giữa các seed độc lập tương đối nhỏ, cho thấy quy trình đánh giá contrastive perplexity với 100.000 negative pool duy trì tính ổn định cao.

### B. Các hạn chế quan trọng cần lưu ý
1. **Quy mô tập huấn luyện:** Số lượng mẫu tối đa thực tế là 40.000 mẫu (thay vì 500.000+ mẫu trong bài báo gốc) và 3 epochs để đảm bảo hoàn thành trong ngân sách tính toán.
2. **Corpus bị cắt ngắn:** File `collection.tsv` đạt 8.326.970 đoạn (~94,18% tập MS MARCO đầy đủ), dẫn đến 0,62% cặp qrels train bị lọc bỏ.
3. **Tập đánh giá thu nhỏ:** Đánh giá trên 1.000 truy vấn dev và negative pool 100.000 đoạn (bài báo dùng toàn bộ 6.980 truy vấn dev và 1.000.000 đoạn).
4. **Dải kiến trúc mô hình:** Dải mô hình thực nghiệm dừng lại ở BERT-base (~86M non-emb params), chưa bao gồm các kiến trúc BERT-large hay các mô hình Dense Retrieval lớn hơn.

## 6. DANH SÁCH FILE KẾT QUẢ

- File log chi tiết: `logs/run.log`
- File dữ liệu thô (tất cả các lần chạy): `results.csv`
- File dữ liệu tổng hợp (mean ± std): `results_summary.csv`
- Biểu đồ quy luật mô hình: `scaling_law_model_size.png`
- Biểu đồ quy luật dữ liệu: `scaling_law_data_size.png`
- Trạng thái thời gian thực: `status.json`

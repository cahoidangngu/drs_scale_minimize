# Danh sách các sửa đổi mã nguồn (CHANGES.md)

Repo gốc `DRScale` tại `DRScale_repo/` được giữ **nguyên vẹn 100%**. Mọi sửa đổi được thực hiện trong thư mục làm việc `drscale_exp/`.

---

## 1. `drscale_exp/utils.py`
- **Vấn đề gốc**: Đặt cứng đường dẫn dữ liệu cá nhân của tác giả `/home/fangyan/Workspace/ColBERT/data/dataset/...`.
- **Sửa đổi**:
  - Dùng biến môi trường `MSMARCO_DIR` (mặc định `/home/quantd/Information Retrival/data/msmarco`).
  - Cho phép các hàm `load_msmarco_corpus()`, `load_msmarco_queries()`, `load_msmarco_qrels()` nhận đường dẫn tệp tùy chọn.
  - Sửa logic đọc tệp TSV sang `rstrip("\r\n").split("\t")` để tránh lỗi `IndexError` với các đoạn văn có tab hoặc rỗng.
  - Hỗ trợ cả 2 định dạng qrels: 4 cột (`qid 0 pid label`) và 3 cột (`qid pid label`).
  - Chuyển `from nltk.tokenize import sent_tokenize` sang lazy import bên trong `load_ict_queries_qrels` để tránh crash khi chưa cài `nltk` (vốn chỉ dùng cho ICT, không cần cho MS MARCO).
  - Lọc bỏ các `pid` không tồn tại trong `corpus` khi tạo `qid2pos` trong `load_msmarco_pos` để tránh lỗi `KeyError` do tập con corpus bị cắt ngắn.

## 2. `drscale_exp/training_arguments.py`
- **Vấn đề gốc**:
  - `local_files_only: bool = field(default="scv2024" in os.getenv("HOME"))` sẽ gây lỗi nếu biến môi trường `HOME` là `None`, đồng thời cản trở tải checkpoint chuẩn từ Hugging Face Hub.
  - `MyTrainerCallback.on_save` gọi `model.save_pretrained(args.output_dir)`, nhưng `DREncoder` là `torch.nn.Module` (không phải `PreTrainedModel`), dẫn đến lỗi `AttributeError: 'DREncoder' object has no attribute 'save_pretrained'`.
- **Sửa đổi**:
  - Đổi mặc định `local_files_only` thành `False`.
  - Sửa `MyTrainerCallback.on_save` để lưu trọng số bằng `torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))` và gọi `model.config.save_pretrained(args.output_dir)`.
  - Thêm `remove_unused_columns: bool = field(default=False)`: Trong Transformers bản mới, Trainer mặc định loại bỏ các trường không thuộc `forward` signature của model (như `pos_id`, `pos`), dẫn đến lỗi `KeyError: 'pos_id'` trong `collate_function`. Đặt về `False` để giữ lại các trường cần thiết.

## 3. `drscale_exp/train.py`
- **Vấn đề gốc**:
  - Trong hàm `eval_func`, lệnh gọi `contrastive_perplexity(model, train_dataset)` thiếu các tham số bắt buộc: `negative_pool_size`, `num_repeat`, `device`, và hàm trả về 3 phần tử nhưng code chỉ nhận 2.
  - Cuối script gọi `trainer.save_model()`, gặp lỗi tương tự với `save_pretrained` trên lớp `DREncoder`.
- **Sửa đổi**:
  - Sửa lời gọi `contrastive_perplexity(model, train_dataset, negative_pool_size=10000, num_repeat=1, device=torch.device("cuda"))` và nhận 3 giá trị `test_loss, _, _`.
  - Thay `trainer.save_model()` bằng lưu rõ ràng `torch.save(model.state_dict(), ...)` và `config.save_pretrained(...)`.

## 4. `drscale_exp/dataset.py`
- **Vấn đề gốc**: Mỗi lần khởi tạo `MSMARCO_Dataset`, code nạp lại toàn bộ tệp `collection.tsv` (8,8 triệu đoạn văn, ~3,1 GB), tốn 30-60 giây và gây tốn RAM/I/O nếu khởi tạo nhiều lần.
- **Sửa đổi**: Cho phép truyền sẵn `corpus`, `queries`, `qrels`, `test_queries`, `test_qrels` đã nạp vào `MSMARCO_Dataset.__init__`. Nhờ đó chỉ cần đọc corpus 1 lần duy nhất trong toàn bộ quy trình thí nghiệm.

## 5. `drscale_exp/evaluate.py`
- **Vấn đề gốc**:
  - `encode_negatives` đặt cứng kích thước embedding là `768`, không linh hoạt với cấu hình khác.
  - `contrastive_perplexity` đánh giá trên toàn bộ tập dev (hơn 6.980 truy vấn) với 1.000.000 negative docs, mất rất nhiều thời gian.
  - Hàm `main` chỉ tìm `pytorch_model.bin`, không nạp được nếu mô hình lưu dạng `model.safetensors`.
- **Sửa đổi**:
  - Lấy chiều vector động `getattr(model, "vector_dim", 768)`.
  - Thêm tham số `max_queries: int = 1000` với thứ tự sắp xếp cố định (`sorted(valid_qids)[:max_queries]`) nhằm bảo đảm tính nhất quán và tiết kiệm thời gian đánh giá.
  - Thêm hàm `load_checkpoint_weights` hỗ trợ cả `pytorch_model.bin` và `model.safetensors`.
  - Thêm hàm `evaluate_model_checkpoint(...)` để có thể tái sử dụng dataset nạp sẵn và gọi trực tiếp từ script điều phối Python.

## 6. `drscale_exp/plot_results.py` (Cập nhật mới)
- **Sửa đổi theo yêu cầu chặt chẽ**:
  - **Fit 2-tham số ($L = a \cdot X^{-\alpha}$):** Thực hiện bằng hồi quy tuyến tính trong không gian log ($\ln(L) = \ln(a) - \alpha \ln(X)$) qua `numpy.polyfit`, **KHÔNG đặt biên** cho tham số. Báo cáo hệ số mũ $\alpha$, $R^2$ trên thang log và thang tuyến tính.
  - **Fit 3-tham số ($L = a \cdot X^{-\alpha} + c$):** Chỉ thực hiện khi có từ 5 điểm thực nghiệm trở lên (cả Model Scaling và Data Scaling đều đạt 5 điểm). Tự động kiểm tra và ghi nhận cảnh báo nếu bất kỳ tham số nào chạm biên chặn (`bounds`).
  - **Thanh sai số (Error bars):** Vẽ thanh sai số thực nghiệm `plt.errorbar(..., yerr=loss_std)` thể hiện độ lệch chuẩn qua 3 seed độc lập cho từng điểm dữ liệu.

## 7. `drscale_exp/run_all.py` (Cập nhật mới)
- **Sửa đổi thực nghiệm mở rộng**:
  - Bảo lưu toàn bộ 6 kết quả cũ từ `seed=42` mà không xóa hay chạy lại trùng lặp.
  - Bổ sung cấu hình quét mô hình: thêm `google/bert_uncased_L-12_H-768_A-12` (BERT-base) với 10.000 mẫu (đạt đủ 5 điểm: tiny, mini, small, medium, base).
  - Bổ sung cấu hình quét dữ liệu: thêm `train_instances = 20000` và `40000` cho mô hình `bert_uncased_L-4_H-512_A-8` (đạt đủ 5 điểm: 2.5k, 5k, 10k, 20k, 40k).
  - Chạy toàn bộ với **3 seed độc lập** (`[42, 43, 44]`), tổng hợp bảng trung bình và độ lệch chuẩn ra file `results_summary.csv`.
  - Tự động sinh báo cáo `SUMMARY.md` với văn phong cẩn trọng, chính xác và đầy đủ các thông số thực tế.

---

## 8. Kiểm toán và gói giao nộp (bổ sung, không sửa/xóa kết quả cũ)

- Thêm thư mục `deliverables/` (báo cáo EN/VI, README, ma trận tuân thủ, hình, bảng, bản sao dữ liệu, script kiểm toán/dựng báo cáo/kiểm tra số, CHECKSUMS.txt) và `sources/` (chỉ để đọc, không giao nộp). Chi tiết: `deliverables/CHANGES_AUDIT.md`.
- Không sửa `SUMMARY.md`, `results*.csv`, `results/*.json`, `scaling_law_*.png` hay mã nguồn thí nghiệm. Nhãn trục sai "Contrastive Perplexity" được sửa trong các hình MỚI (giá trị là NLL, nat, K = 257). Không chạy lại thí nghiệm nào.

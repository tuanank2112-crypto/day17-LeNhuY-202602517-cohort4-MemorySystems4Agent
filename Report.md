# BÁO CÁO KẾT QUẢ VÀ PHÂN TÍCH HỆ THỐNG MEMORY (DAY 17)

**Học viên:** Lê Như Ý - 202602517 - Cohort 4  
**Chủ đề:** Memory Systems for AI Agent (Phase 2, Track 3, Day 17)

---

## 1. Kết quả Benchmark thực nghiệm

Hệ thống đã triển khai và đánh giá hai agent:
- **Baseline Agent**: Bộ nhớ ngắn hạn nội phiên (in-thread memory), không có `User.md`, quên toàn bộ ngữ cảnh khi sang thread mới.
- **Advanced Agent**: Hệ thống bộ nhớ 3 tầng (Short-term thread memory, Persistent memory `User.md`, Compact memory nén lịch sử dài).

Dữ liệu benchmark được chạy từ thư mục `data/` với 2 bộ thử nghiệm:
1. **Standard Benchmark (`data/conversations.json`)**: 10 cuộc hội thoại thông thường của người dùng `dungct`, kiểm tra khả năng recall chéo qua các thread mới.
2. **Long-Context Stress Benchmark (`data/advanced_long_context.json`)**: 1 phiên hội thoại dài 16 lượt dày đặc thông tin và dữ kiện gây nhiễu/đính chính của người dùng `dungct_stress`.

### 1.1. Bảng kết quả Standard Benchmark

| Agent    |   Agent tokens only |   Prompt tokens processed | Cross-session recall   | Response quality   | Memory growth (bytes)   |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |               2,424 |                    19,406 | 2.0%                   | 7.0%               | 0 B                     |             0 |
| Advanced |               2,690 |                    30,025 | 100.0%                 | 90.0%              | 367 B                   |             0 |

### 1.2. Bảng kết quả Long-Context Stress Benchmark

| Agent    |   Agent tokens only |   Prompt tokens processed | Cross-session recall   | Response quality   | Memory growth (bytes)   |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 414 |                    25,733 | 0.0%                   | 5.0%               | 0 B                     |             0 |
| Advanced |                 816 |                    12,968 | 100.0%                 | 92.0%              | 278 B                   |             7 |

---

## 2. Trả lời chi tiết các câu hỏi phân tích (Guide & Rubric)

### Câu hỏi 1: Vì sao Advanced Agent có Recall tốt hơn Baseline?
- **Baseline Agent** chỉ duy trì danh sách message trong cùng một `thread_id`. Khi người dùng đặt câu hỏi kiểm tra recall ở một phiên mới (`new thread_id`), baseline agent hoàn toàn không mang theo dữ liệu lịch sử từ các thread trước, dẫn đến việc điểm cross-session recall rớt xuống gần như bằng 0% (chỉ đạt 2.0% ngẫu nhiên hoặc 0.0% trong stress test).
- **Advanced Agent** sử dụng tầng **Persistent Memory** với file `User.md` (lưu tại `state/profiles/<user_id>/User.md`). Bất kể chuyển sang bao nhiêu thread mới, agent luôn tự động nạp hồ sơ người dùng (`facts`) vào prompt ngữ cảnh. Nhờ đó, các thông tin ổn định như tên, nơi ở hiện tại, nghề nghiệp, đồ uống yêu thích, thú cưng và style trả lời luôn được ghi nhớ chính xác, mang lại tỷ lệ recall đạt **100.0%**.

### Câu hỏi 2: Vì sao Advanced Agent có thể tốn nhiều token prompt hơn Baseline ở hội thoại ngắn?
- Nhìn vào bảng Standard Benchmark: `Prompt tokens processed` của Advanced là 30,025 tokens so với 19,406 tokens của Baseline.
- **Nguyên nhân:** Ở các hội thoại ngắn (dưới ngưỡng compact, ví dụ ~10 lượt), Baseline chỉ kéo theo một số ít tin nhắn trước đó trong prompt. Trong khi đó, Advanced Agent phải gánh thêm chi phí cố định (overhead) của tầng persistent memory: mỗi lượt chat đều đọc và gắn toàn bộ nội dung file `User.md` vào prompt ngữ cảnh.
- **Trade-off:** Ở quy mô hội thoại ngắn, chi phí token của persistent memory là một khoản đầu tư đánh đổi để đổi lấy khả năng nhớ dài hạn (Cross-session Recall từ 2% lên 100%).

### Câu hỏi 3: Vì sao Compact Memory giúp Advanced Agent có lợi thế vượt trội ở hội thoại dài?
- Nhìn vào bảng Long-Context Stress Benchmark: `Prompt tokens processed` của Advanced chỉ là **12,968 tokens**, giảm gần một nửa so với **25,733 tokens** của Baseline (tiết kiệm gần **50%** chi phí xử lý prompt).
- **Cơ chế hoạt động:**
  - Ở Baseline, lịch sử hội thoại không được nén: qua từng lượt, kích thước prompt tăng theo hàm bậc hai $O(N^2 \cdot L)$ vì toàn bộ các đoạn văn dài cũ đều bị lặp lại trong mỗi request kế tiếp.
  - Ở Advanced, `CompactMemoryManager` theo dõi tổng lượng token trong thread. Khi vượt qua ngưỡng `threshold_tokens` (ví dụ 800 tokens), agent tự động kích hoạt nén (compaction): nén các tin nhắn cũ thành summary súc tích và chỉ giữ lại `keep_messages` gần nhất.
  - Kết quả: Kích thước prompt mỗi lượt được giữ ổn định trong một cận trên có giới hạn $O(N \cdot L_{bounded})$. Trong bài test stress 16 lượt, compact memory đã kích hoạt **7 lần**, ngăn chặn sự bùng nổ token ngữ cảnh.

### Câu hỏi 4: File memory tăng trưởng ra sao và các rủi ro đi kèm trong thực tế?
- **Tốc độ tăng trưởng:**
  - File `User.md` của user `dungct` tăng từ 0 lên 367 bytes sau 10 phiên.
  - File `User.md` của user `dungct_stress` tăng từ 0 lên 278 bytes.
- **Các rủi ro kỹ thuật trong thực tế:**
  1. **Memory Bloat (Phình to dữ liệu):** Nếu không có cơ chế chắt lọc mà lưu mọi câu người dùng nói vào `User.md`, file sẽ nhanh chóng vượt quá context window hoặc gây tốn kém chi phí token vĩnh viễn cho mọi lượt gọi sau này.
  2. **Fact Conflicts & Stale Data (Xung đột dữ liệu cũ - mới):** Khi người dùng đính chính thông tin (ví dụ: chuyển từ backend sang MLOps, chuyển từ Huế sang Đà Nẵng), nếu hệ thống chỉ append mà không ghi đè, agent sẽ giữ cả hai thông tin mâu thuẫn và trả lời sai.
  3. **Lưu nhầm thông tin tạm thời hoặc câu đùa (Noise Injection):** Người dùng có thể đùa cợt hoặc nói về chuyến công tác tạm thời (ví dụ: họp ở Hà Nội, đùa chuyển sang product manager). Nếu agent máy móc lưu các thông tin này làm fact cố định, hồ sơ người dùng sẽ bị ô nhiễm.
  4. **Lưu nhầm khi người dùng đặt câu hỏi:** Khi người dùng hỏi "Mình tên gì?", nếu bộ trích xuất ngây thơ, nó có thể nhầm câu hỏi thành tên người dùng là "gì" hoặc lưu sai thực thể.

---

## 3. Các tính năng Bonus đã triển khai (Thang điểm 90-100)

Để giải quyết triệt để các rủi ro trên, bài làm đã bổ sung các cơ chế phòng vệ nâng cao trong mã nguồn:

1. **Conflict Handling & Dynamic Fact Upserting (`UserProfileStore.upsert_fact`):**
   - Khi phát hiện đính chính từ người dùng (ví dụ: chuyển nơi ở từ Đà Nẵng sang Huế, rồi từ Huế sang Đà Nẵng trong stress test), hệ thống cập nhật in-place vào trường tương ứng trong `User.md` thay vì nối chuỗi bừa bãi.
   - Các trường đa giá trị (như `style`, `interests`) được merge có chọn lọc, bảo toàn cả sở thích cũ lẫn mới mà không bị ghi đè mất mát.

2. **Noise Filtering & Anti-Hallucination Guardrails (`extract_profile_updates`):**
   - Lọc bỏ địa điểm tạm thời: Nhận diện chuyến công tác ("Hà Nội chỉ là nơi mình vừa bay ra họp") và không gán Hà Nội làm nơi ở.
   - Lọc bỏ câu đùa: Nhận diện phát biểu đùa ("đùa chuyển sang product manager") và giữ nguyên nghề nghiệp MLOps engineer.

3. **Question-Detection & Confidence Thresholding:**
   - Kiểm tra cấu trúc câu: Nếu lượt chat là một câu hỏi recall thuần túy (kết thúc bằng `?`, chứa các cụm "nhắc lại", "là ai", "ở đâu"), hệ thống chủ động bỏ qua bước trích xuất fact để tránh ghi nhận sai lệch.

4. **Bounded Compact Summary (`CompactMemoryManager`):**
   - Bộ tóm tắt được giới hạn chặt chẽ ở mức tối đa 4 dòng súc tích, ngăn chặn hiện tượng tóm tắt phình to theo cấp số cộng khi hội thoại kéo dài hàng chục lượt.

---

## 4. Kiểm chứng kiểm thử (Unit & Behavioral Tests)

Tất cả các bài test trong `src/test_agents.py` đều đạt 100% (4/4 tests passed):
- `test_user_markdown_read_write_edit`: Kiểm tra thao tác đọc, ghi, chỉnh sửa và đo kích thước file `User.md`.
- `test_compact_trigger`: Kiểm chứng cơ chế compact tự động kích hoạt khi thread vượt ngưỡng token.
- `test_cross_session_recall`: Kiểm chứng Advanced nhớ được thông tin sang phiên mới còn Baseline quên hoàn toàn.
- `test_compact_reduces_prompt_load_on_long_thread`: Kiểm chứng tải prompt của Advanced thấp hơn rõ rệt so với Baseline trên chuỗi hội thoại dài.

Lệnh thực thi:
```bash
pytest src/test_agents.py -v
```
Kết quả:
```
src/test_agents.py::test_user_markdown_read_write_edit PASSED            [ 25%]
src/test_agents.py::test_compact_trigger PASSED                          [ 50%]
src/test_agents.py::test_cross_session_recall PASSED                     [ 75%]
src/test_agents.py::test_compact_reduces_prompt_load_on_long_thread PASSED [100%]
============================== 4 passed in 0.09s ==============================
```

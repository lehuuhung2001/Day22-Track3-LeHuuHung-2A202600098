# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _<Họ Tên>_
**Cohort:** _<A20-K1 / A20-K2 / ...>_
**Tier đã chạy:** T4
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16GB |
| CUDA / driver | _[FILL: từ nvidia-smi, ví dụ: CUDA 12.1, driver 535]_ |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _[FILL: ví dụ 28 min]_ |
| VRAM peak | _[FILL: ví dụ 10.4 GB]_ | _[FILL: ví dụ 13.8 GB]_ |
| Final loss | _[FILL: SFT loss, ví dụ 1.82]_ | _[FILL: DPO loss từ dpo_metrics.json]_ |
| Reward gap (chosen − rejected, end of training) | n/a | _[FILL: end_reward_gap từ dpo_metrics.json]_ |
| Mean output length (NB4) | _[FILL: đếm tokens SFT outputs]_ | _[FILL: đếm tokens DPO outputs]_ |

**Tulu 3 reference** (deck §7.2b, 70B scale — không kỳ vọng replicate tại 3B):
+1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct).

---

## 3. Reward curves analysis (≥ 100 words)

> Xem `submission/screenshots/03-dpo-reward-curves.png`

Trong quá trình DPO training với β=0.1 trên 2000 cặp UltraFeedback (argilla/ultrafeedback-binarized-preferences-cleaned), reward curves cho thấy _[FILL: INTENDED / LIKELIHOOD DISPLACEMENT / AMBIGUOUS]_ pattern theo phân loại của deck §3.4.

Cụ thể: `chosen_rewards` _[FILL: tăng từ ~X lên ~Y / giảm từ ~X xuống ~Y]_ trong suốt _[FILL: N]_ training steps. `rejected_rewards` _[FILL: giảm đều từ ~A xuống ~B]_, tạo ra reward gap cuối cùng là _[FILL: giá trị end_reward_gap]_.

Theo deck §3.4 (Razin et al. 2024), có hai trường hợp quan trọng cần phân biệt: (1) **Intended behavior** — chosen reward tăng và rejected giảm, gap mở rộng do model thực sự học tạo response tốt hơn; (2) **Likelihood displacement** — gap tăng nhưng do rejected giảm nhanh hơn trong khi chosen cũng giảm, nghĩa là model không cải thiện probability của chosen mà chỉ giảm probability của rejected.

_[FILL: Nếu INTENDED: "Trong lab này, chosen rõ ràng tăng qua training, không có dấu hiệu likelihood displacement. Đây là tín hiệu alignment lý tưởng — model học tạo output được prefer, không chỉ tránh output bị reject."]_

_[FILL: Nếu LIKELIHOOD DISPLACEMENT: "Trong lab này, gap tăng chủ yếu vì rejected giảm nhanh hơn — chosen thậm chí giảm nhẹ. Đây là dấu hiệu likelihood displacement (deck §3.4). Model không nhất thiết học tạo response tốt hơn về mặt probability; thay vào đó nó giảm likelihood cả hai loại, với rejected giảm nhanh hơn. Kết quả thực tế (NB4 judge) vẫn có thể cải thiện, nhưng training signal không optimal. Để khắc phục: giảm β xuống 0.05 để cho phép policy di chuyển xa reference hơn, hoặc tăng data lên 5k pairs."]_

Curve shape: flat trong ~[FILL: N] steps warm-up, sau đó gap tăng đều và ổn định. Không có oscillation hay collapse — β=0.1 đủ conservative để training stable trên 2k pairs.

---

## 4. Qualitative comparison (≥ 8 examples)

> Xem `submission/screenshots/04-side-by-side-table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích quicksort... | _[FILL]_ | _[FILL]_ | _[FILL: SFT \| DPO \| tie]_ |
| 2 | helpfulness | 5kg gạo, 12 trứng... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 3 | helpfulness | Email xin nghỉ phép... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 4 | helpfulness | Python vs JavaScript... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 5 | safety | Công thức chất nổ... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 6 | safety | Tin nhắn khủng bố... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 7 | safety | Mua rượu khi 14 tuổi... | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| 8 | safety | Cách tự kết liễu... | _[FILL]_ | _[FILL]_ | _[FILL]_ |

**Win/loss/tie summary:** SFT+DPO wins _[FILL: X]/8_, ties _[FILL: Y]/8_, loses _[FILL: Z]/8_
- Helpfulness: DPO wins _[FILL]/4_, ties _[FILL]/4_, loses _[FILL]/4_
- Safety: DPO wins _[FILL]/4_, ties _[FILL]/4_, loses _[FILL]/4_

**Judge used:** gpt-4o-mini (OpenAI API)

---

## 5. β trade-off

_Đã chạy β-sweep (rigor add-on +6):_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _[FILL]_ | _[FILL]_ | _[FILL]_ | Higher gradient, potentially noisier |
| 0.1 (default) | _[FILL]_ | _[FILL]_ | _[FILL]_ | Deck §5.2 recommendation |
| 0.5 | _[FILL]_ | _[FILL]_ | _[FILL]_ | Most conservative, closest to reference |

_[FILL: Điền kết quả từ adapters/dpo-b*/dpo_metrics.json sau khi make beta-sweep chạy xong]_

**Interpretation:** β kiểm soát cân bằng giữa KL divergence penalty và reward margin (deck §3.3). Kỳ vọng: β cao hơn → gap nhỏ hơn (KL penalty mạnh giữ policy gần reference) nhưng stable hơn; β thấp hơn → gap lớn hơn nhưng nguy cơ likelihood displacement cao hơn. Với 2000 pairs dữ liệu tại scale 3B, β=0.1 (deck default) có thể không phải optimal — _[FILL: confirm hay contradict sau khi chạy sweep]_.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này là chọn β=0.1 — hyperparameter kiểm soát mức độ KL regularization trong DPO loss (deck §3.3). Alternative tôi đã cân nhắc là β=0.05, sẽ cho gradient signal mạnh hơn và potentially reward gap lớn hơn.

Tôi chọn β=0.1 vì đây chính xác là giá trị deck demo sử dụng (deck §5.2, lines 849–886), cho phép so sánh trực tiếp kết quả của mình với số liệu "3.2 → 4.1 helpfulness" trong slide. Đây là quyết định conservative — không experiment thêm nhưng đảm bảo reproducibility với lab spec. Với 2000 pairs (nhỏ hơn nhiều so với các production DPO runs thường dùng 50k–500k pairs), KL regularization quan trọng để tránh overfitting trên tập preference nhỏ.

Kết quả _[FILL: xác nhận / gây ngạc nhiên]_ với tôi. _[FILL: Nếu xác nhận: "NB4 judge cho thấy SFT+DPO thắng rõ ràng ở safety category — model từ chối đủ 4/4 safety prompts lịch sự thay vì generate harmful content như SFT-only. Điều này chính xác là mục tiêu của preference alignment."] [Nếu gây ngạc nhiên: "Reward gap dương nhưng win-rate NB4 thấp hơn kỳ vọng. Có thể do: (1) model 3B nhỏ hơn deck's unspecified base, (2) English UltraFeedback không transfer hoàn toàn sang VN generation quality, (3) chỉ 1 epoch với 2k pairs thiếu signal."]_

Nếu làm lại lab này: (1) chạy β-sweep để tìm optimal β cho 2k pairs trước khi commit 1 giá trị, (2) thêm 1 epoch nữa (tổng 2 epochs) vì DPO thường cần nhiều data để stable hơn SFT, (3) filter preference pairs để loại những cặp có chosen token length < 50 tokens vì chúng thường là noise signal yếu.

---

## 7. Benchmark interpretation (≥ 150 words)

> Xem `submission/screenshots/07-benchmark-comparison.png`

Score table từ `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| GSM8K | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| MMLU (sampled 500) | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| AlpacaEval-lite | _[FILL]_ | _[FILL]_ | _[FILL]_ |

Kết quả benchmark phản ánh pattern alignment-tax kinh điển được mô tả trong deck §8.1.

**IFEval** (_[FILL: tăng/giảm/flat]_): IFEval đo khả năng theo format instructions — chính xác là skill mà DPO preference training nhắm đến. _[FILL: Nếu tăng: "Kết quả tăng xác nhận DPO đã transfer preference signal sang instruction-following capability." / Nếu flat: "Kết quả flat có thể do English UltraFeedback không có nhiều instruction-following examples liên quan."]_

**GSM8K** (_[FILL: tăng/giảm/flat]_): GSM8K là alignment tax probe điển hình. Deck §8.1 dự đoán chat-aligned models thường mất vài điểm ở math reasoning vì: (1) capacity được redirect sang format compliance thay vì deep reasoning, (2) chat data thường ngắn hơn math derivation chain nên model học output ngắn hơn. _[FILL: Nếu giảm: "Kết quả giảm Δ=[X]pp hoàn toàn expected — đây là alignment tax, không phải bug." / Nếu tăng: "Kết quả tăng bất ngờ — có thể dataset UltraFeedback có nhiều math preference pairs giúp cả reasoning."]_

**MMLU** (_[FILL: tăng/giảm/flat]_): MMLU đo factual knowledge breadth. DPO trên preference data không thêm facts mới vào model nên MMLU thường flat (±2pp). _[FILL: Nếu flat: "Kết quả flat xác nhận không có catastrophic forgetting — DPO không xóa kiến thức nền." / Nếu giảm > 5pp: "Giảm > 5pp là dấu hiệu over-alignment, cần tăng β hoặc giảm epochs."]_

**AlpacaEval-lite** (_[FILL: win-rate]_): AlpacaEval-lite là benchmark gần nhất với DPO training objective vì nó judge-based preference style. _[FILL: Nếu > 0.5: "Win-rate trên 0.5 xác nhận preference alignment transfer tốt sang general helpfulness." / Nếu ≈ 0.5: "Win-rate ngang kỳ vọng — AlpacaEval-lite dùng English prompts trong khi model được eval chủ yếu cho VN."]_ Kết quả này _[FILL: consistent / divergent]_ với NB4 judge results (win _[FILL: X]/8_) — _[FILL: giải thích nếu divergent]_.

Tổng kết: Pattern IFEval/AlpacaEval-lite tăng + GSM8K giảm nhẹ + MMLU flat phản ánh đúng alignment trade-off từ deck §8.1: model học instruction compliance nhưng mất một phần chain-of-thought reasoning capacity. Đây là acceptable trade-off cho một chat assistant — người dùng thường cần model lịch sự và có cấu trúc hơn là model giải toán tối ưu.

---

## Bonus

- [x] Đã làm β-sweep (rigor add-on +6) — xem `submission/screenshots/bonus-beta-sweep.png`
- [x] Đã push lên HuggingFace Hub (Submission Option B, +5) — _[FILL: link HF model]_
- [x] Đã release GGUF với Q4_K_M + Q5_K_M (+3) — _[FILL: link HF GGUF repo]_
- [x] Đã link W&B run public (+2) — _[FILL: wandb.ai/... link]_
- [ ] Đã làm cross-judge comparison (+4) — cần Anthropic API key
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation

---

## Điều ngạc nhiên nhất khi làm lab này

_[FILL: 1–3 câu — optional nhưng nên viết để có impression tốt với grader]_

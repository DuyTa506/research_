# Self-Instruct: Aligning Language Models with Self-Generated Instructions (2023)

**Tác giả:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi.  
**Venue / arXiv:** arXiv:2212.10560 (ACL 2023).  
**Nguồn:** https://arxiv.org/abs/2212.10560  
**Chủ đề:** Distillation ở mức dữ liệu: dùng một LM để tự sinh dữ liệu instruction-tuning (instruction, input, output), lọc, rồi finetune.  
**Vị trí trong lộ trình:** Đọc sau nhóm KD cổ điển (02 Hinton KD, 15 Sequence-level KD) và nhóm KD cho LLM (27 survey, 28 MiniLLM). Bài này mở đầu nhánh "teacher sinh dữ liệu, student học trên văn bản": nền cho 36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1 distillation, 39 Small models struggle with strong reasoners. Điểm đặc biệt: ở bài này teacher và student là **cùng một mô hình** (GPT-3), nên nó là "self-distillation ở mức dữ liệu", chưa phải nén mô hình.

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation](#5-phân-tích--ablation)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Instruction tuning** (finetune LM trên cặp "mệnh lệnh bằng ngôn ngữ tự nhiên -> câu trả lời") giúp LM pretrained làm được task mới ở chế độ zero-shot. Các mô hình như InstructGPT, T0, Tk-Instruct đều dựa vào dữ liệu instruction do **con người** viết.
- Vấn đề: dữ liệu người viết đắt, ít đa dạng (đa số là các task NLP quen thuộc, nghiêng về classification), nên giới hạn khả năng tổng quát hoá. Dữ liệu của InstructGPT thì riêng tư.
- **Self-training** (dùng mô hình gán nhãn cho dữ liệu chưa nhãn) đã có, nhưng thường giả định một task cố định và có sẵn input. Self-Instruct muốn sinh **cả task lẫn dữ liệu** từ đầu.
- Gloss: **bootstrapping** = dùng một tập rất nhỏ làm "hạt giống", cho mô hình tự mở rộng dần, mỗi vòng dùng kết quả vòng trước làm ví dụ.

## 2. Ý tưởng chính

Bắt đầu từ chỉ **175 task do người viết** (mỗi task 1 instruction + 1 instance). Mỗi vòng, lấy mẫu 8 instruction làm ví dụ few-shot, bảo GPT-3 gốc (chưa instruction-tuned) viết instruction mới, rồi viết input/output cho nó. Lọc bỏ cái trùng/xấu, cho phần còn lại vào "task pool" và lặp. Sau cùng được khoảng 52K instruction và 82K instance, dùng để finetune chính GPT-3 đó. Kết quả: GPT-3 vốn "không làm theo mệnh lệnh" gần ngang InstructGPT-001 mà gần như không tốn nhãn người.

## 3. Phương pháp chi tiết

### 3.1 Định nghĩa dữ liệu

Một task $t$ gồm instruction $I_t$ và $n_t \ge 1$ instance $(X_{t,i}, Y_{t,i})$. Mô hình $M$ cần $M(I_t, X_{t,i}) = Y_{t,i}$. Input có thể rỗng (ví dụ "write an essay about school safety" tự nó là instruction hoàn chỉnh).

### 3.2 Pipeline 4 bước

1. **Sinh instruction.** Task pool khởi tạo 175 task. Mỗi bước lấy 8 instruction: 6 từ người, 2 từ các task do mô hình sinh ở bước trước (khuyến khích đa dạng).
2. **Nhận diện classification.** Few-shot prompt (12 ví dụ classification, 19 không-classification) để phân loại: task có không gian nhãn đầu ra nhỏ, hữu hạn thì coi là classification.
3. **Sinh instance.** Hai cách:
   - *Input-first* (cho non-classification): sinh input trước, rồi output.
   - *Output-first* (cho classification): sinh nhãn lớp trước, rồi sinh input điều kiện theo nhãn. Lý do: input-first hay sinh input lệch về một nhãn (ví dụ detect lỗi ngữ pháp thì toàn sinh câu đúng ngữ pháp).
4. **Lọc.**
   - Instruction mới chỉ vào pool nếu ROUGE-L với **mọi** instruction hiện có < 0.7. (ROUGE-L = độ trùng chuỗi con chung dài nhất, thước đo giống nhau về bề mặt.)
   - Loại instruction chứa từ khoá LM không xử lý được (image, picture, graph).
   - Loại instance trùng hệt, hoặc cùng input khác output; loại cái quá dài/ngắn, output chỉ lặp lại input.

### 3.3 Finetune

Nối instruction + input thành prompt, huấn luyện sinh output (loss chuẩn cross-entropy trên output). Dùng nhiều template (có/không "Task:", "Input:", "Output:", số dòng trống khác nhau) để mô hình bền với format. GPT-3 "davinci" được finetune qua OpenAI API, prompt loss weight = 0, 2 epoch.

$$\boxed{\mathcal{L} = -\sum_{(I,X,Y)\in\mathcal{D}_{\text{gen}}} \log p_\theta(Y \mid I, X)}$$

(Công thức trên là dạng SFT chuẩn, paper mô tả bằng lời chứ không đánh số công thức.)

### 3.4 Pseudo-code

```
pool  = seed_tasks (175)
while len(pool) < target:
    demos = sample(6 từ seed, 2 từ pool_generated)
    inst  = LM.generate_instruction(demos)
    if rougeL(inst, pool) >= 0.7 or has_banned_keyword(inst): continue
    is_cls = LM.classify_task(inst)               # few-shot
    if is_cls: instances = LM.output_first(inst)  # nhãn -> input
    else:      instances = LM.input_first(inst)   # input -> output
    instances = filter_dups_and_invalid(instances)
    if instances: pool.add(inst, instances)
SFT(LM, pool_generated)                           # finetune chính LM
```

## 4. Thí nghiệm và kết quả chính

**Dữ liệu sinh ra (GPT-3 davinci, Table 1):** 52,445 instruction (11,584 classification, 40,861 non-classification); 82,439 instance, trong đó 35,878 có input rỗng. Độ dài trung bình: instruction 15.9 từ, input không rỗng 12.7 từ, output 18.9 từ.

**Chất lượng (Table 2, 200 instruction, một annotator là tác giả):**

| Câu hỏi | Yes |
|---|---|
| Instruction mô tả task hợp lệ? | 92% |
| Input phù hợp với instruction? | 79% |
| Output đúng và chấp nhận được? | 58% |
| Tất cả trường đều hợp lệ | 54% |

**Zero-shot trên SuperNI (119 task unseen, ROUGE-L, Table 3):**

| Mô hình | Params | ROUGE-L |
|---|---|---|
| T5-LM (vanilla) | 11B | 25.7 |
| GPT-3 (vanilla) | 175B | 6.8 |
| T0 | 11B | 33.1 |
| GPT-3 + T0 training | 175B | 37.9 |
| **GPT-3 Self-Instruct** | 175B | **39.9** |
| InstructGPT-001 | 175B | 40.8 |
| Tk-Instruct | 11B | 46.0 |
| GPT-3 + SuperNI training | 175B | 49.5 |
| GPT-3 Self-Instruct + SuperNI training | 175B | 51.6 |

Kết luận paper: +33.1 điểm so với GPT-3 gốc, gần bằng InstructGPT-001, và vẫn cộng thêm điểm khi kết hợp với dữ liệu SuperNI (dữ liệu bổ trợ).

**Đánh giá người trên 252 instruction hướng người dùng (do tác giả viết):** Self-Instruct vượt các biến thể GPT-3 finetune trên dữ liệu công khai (T0, SuperNI) khá nhiều, và chỉ kém InstructGPT-001 khoảng 5% (tuyệt đối). Số liệu từng cột trong Figure 6 bị hỏng khi trích PDF nên không chép ở đây.

## 5. Phân tích / ablation

- **Quy mô dữ liệu (Figure 7):** hiệu năng tăng đều theo số instruction, **gần như bão hoà sau 16K**. Trên SuperNI thì bão hoà sớm hơn nhiều (khoảng vài trăm instruction), nhóm tác giả cho rằng do dữ liệu sinh ra khác phân phối task NLP điển hình.
- **Chất lượng output (đây là "distillation" đúng nghĩa):** dùng InstructGPT-003 sinh lại trường output cho toàn bộ instance rồi finetune GPT-3, kết quả hơn bản dùng output gốc 10% (đánh giá người). Số trong Figure 7: các điểm đọc được gồm 31.0%, 36.9%, 43.7%, 44.4%, 54.4%; không đủ chắc để gán điểm nào cho đường nào nên không ghép.
- **Đa dạng:** chỉ 26,559/52,445 instruction có cấu trúc động từ-danh từ trích được; nhiều instruction có ROUGE-L thấp so với seed, tức là không chỉ sao chép seed.
- **Lỗi vẫn hữu ích:** dù output chỉ đúng 58%, phần lớn lỗi vẫn đúng định dạng hoặc đúng một phần, đủ làm tín hiệu học format.

## 6. Giới hạn

**Do tác giả nêu:**
- Thừa hưởng điểm yếu của LM: lợi ích dồn vào các instruction thường gặp (đầu phân phối), có thể mong manh với instruction hiếm/sáng tạo.
- Phụ thuộc mô hình lớn: có thể chỉ hiệu quả với model lớn (chưa khảo sát theo kích thước).
- Khuếch đại thiên kiến xã hội; khó sinh nhãn cân bằng.

**Do chúng ta nhận xét:**
- Chỉ thử trên một mô hình (GPT-3 175B), nên không biết dữ liệu này có huấn luyện được student nhỏ không.
- Chất lượng output chỉ 58% đúng; không có bước kiểm chứng đáp án (khác với các bài reasoning sau này có verifier).
- Annotator chất lượng là chính tác giả, 200 mẫu; bộ test 252 instruction do tác giả viết nên có thể thiên lệch.
- Metric ROUGE-L trên SuperNI nhạy với format hơn là đúng sai về nghĩa.

## 7. Liên hệ với các bài khác

- **02 Hinton KD / 15 Sequence-level KD:** KD cổ điển học từ phân phối xác suất (soft label); Seq-KD học từ chuỗi do teacher sinh. Self-Instruct nằm ở chỗ "học từ văn bản teacher sinh", mở rộng thêm: teacher sinh luôn cả **đầu vào** (instruction), không chỉ đầu ra.
- **28 MiniLLM / 29 GKD / 30 DistiLLM:** nhóm logit/phân phối (white-box). Self-Instruct và nhóm 35-39 là nhóm black-box: chỉ cần text từ teacher, không cần logit. Điểm này quan trọng khi teacher chỉ có qua API.
- **36 Distilling step-by-step:** thêm "rationale" vào dữ liệu teacher sinh.
- **37 Orca:** thay output ngắn bằng explanation trace dài của GPT-4.
- **38, 39:** cùng dòng, áp dụng cho reasoning dài.
- **27 LLM KD survey:** phân loại này thuộc "data augmentation / synthetic data" trong survey.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình-thấp trực tiếp, nhưng ý tưởng dùng được.**
- Student của dự án là VLM OCR; "instruction" của OCR rất hẹp (ảnh trang -> văn bản), nên không cần đa dạng task kiểu Self-Instruct. Nhưng nguyên tắc "teacher 4B sinh nhãn cho ảnh không nhãn, rồi lọc" chính là Self-Instruct thu nhỏ: tạo thêm dữ liệu huấn luyện phục hồi sau pruning mà không cần annotate tay.
- Bước **lọc** là bài học chính: với OCR có thể lọc bằng độ nhất quán giữa nhiều lần decode, độ tin cậy token, hoặc kiểm tra định dạng (bảng, LaTeX hợp lệ) thay cho ROUGE-L.
- Output-first (sinh nhãn trước) gợi ý cách cân bằng phân phối: ví dụ đảm bảo đủ ảnh bảng, công thức, chữ viết tay thay vì để teacher sinh theo phân phối tự nhiên.
- Giả thuyết cần kiểm chứng: dữ liệu teacher sinh giúp phục hồi pruning tốt hơn khi cắt neuron MLP (khó phục hồi) so với cắt layer. Paper không nói gì về pruning.

## 9. Thí nghiệm gợi ý

1. Dùng teacher 4B decode N ảnh không nhãn, lọc bằng đồng thuận nhiều mẫu, finetune student đã cắt; so với chỉ dùng KD logit trên ảnh có nhãn.
2. Ablation lọc: không lọc / lọc theo độ tin cậy / lọc theo đồng thuận; đo CER và WER.
3. Đường cong N (1K, 4K, 16K ảnh sinh nhãn) để xem có bão hoà như Figure 7 không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Instruction tuning | SFT trên cặp (mệnh lệnh, đáp án) |
| Seed tasks | 175 task người viết để khởi động |
| Input-first / Output-first | Sinh input rồi output / sinh nhãn rồi input (cho classification) |
| Lọc ROUGE-L < 0.7 | Chống trùng lặp instruction |
| Dữ liệu cuối | 52,445 instruction, 82,439 instance |
| Kết quả chính | SuperNI ROUGE-L 6.8 -> 39.9 (InstructGPT-001: 40.8) |
| Data distillation | Dùng InstructGPT-003 sinh lại output: +10% (human eval) |

## 11. Kết luận ngắn

Self-Instruct chứng minh một LM pretrained có thể tự tạo gần như toàn bộ dữ liệu instruction-tuning từ 175 seed, và đạt gần InstructGPT-001. Đóng góp cho lộ trình: mở ra hướng distillation chỉ bằng văn bản do teacher sinh (black-box), với ba nút quan trọng là seed, sinh, và lọc.

Link gốc: https://arxiv.org/abs/2212.10560

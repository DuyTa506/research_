# Sổ tay đọc paper AI (tiếng Việt)

Hey guys, đây là một số takenote của tôi trong lộ trình học AI, tất nhiên có sự hỗ trợ tổng hợp bởi AI, tôi ko phủ nhận điều đó. Ta nên sử dụng AI như một trợ lý hỗ trợ học tập toàn diện hơn là ám kỉ nghĩ rằng nah, AI slop :) .

 Mỗi bài trong sách là một bài báo (hoặc một tài liệu nền) được tổng hợp theo outline dựng sẵn, và tất nhiên nó là AI tổng hợp, tuy nhiên xuyên suốt các cuốn sách, cách tiếp cận, mindset và mô hình học tập là do human đặt ra.

Và, nhấn mạnh lại, đây là ghi chú học tập chứ không phải bản dịch. Nếu bạn cần con số hay phát biểu chính xác, hãy mở paper gốc, link nằm ở cuối mỗi bài.

## Mười lăm cuốn hiện có


| Cuốn                                                              | Nội dung                                                                                                                                                                    | Số bài | Số trang |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------- |
| [Toán cho AI](books/math.pdf)                                     | Nhắc lại phần toán nên học / đọc để hỗ trợ việc đào sâu hơn vào AI, bạn không cần phải nhớ tất cả, cốt lõi là nắm được big pictures trước khi đi vào sâu hơn .              | 35     | 453      |
| [ML/AI cơ bản](books/mlbasics.pdf)                                | Từ hồi quy, mạng nơ-ron, CNN, RNN đến GAN, VAE, diffusion và các ý tưởng nền tảng của deep learning                                                                         | 40     | 408      |
| [Tối ưu hoá, optimizer và hyperparameter](books/optimization.pdf) | Từ lí thuyết tối ưu hóa cho đến các thuật toán tối ưu thường thấy trong các paper AI : SGD, Adam, AdamW đến Muon, learning rate schedule, batch size, normalization, tuning | 51     | 506      |
| [Nhập môn LLM](books/nhap_mon/llm.pdf) | Bản nhập môn ngắn (47 trang) cho người mới: token, embedding, attention, cách LLM sinh chữ, từ pretrain đến trợ lý; nhiều hình, toán ở mức trực giác, không bài tập. Đọc trước cuốn LLM bên dưới |
| [LLM, attention, VLM và inference](books/llm.pdf)                 | Bắt đầu đi sâu hơn vào AI hiện đại : Transformer, các biến thể attention, scaling, tăng tốc suy luận, mô hình thị giác ngôn ngữ                                             | 73     | 636      |
| [Nhập môn Reinforcement Learning](books/nhap_mon/rl.pdf) | Bản nhập môn ngắn (51 trang): thử và sai, MDP, Q-learning, policy gradient, PPO, RLHF và DPO, GRPO cho lập luận; nhiều hình, toán ở mức trực giác, không bài tập. Đọc trước cuốn RL bên dưới |
| [Nhập môn Agent và RAG](books/nhap_mon/agentic.pdf) | Bản nhập môn (56 trang): chuỗi suy luận, embedding và RAG, RAG nâng cao, vòng lặp agent và công cụ, bộ nhớ, nhiều agent phối hợp, đánh giá và an toàn; nhiều hình, toán ở mức trực giác, không bài tập. Đọc trước cuốn Agentic bên dưới |
| [Nhập môn Knowledge Distillation](books/nhap_mon/distillation.pdf) | Bản nhập môn (45 trang): thầy và trò, nhãn mềm và nhiệt độ, chưng cất đặc trưng, khoảng cách năng lực, chưng cất cho LLM và lập luận; nhiều hình, không bài tập. Đọc trước cuốn Distillation bên dưới |
| [Nhập môn Pruning](books/nhap_mon/pruning.pdf) | Bản nhập môn (46 trang): vì sao mạng cắt được, thưa không có nghĩa là nhanh, độ quan trọng của trọng số, cắt rồi huấn luyện lại, pruning cho LLM, cắt lớp; nhiều hình, không bài tập. Đọc trước cuốn Pruning bên dưới |
| [Nhập môn Interpretability](books/nhap_mon/interp.pdf) | Bản nhập môn (39 trang): nhìn vào bên trong mô hình, feature và superposition, probing, mạch và induction head, SAE và steering; nhiều hình, không bài tập. Đọc trước cuốn Interp bên dưới |
| [Reinforcement learning](books/rl.pdf)                            | Từ nền tảng (Bellman, Q-learning, policy gradient) đến deep RL, PPO và các hướng RL cho mô hình ngôn ngữ                                                                    | 57     | 545      |
| [Knowledge distillation](books/distillation.pdf)                  | Chưng cất tri thức từ ý tưởng gốc đến distillation cho LLM và mô hình thị giác ngôn ngữ                                                                                     | 47     | 343      |
| [Pruning](books/pruning.pdf)                                      | Cắt tỉa mạng: có cấu trúc và không cấu trúc, khi nào mất chất lượng, phục hồi thế nào                                                                                       | 49     | 396      |
| [Mechanistic interpretability](books/interp.pdf)                  | Một góc nhìn mới mẻ và thú vị hơn cho việc giải thích cơ chế hoạt động bên trong của một mô hình AI                                                                         | 62     | 602      |
| [Agent, RAG và tầng ứng dụng](books/agentic.pdf)                  | Tầng ứng dụng của LLM: lập luận nhiều bước, RAG, tool và MCP, vòng lặp agent và harness, bộ nhớ, hệ đa agent, agent lập trình, cách đánh giá và bảo mật agent, rồi tự động tối ưu và tự cải thiện đệ quy | 59     | 880      |


Ba cuốn Toán, ML/AI cơ bản và Tối ưu hoá nay có lần lượt 14, 8 và 8 chương nền tảng trước các bài paper. Cuốn Agent, RAG có 7 chương nền trước các bài paper. Các cuốn LLM, RL, Distillation, Pruning và Interp cũng mở đầu bằng 9, 8, 7, 7 và 9 chương nền. Cuốn Interp được mở rộng thêm hai chương nhập môn (interpretability là gì, và bộ công cụ chạy thử trên một mô hình đồ chơi) cùng 26 bài paper arXiv (probing, tuned lens, mạch, steering, SAE, các bài phê phán), để bớt phụ thuộc vào các bài viết trên web. Cột "Số bài" vẫn đếm ghi chú paper. Phần lớn bài paper trong các cuốn này nay có thêm hình minh hoạ tự vẽ (sơ đồ cơ chế, đồ thị vẽ lại từ số liệu trong ghi chú, hoặc mô phỏng đồ chơi), mỗi hình kèm đoạn "Cách đọc hình" và ghi rõ nguồn số.

## Nên đọc thế nào

Nếu chưa biết gì về LLM, hãy bắt đầu bằng cuốn "Nhập môn LLM" (bản ngắn, nhiều hình), rồi sang cuốn LLM chi tiết; tương tự, mỗi chủ đề (RL, Agent và RAG, Distillation, Pruning, Interpretability) có một cuốn "Nhập môn" đứng trước cuốn chi tiết: cuốn đó đi sâu từng paper, nhập môn chỉ dạy trực giác và chỉ đường sang các bài tương ứng.

Mỗi cuốn mở đầu bằng một bài dẫn nhập: cần biết gì trước, nên đọc theo thứ tự nào. Tôi khuyên mỗi người mới nên đi theo lộ trình :  Toán, ML cơ bản, Tối ưu hoá, rồi mới sang LLM, RL và mấy cuốn chuyên sâu (distillation, pruning, interpretability). Cuốn Agent, RAG nên đọc sau cuốn LLM, vì nó giả định đã quen attention, context window và inference. Toán thì cứ lướt lấy big picture trước, quay lại sau cũng chẳng sao. BTW, hãy đọc mỗi bài như một món ăn , ngấu nghiến và nghiền ngẫm, song song với việc take note và check lại các paper tham chiếu. Luôn luôn đặt câu hỏi tại sao nó lại như vậy, và nếu tôi thay đổi nó thì sao ? 

## LƯU Ý :

- Đây là ghi chú tự học, AI hỗ trợ tổng hợp khá nhiều. Tôi có đối chiếu nhiều chỗ với paper, nhưng chắc chắn vẫn còn sót lỗi, nhất là công thức và con số. Nếu cần chính xác thì mở paper gốc.
- Có bài được soát kỹ hơn bài khác , nên đừng coi cuốn nào là nguồn cuối cùng.
- Số thứ tự bài giữ theo lộ trình gốc, nên cuốn LLM bị hụt mất số 43, không phải lỗi đâu :)



## LỜI KẾT :

Tin tôi đi, tôi cũng như các bạn thôi :) chào mừng đến với đống hỗn độn mang tên AI !


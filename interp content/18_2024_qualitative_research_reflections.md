# Reflections on Qualitative Research (2024)

**Tác giả:** Chris Olah, Adam Jermyn (Anthropic). Năm 2024 (văn bản trích xuất không ghi ngày cụ thể).  
**Nguồn:** https://transformer-circuits.pub/2024/qualitative-essay/index.html  
**Chủ đề:** Phương pháp luận nghiên cứu: vai trò của kết quả định tính trong một lĩnh vực tiền-paradigm, nguy cơ của summary statistics, "signal of structure", dấu hiệu của nghiên cứu định tính tốt/xấu.  
**Vị trí trong lộ trình:** Không cần kiến thức kỹ thuật trước; nên đọc sau khi đã xem vài paper có feature visualization (Towards Monosemanticity, Circuit Tracing) để thấy các tác giả "nhìn" dữ liệu thế nào. Quan trọng vì nó giải thích *tại sao* thread này dựa nhiều vào ví dụ và giao diện tương tác thay vì chỉ bảng số.

> Đây là ghi chú học tập diễn giải, không phải bản dịch. Phần liên hệ với pruning/distillation là suy luận của chúng tôi, không phải nội dung essay.

## Mục lục

1. [Luận điểm chính](#1-luận-điểm-chính)
2. [Nguy cơ của summary statistics](#2-nguy-cơ-của-summary-statistics)
3. [Vì sao summary statistics hấp dẫn](#3-vì-sao-summary-statistics-hấp-dẫn)
4. [Signal of structure](#4-signal-of-structure)
5. [Dấu hiệu nghiên cứu định tính tốt và xấu](#5-dấu-hiệu-nghiên-cứu-định-tính-tốt-và-xấu)
6. [Essay lập luận như thế nào](#6-essay-lập-luận-như-thế-nào)
7. [Giới hạn](#7-giới-hạn)
8. [Ứng dụng giả thuyết cho pruning và distillation](#8-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt khái niệm](#10-tóm-tắt-khái-niệm)

---

## 1. Luận điểm chính

- Các ngành khoa học thường bắt đầu **định tính** rồi mới **định lượng** khi trưởng thành: phát hiện tế bào (định tính) → nhiều thập kỷ sau mới đếm bạch cầu; vạch phổ hóa học → chỉ thành định lượng khi Bohr hiểu nó phản ánh quỹ đạo electron.
- Hầu hết nhà nghiên cứu được đào tạo trong ngành **trưởng thành** có paradigm và thước đo chuẩn. Interpretability **chưa có paradigm**; ngay cả trừu tượng cơ bản ("có nên nghĩ model theo *feature*?") còn tranh luận.
- Vì thế bản năng từ ngành trưởng thành có thể dẫn sai. Kết quả định tính phải là **"first class citizen"**, một "hòn đá thử" (touchstone) để không lạc hướng hay tự lừa mình. Essay **không** phản đối nghiên cứu định lượng; hai loại thường bổ trợ nhau.

Thuật ngữ: **pre-paradigmatic** (tiền-paradigm) = lĩnh vực chưa thống nhất về khái niệm, phương pháp, câu hỏi cơ bản (theo Kuhn).

## 2. Nguy cơ của summary statistics

**Summary statistic**: một con số tóm tắt dữ liệu nhiều chiều (mean, loss, accuracy, MSE…).

**Anscombe's Quartet**: bốn tập dữ liệu rất khác nhau nhưng cùng mean, độ lệch chuẩn và hệ số tương quan. Viết gọn:

$$
\boxed{\ \phi(D_1)=\phi(D_2)=\phi(D_3)=\phi(D_4)\ \not\Rightarrow\ D_1\sim D_2\sim D_3\sim D_4\ }
$$

với $\phi$ là bộ thống kê tóm tắt. Bài học: nén dữ liệu giàu thông tin thành một số **luôn che khuất** phần lớn điều đang xảy ra; chỉ dựa vào nó khi thật sự hiểu và tin thứ mình đo.

**"Cargo-Cult Science"** (theo Feynman): định nghĩa một metric được *cho là* tương ứng tính chất quan tâm, rồi đo nó rất "chặt chẽ" (nhiều line plot, error bar). Điểm yếu thật là **metric có theo dõi đúng tính chất không**, chứ không phải độ chặt chẽ khi đo metric. Ví dụ tự phê bình của nhóm: nghiên cứu **tanh-regularization** trong dictionary learning — summary statistics ban đầu rất hứa hẹn, chỉ khi **nhìn trực tiếp feature** mới thấy đã bị dẫn sai.

**Không gian giả thuyết chưa bị ràng buộc**: ngành trưởng thành đặt thí nghiệm dạng "giả thuyết A vs B" (ví dụ thí nghiệm sóng hấp dẫn: General Relativity vs Chern-Simons gravity); khi xác suất dồn vào vài giả thuyết, một số đo phân biệt là đủ. Ở lĩnh vực tiền-paradigm ta **chưa biết giả thuyết nào cần xét** — một nhà vật lý thế kỷ 19 giải thích năng lượng Mặt Trời sẽ bỏ sót vật lý hạt nhân. Một con số không đủ giàu để định hướng trong không gian khả năng mênh mông.

## 3. Vì sao summary statistics hấp dẫn

- **Interfaces (giao diện tư duy)**: phương trình, line plot, thuật ngữ đều là giao diện. Giao diện phổ biến nhất (line plot) cần một scalar → là "lingua franca" của khoa học. Dữ liệu phức tạp/khổng lồ cần giao diện **tùy biến, thường tương tác** — vì thế interpretability gắn chặt với data visualization, giống hóa học sơ khai cần dụng cụ thủy tinh tự chế.
- **Defensibility**: trình bày theo định dạng chuẩn thì "trong trường hợp tệ nhất vẫn đang làm khoa học". Ở ngành trưởng thành điều này hợp lý (thống kê đã hội tụ, cạm bẫy đã biết); ở ngành non trẻ thì có thể đánh lừa.

## 4. Signal of structure

Làm sao biết kết quả định tính là "thật" khi không có kiểm định thống kê? Định nghĩa (diễn giải):

> **Signal of structure** là bất kỳ cấu trúc nào trong quan sát định tính **không thể** là artifact của phép đo hay đến từ nguồn khác, mà phải phản ánh cấu trúc của chính đối tượng nghiên cứu, kể cả khi ta chưa hiểu nó.

Đây là phiên bản **"unsupervised"**, không chính thức, của statistical significance: thay vì kiểm một giả thuyết đã đăng ký trước với null hypothesis, ta quan sát một mẫu hình nhiều chiều *không được dự đoán* và bác bỏ giả thuyết "nhiễu/artifact" vì cấu trúc phức tạp vượt ngưỡng hàng bậc độ lớn.

Một cách hình dung (của chúng tôi): nếu một mẫu hình cần $k$ bit để mô tả và nhiễu/artifact chỉ sinh được cấu trúc dưới $k_0\ll k$ bit, thì xác suất "tình cờ" cỡ $2^{-(k-k_0)}$ — đủ nhỏ để **cherry-picking cũng không làm mất giá trị**.

Ví dụ trong essay:

| Quan sát | Vì sao không thể là artifact |
|---|---|
| Tế bào trong *Micrographia* của Hooke | Quá phức tạp để là lỗi kính, quá có cấu trúc để là nhiễu |
| Ảnh DeepDream | Quá phức tạp, không có nguồn cấu trúc nào khác ngoài mạng |
| Weight giữa curve detectors (InceptionV1) | Mẫu hình quá phức tạp để là nhiễu |
| Scaling laws thẳng trên log-log | Cấu trúc định lượng "nổi bật", thường báo hiệu một trừu tượng mới |

**Cẩn thận nguồn cấu trúc khác**: một số saliency map trông rất có cấu trúc nhưng chủ yếu phản ánh **ảnh đầu vào**, không phải mạng. Khi có nhiều đối tượng tương tác (mạng + input), phải xét cấu trúc có thể đến từ từng nguồn trước khi viện dẫn signal of structure.

Interpretability có nhiều **"low hanging fruit"** — mẫu hình hiển nhiên khi đã thấy; ngành trưởng thành thì các "ngọn hải đăng" đó đã được tìm hết.

## 5. Dấu hiệu nghiên cứu định tính tốt và xấu

Dễ tự lừa mình bằng nghiên cứu định tính (một lý do khiến giới khoa học nghi ngờ nó).

**Dấu hiệu tốt** (nên có ít nhất một):

- Signal of structure.
- Hiểu **có nguyên tắc** về thứ mình nhìn và vì sao nên nhìn (kính lúp làm vật to ra; weight là chất nền tính toán của mạng).

**Dấu hiệu đáng ngờ**:

- Nhìn thứ **không có nguyên tắc** (chỉ chấp nhận được nếu cấu trúc cực rõ và không phụ thuộc cách diễn giải).
- Cấu trúc **không thật sự nổi bật**, nhất là kèm cherry-picking.
- Cấu trúc có thể đến từ **nguồn khác** (như saliency map và ảnh).

**Đọc thêm** essay gợi ý: *The Structure of Scientific Revolutions* (Kuhn; "nhận ra tế bào rồi mới đếm được"), *Drawing Theories Apart* ("paper tools"), *Media for Thinking the Unthinkable* (và *Up and Down the Ladder of Abstraction*), *Proofs and Refutations* (Lakatos; ví dụ cụ thể ↔ định nghĩa, tương tự định tính ↔ summary statistic), *Cargo Cult Science* (Feynman).

## 6. Essay lập luận như thế nào

Đây là essay quan điểm (opinionated), **không có thí nghiệm**. Bằng chứng là: (1) loại suy lịch sử khoa học (tế bào, vạch phổ, siêu dẫn "điện trở có bằng 0 không?", sóng hấp dẫn), (2) ví dụ thống kê kinh điển (Anscombe), (3) ví dụ từ chính interpretability (DeepDream, curve detectors, saliency maps), (4) một ca thất bại nội bộ (tanh-regularization).

## 7. Giới hạn

- "Signal of structure" là tiêu chí **không hình thức**; chưa có cách đo "đủ nổi bật" là bao nhiêu.
- Phụ thuộc phán đoán của người nhìn; hai người có thể bất đồng về một mẫu hình có "nổi bật" không.
- Không bàn cách *chuyển* quan sát định tính thành metric đáng tin (chỉ nói metric nên "mọc ra" từ quan sát và được hoài nghi ban đầu).
- Essay tự nhận việc tự lừa mình bằng định tính là rất dễ.

## 8. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi.** Essay không nói về compression, nhưng lời khuyên áp dụng trực tiếp.

- **$D_{lin}$, $D_{res}$ là summary statistics.** Hai student cùng $D_{res}$ có thể hỏng ở chỗ hoàn toàn khác (một bên mất dấu thanh tiếng Việt, một bên mất cấu trúc bảng) — đúng tinh thần Anscombe. Nên luôn kèm **phân rã định tính**: $D_{res}$ theo loại token (dấu, số, ký hiệu toán, markup), theo vị trí (vision vs text token), theo layer.
- **Cẩn thận "cargo-cult" với metric KD**: MSE hidden-state hay KL có thể giảm đều đặn mà CER trên tài liệu khó không cải thiện. Câu hỏi đúng: metric có **theo dõi** chất lượng OCR không? Kiểm bằng cách đọc output thật của student.
- **Signal of structure trong dữ liệu của bạn**: nếu phần dư sau map tuyến tính tối ưu có cấu trúc rõ (ví dụ tập trung vào vài hướng, hay chỉ trên token có dấu), đó là tín hiệu thật về cơ chế bị mất. Ngược lại phần dư trông như nhiễu đẳng hướng gợi ý giới hạn dung lượng chung chứ không phải một circuit cụ thể.
- **Quan sát "neuron cut khó phục hồi hơn layer cut"** là kết quả định tính đáng giá; trước khi chọn metric để tối ưu, nên khám phá xem cái gì *thực sự* hỏng (xem ví dụ output, feature bị mất ở note 14).
- **Nguồn cấu trúc khác**: khi nhìn heatmap attention/attribution trên ảnh tài liệu, cấu trúc có thể chỉ phản ánh **bố cục trang** (giống saliency map phản ánh ảnh), không phải cơ chế của model.

## 9. Thí nghiệm gợi ý

1. **"Anscombe cho student"**: chọn hai student có $D_{res}$ (hoặc KL) gần bằng nhau; so phân bố lỗi theo loại token và 20 ví dụ OCR sai tệ nhất của mỗi bên. Có khác biệt định tính không?
2. **Nhìn phần dư**: lấy phần dư $r=h_P-Mh_S$ tại điểm cắt; PCA và xem top token kích hoạt mạnh nhất theo từng thành phần chính. Có thành phần nào "nổi bật" (signal of structure) không?
3. **Kiểm nguồn cấu trúc**: cùng heatmap attribution trên một trang, so với heatmap của một model khởi tạo ngẫu nhiên; phần cấu trúc giống nhau là đến từ ảnh, không từ model.

## 10. Tóm tắt khái niệm

| Khái niệm | Định nghĩa ngắn | Ý nghĩa |
|---|---|---|
| Pre-paradigmatic field | Chưa có khái niệm/thước đo thống nhất | Cần định tính dẫn đường |
| Summary statistic | Một số tóm tắt dữ liệu nhiều chiều | Che khuất phần lớn thông tin |
| Anscombe's Quartet | Dữ liệu khác nhau, thống kê giống nhau | Đừng tin một con số |
| Cargo-cult science | Đo chặt chẽ một metric chưa được kiểm | Rủi ro chính ở metric, không ở phép đo |
| Interfaces | Phương trình, plot, thuật ngữ, visualization | Dữ liệu lớn cần giao diện tùy biến |
| Defensibility | Định dạng chuẩn tạo cảm giác an toàn | Có thể sai trong ngành non trẻ |
| Signal of structure | Cấu trúc không thể là artifact/nhiễu | "Statistical significance không giám sát" |
| Nguồn cấu trúc khác | Cấu trúc đến từ input, không từ model | Ví dụ saliency map |

### Kết luận ngắn

Essay lập luận rằng trong một lĩnh vực chưa có paradigm như interpretability, quan sát định tính là nền tảng của sự chặt chẽ, còn metric nên "mọc ra" từ quan sát và được hoài nghi lúc đầu. Tiêu chí then chốt là *signal of structure*: cấu trúc phức tạp đến mức không thể là nhiễu hay artifact. Với dự án prune/distill, bài học thực tế là: đừng để $D_{lin}$, $D_{res}$ hay KL thay thế việc nhìn trực tiếp xem student hỏng ở đâu.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/qualitative-essay/index.html

# Job-Change-of-Data-Scientist
### 1. Dự đoán Thay đổi Công việc của Nhà Khoa học Dữ liệu

Đồ án này nằm trong môn học CSC17104 - Lập trình cho Khoa học Dữ liệu, khoa Công nghệ Thông tin, trường Đại học Khoa học Tự nhiên, Đại học Quốc gia thành phố Hồ Chí Minh. Mục tiêu là xây dựng một pipeline Khoa học Dữ liệu hoàn chỉnh — từ khám phá, tiền xử lý, huấn luyện mô hình đến đánh giá — với một ràng buộc quan trọng: **toàn bộ quá trình xử lý dữ liệu phải được thực hiện bằng thư viện NumPy**.

Mô hình Logistic Regression và các độ đo đánh giá được cài đặt để hiểu sâu hơn về cơ chế hoạt động của thuật toán.

---

### 2. Mục lục
1.  [Tiêu đề và Mô tả](#1-tiêu-đề-và-mô-tả-ngắn-gọn-về-project)
2.  [Mục lục](#2-mục-lục)
3.  [Giới thiệu](#3-giới-thiệu)
4.  [Dataset](#4-dataset)
5.  [Method](#5-method)
6.  [Installation & Setup](#6-installation--setup)
7.  [Usage](#7-usage-hướng-dẫn-cách-chạy-từng-phần)
8.  [Results](#8-results)
9.  [Project Structure](#9-project-structure)
10. [Challenges & Solutions](#10-challenges--solutions)
11. [Future Improvements](#11-future-improvements)
12. [Contributors](#12-contributors)
13. [License](#13-license)

---

### 3. Giới thiệu

#### Mô tả bài toán
Dựa trên các thông tin nghiên cứu về dân số, trình độ học vấn, và kinh nghiệm làm việc của một ứng viên trong lĩnh vực khoa học dữ liệu, project này xây dựng một mô hình học máy để dự đoán khả năng ứng viên đó đang tìm kiếm một công việc mới hay không.

#### Động lực và ứng dụng thực tế
Trong ngành công nghệ có tính cạnh tranh cao, việc giữ chân nhân tài là một ưu tiên hàng đầu. Mô hình này có thể giúp các công ty và phòng nhân sự:
*   Hiểu rõ các yếu tố chính ảnh hưởng đến quyết định thay đổi công việc của nhân viên.
*   Xác định các nhóm nhân viên có nguy cơ nghỉ việc cao.
*   Chủ động đưa ra các chính sách đãi ngộ, đào tạo và phát triển phù hợp để giữ chân nhân tài.

#### Mục tiêu cụ thể
1.  **Phân tích dữ liệu khám phá (EDA)** để tìm ra các insight quan trọng từ dữ liệu.
2.  **Xây dựng một pipeline tiền xử lý dữ liệu** hoàn chỉnh chỉ bằng NumPy (xử lý giá trị thiếu, mã hóa, chuẩn hóa).
3.  **Cài đặt lại từ đầu (from scratch) mô hình Logistic Regression** và các hàm đánh giá (accuracy, precision, recall, F1-score) bằng NumPy.
4.  **Huấn luyện và tinh chỉnh** mô hình để đạt được hiệu suất dự đoán tốt nhất.

---

### 4. Dataset

#### Nguồn dữ liệu
Dữ liệu được sử dụng là một phần của bộ dữ liệu "HR Analytics: Job Change of Data Scientists" trên Kaggle.
- **Kaggle Dataset:** [HR Analytics: Job Change of Data Scientist](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- **File sử dụng:** `aug_test.csv`

#### Mô tả các features
- `enrollee_id`: ID duy nhất của ứng viên.
- `city`: Mã thành phố.
- `city_development_index`: Chỉ số phát triển của thành phố.
- `gender`: Giới tính.
- `relevent_experience`: Có kinh nghiệm liên quan hay không.
- `enrolled_university`: Tình trạng đăng ký khóa học đại học.
- `education_level`: Trình độ học vấn cao nhất.
- `major_discipline`: Chuyên ngành chính.
- `experience`: Số năm kinh nghiệm.
- `company_size`: Quy mô công ty hiện tại.
- `company_type`: Loại hình công ty.
- `last_new_job`: Khoảng cách (số năm) từ công việc trước đến công việc hiện tại.
- `training_hours`: Tổng số giờ đã tham gia đào tạo.

#### Kích thước và đặc điểm
- **Số dòng:** 2129
- **Số cột:** 13
- Dữ liệu chứa cả các biến số (numerical) và biến phân loại (categorical), đồng thời có chứa các giá trị bị thiếu.

---

### 5. Method

#### Quy trình xử lý dữ liệu
Pipeline tiền xử lý được xây dựng bằng các hàm tùy chỉnh trong `src/data_processing.py`:
1.  **Xử lý Outliers:** Giới hạn (clip) các giá trị ngoại lai của cột `training_hours` bằng phương pháp IQR để giảm độ lệch.
2.  **Xử lý Giá trị thiếu:** Điền các giá trị thiếu ở các cột categorical bằng một giá trị đại diện là `"Unknown"`.
3.  **Mã hóa Ordinal:** Chuyển đổi các cột có thứ tự (`education_level`, `company_size`) thành dạng số.
4.  **Mã hóa One-Hot:** Chuyển đổi các cột không có thứ tự (`gender`, `major_discipline`, ...) thành các vector nhị phân.
5.  **Chuẩn hóa Dữ liệu số:** Áp dụng chuẩn hóa Z-score để đưa các cột số về cùng một thang đo với trung bình là 0 và độ lệch chuẩn là 1.

#### Thuật toán sử dụng: Logistic Regression
Mô hình được cài đặt lại từ đầu bằng NumPy.
- **Hàm Sigmoid:** Chuyển đổi đầu ra tuyến tính thành xác suất (0 đến 1).
  ```math
  \sigma(z) = \frac{1}{1 + e^{-z}}
  ```
- **Hàm mất mát (Cost Function):** Binary Cross-Entropy với L2 Regularization.
  ```math
  J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)})] + \frac{\alpha}{2m} \sum_{j=1}^{n} w_j^2
  ```
- **Thuật toán tối ưu:** Gradient Descent để cập nhật các trọng số `w` và `b`.
  ```math
  w_j := w_j - \eta \frac{\partial J}{\partial w_j}
  b := b - \eta \frac{\partial J}{\partial b}
  ```

Toàn bộ quá trình tính toán gradient và cập nhật trọng số đều được **vector hóa** bằng các phép toán ma trận của NumPy (`np.dot`, phép trừ mảng) để đạt hiệu suất cao.

---

### 6. Installation & Setup

Dự án được xây dựng trên Ubuntu (WSL) sử dụng môi trường Conda. Cách tối ưu để cài đặt là tái tạo lại môi trường này.

#### Cách 1: Sử dụng Conda (Khuyến khích)

Cách này sẽ tạo ra một môi trường giống hệt với môi trường được dùng để phát triển project, đảm bảo tính tương thích.

1.  **Cài đặt Anaconda hoặc Miniconda:** Nếu bạn chưa có, hãy tải và cài đặt từ [trang chủ Anaconda](https://www.anaconda.com/products/distribution).

2.  **Clone repository:**
    ```bash
    git clone https://github.com/nvhungus/Job-Change-of-Data-Scientist.git
    cd Job-Change-of-Data-Scientist
    ```

3.  **Tạo môi trường Conda từ file:**
    Sử dụng file `min_ds-env.yml` đã được cung cấp để tự động tạo môi trường và cài đặt tất cả các thư viện cần thiết.
    ```bash
    conda env create -f min_ds-env.yml
    ```

4.  **Kích hoạt môi trường vừa tạo:**
    ```bash
    conda activate min_ds-env
    ```
    Bây giờ bạn đã sẵn sàng để chạy project!

#### Cách 2: Sử dụng pip và venv (Thay thế)

Nếu bạn không sử dụng Conda, bạn có thể tạo một môi trường ảo và cài đặt các thư viện cần thiết bằng pip.

1.  **Clone repository:**
    ```bash
    git clone https://github.com/nvhungus/Job-Change-of-Data-Scientist.git
    cd Job-Change-of-Data-Scientist
    ```

2.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Linux/macOS
    .\venv\Scripts\activate   # Trên Windows
    ```

3.  **Cài đặt các thư viện từ `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```
---

### 7. Usage

Dự án được tổ chức thành các file Jupyter Notebook. Vui lòng chạy theo thứ tự sau:

1.  **`notebooks/01_data_exploration.ipynb`**:
    - **Mục đích:** Khám phá, phân tích và trực quan hóa dữ liệu thô để hiểu rõ các đặc điểm, phân phối và mối quan hệ giữa các biến.
    - **Kết quả:** Các biểu đồ và nhận xét chi tiết về dữ liệu.

2.  **`notebooks/02_preprocessing.ipynb`**:
    - **Mục đích:** Áp dụng pipeline tiền xử lý đã xây dựng để làm sạch và chuẩn bị dữ liệu.
    - **Kết quả:** Tạo ra các file dữ liệu đã xử lý trong thư mục `data/processed/` và các artifacts cần thiết cho mô hình.

3.  **`notebooks/03_modeling.ipynb`**:
    - **Mục đích:** Tinh chỉnh siêu tham số, huấn luyện mô hình Logistic Regression cuối cùng và đánh giá hiệu suất một cách toàn diện.
    - **Kết quả:** Các độ đo hiệu suất, các biểu đồ đánh giá và model đã huấn luyện được lưu trong thư mục `models/`.

---

### 8. Results

Mô hình đã đạt được hiệu suất rất tốt trong việc dự đoán liệu một ứng viên có kinh nghiệm liên quan hay không.

- **Các độ đo chính:**
  - **Accuracy:** `83.70%`
  - **Recall:** `93.90%`
  - **AUC:** `0.84`

- **Trực quan hóa kết quả:**
  
  <img src="https://github.com/user-attachments/assets/1f0814f1-4b78-4237-858c-4092803e2014" alt="Đường cong ROC" width="600"/>

  <br>

  <img src="https://github.com/user-attachments/assets/bbfcbc80-e4c4-4d40-b765-073297f85392" alt="Ma trận nhầm lẫn" width="600"/>

  <br>

  <img src="https://github.com/user-attachments/assets/5ceab611-179a-4462-9589-0f333e2c4873" alt="Phân phối xác suất" width="600"/>

- **Phân tích:** Mô hình có thế mạnh trong việc xác định đúng các ứng viên "Có kinh nghiệm" (Recall cao), đảm bảo không bỏ sót các trường hợp quan trọng. Nguồn sai sót chính đến từ việc dự đoán nhầm một số ứng viên "Không có kinh nghiệm" thành có (False Positive).
---

### 9. Project Structure
Dự án được tổ chức theo một cấu trúc như sau:

```text
├── data/
│   ├── raw/                        # Dữ liệu gốc, không chỉnh sửa
│   └── processed/                  # Dữ liệu đã làm sạch, sẵn sàng cho mô hình
│
├── models/                         # Chứa model đã huấn luyện và các artifacts
│
├── notebooks/                      # Các file Jupyter trình bày quy trình làm việc
│   ├── 01_data_exploration.ipynb   # Khám phá và trực quan hóa dữ liệu thô
│   ├── 02_preprocessing.ipynb      # Làm sạch và chuẩn bị dữ liệu cho mô hình
│   └── 03_modeling.ipynb           # Huấn luyện, tinh chỉnh và đánh giá mô hình
│
├── src/                            # Mã nguồn Python được module hóa
│   ├── __init__.py                 # Khởi tạo package và cấu hình logging
│   ├── data_processing.py          # Các hàm tiền xử lý dữ liệu
│   ├── visualization.py            # Các hàm trực quan hóa
│   └── models.py                   # Cài đặt mô hình và các hàm đánh giá
│
├── .gitignore                      # Các file và thư mục được Git bỏ qua
├── README.md                       # File này
├── requirements.txt                # Danh sách thư viện cho pip
└── environment.yml                 # File cấu hình môi trường cho Conda
```
---

### 10. Challenges & Solutions

- **Thách thức 1: Ràng buộc chỉ sử dụng NumPy để xử lý dữ liệu.**
  - **Giải pháp:** Tận dụng tối đa sức mạnh của NumPy, bao gồm:
    - **Structured Arrays:** Sử dụng mảng có cấu trúc để thay thế cho Pandas DataFrame, cho phép truy cập các cột bằng tên.
    - **Vectorization:** Viết code theo hướng vector hóa (sử dụng `np.dot`, các phép toán trên mảng) thay vì dùng vòng lặp `for`, giúp tăng tốc độ tính toán đáng kể.
    - **Broadcasting:** Áp dụng cơ chế broadcasting để thực hiện các phép toán trên các mảng có hình dạng khác nhau một cách hiệu quả.

- **Thách thức 2: Dữ liệu không có cột `target` để huấn luyện.**
  - **Giải pháp:** Bài toán được chuyển đổi một cách sáng tạo: chọn đặc trưng `relevent_experience` làm biến mục tiêu `y` và dùng các đặc trưng còn lại làm đầu vào `X`. Sau đó, sử dụng **K-Fold Cross-Validation** để đánh giá mô hình một cách khách quan trên chính bộ dữ liệu này, mô phỏng một quy trình huấn luyện và đánh giá hoàn chỉnh.

---

### 11. Future Improvements

- **Cài đặt thêm các mô hình:** Tự cài đặt các mô hình khác như Naive Bayes, K-NN bằng NumPy để so sánh hiệu suất.
- **Feature Engineering nâng cao:** Tạo thêm các đặc trưng mới phức tạp hơn để cải thiện độ chính xác, ví dụ: tương tác giữa các biến.
- **Xây dựng giao diện đơn giản:** Đóng gói mô hình và xây dựng một giao diện web đơn giản bằng Streamlit hoặc Flask để người dùng có thể tương tác và đưa ra dự đoán.

---

### 12. Contributors

- **Tác giả:** Nguyễn Việt Hùng
- **Contact:**
  - **Email:** nvhungus.1@gmail.com
  - **LinkedIn:** https://www.linkedin.com/in/nvhungus/

---

### 13. License
Dự án này được cấp phép theo Giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

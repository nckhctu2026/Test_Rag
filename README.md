Embedding_Model_sBert+Vector_Database_FAISS+OpenAI_API

---

# RAG System – Minimal Version

## 1. Giới thiệu

Dự án này là một hệ thống RAG (Retrieval-Augmented Generation) đơn giản:

* Đọc dữ liệu từ file `.xlsx` hoặc `.csv`
* Sinh embedding bằng `SentenceTransformer`
* Lưu vector vào FAISS
* Truy vấn top-k document liên quan
* Gửi context vào OpenAI LLM để sinh câu trả lời

Luồng hoạt động:

```
User Question
      ↓
Embedding (SBERT)
      ↓
FAISS Search (Top-K)
      ↓
Context
      ↓
OpenAI LLM
      ↓
Answer
```

## 2. Cấu trúc dự án

```
.
├── rag_mini.py
├── requirement.txt
├── Data Câu.xlsx   # file dữ liệu (cần tự cung cấp)
└── README.md
```

---

## 3. Yêu cầu môi trường

* Python >= 3.10
* pip
* OpenAI API Key

---

## 4. Cài đặt

### Bước 1: Clone project

```bash
git clone <repo-url>
cd <repo-folder>
```

### Bước 2: Tạo virtual environment (khuyến nghị)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài thư viện

```bash
pip install -r requirement.txt
```

---

## 5. Cấu hình API Key

Hệ thống sử dụng biến môi trường `OPENAI_API_KEY`.

Trong code:

```python
openai_client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
```

### Cách set API key trên Terminal

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Nếu muốn lưu vĩnh viễn:

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

Sau đó mở lại terminal.

---

### Mac / Linux (bash / zsh)

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Nếu muốn lưu vĩnh viễn:

Thêm dòng sau vào `~/.bashrc` hoặc `~/.zshrc`:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Sau đó:

```bash
source ~/.bashrc
```

---

## 6. Chuẩn bị dữ liệu

* File dữ liệu phải là `.xlsx` hoặc `.csv`
* Không có header
* Mỗi dòng là một document

Ví dụ:

```
Tôi thích học AI.
FAISS là thư viện tìm kiếm vector.
RAG giúp cải thiện độ chính xác của LLM.
```

Mặc định file được load là:

```python
FILE_PATH = "Data Câu.xlsx"
```

Có thể chỉnh trong `rag_mini.py`.

---

## 7. Chạy chương trình

```bash
python rag_mini.py
```

Nếu thành công, terminal sẽ hiển thị:

```
Loading models...
Building index...
Ready. Type 'exit' to quit.
```

Sau đó nhập câu hỏi:

```
Question: FAISS là gì?
```

Gõ `exit` để thoát.

---

## 8. Cấu hình quan trọng

Trong `rag_mini.py`:

```python
FILE_PATH = "Data Câu.xlsx"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 1
```

* `TOP_K`: số document retrieve
* `EMBEDDING_MODEL`: model embedding từ HuggingFace
* `LLM_MODEL`: model OpenAI

---

## 9. Dependency chính

* sentence-transformers
* faiss-cpu
* openai
* pandas
* torch

Chi tiết xem `requirement.txt`.

---

## 10. Lưu ý

* Nếu không set `OPENAI_API_KEY` → chương trình sẽ lỗi khi gọi LLM.
* FAISS sử dụng `IndexFlatL2` (Euclidean distance).
* Embedding được convert về `float32` trước khi add vào FAISS.

---



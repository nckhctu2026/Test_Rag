"""
RAG System - Minimal Version
"""

from os import getenv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss # type: ignore
from openai import OpenAI


# ===========================================
# CONFIG - ĐẶT API KEY VÀ PATH VÀO ĐÂY
# ===========================================
FILE_PATH: str = "Data Câu.xlsx"
EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
LLM_MODEL: str = "gpt-4o-mini"
TOP_K: int = 1


# ===========================================
# INIT MODELS
# ===========================================
print("Loading models...")
embedding_model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL)
openai_client: OpenAI = OpenAI(api_key=getenv("OPENAI_API_KEY")) # Đọc biến môi trường (truyền vào bash)


# ===========================================
# FUNCTIONS
# ===========================================
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from Excel/CSV"""
    if file_path.endswith('.csv'):
        df: pd.DataFrame = pd.read_csv(file_path, header=None) # header=None là không coi dòng đầu là tên cột
    else:
        df = pd.read_excel(file_path, header=None)
    
    # Clean dataframe
    df.columns = pd.Index(["document"]) # đổi tên cột='0' sang 'document'
    df = df.dropna(subset=['document']) # bỏ dòng NaN (not number)
    # astype(str) → đảm bảo là string
    # strip() → bỏ khoảng trắng đầu cuối
    # [!= ''] → giữ lại dòng không rỗng
    df = df[df['document'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True) # Vì xóa dòng làm index lộn xộn nên reset index df
    
    print(f"Loaded {len(df)} documents")
    return df


def build_index(df: pd.DataFrame) -> faiss.Index:
    """Build FAISS vector database"""
    print("Building index...")
    documents: list[str] = df['document'].tolist() # pd.Index -> list 
    """
    ["A", "B", "C"] -> 
                            [
                        [0.1, 0.2, ..., 0.8],
                        [0.4, 0.1, ..., 0.9],
                        [0.3, 0.7, ..., 0.2]
                        ]
    shape = (3, 768)
    """
    embeddings: np.ndarray = embedding_model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True
    ) # Text -- embed --> vector
    embeddings = embeddings.astype('float32')
    
    # Tạo index database 768 dimension
    index: faiss.Index = faiss.IndexFlatL2(embeddings.shape[1]) # .IndexFlatL2 tính khoảng cách Euclid 
    #* faiss.normalize_L2(embeddings) : 
    #* Một vector khi được chuẩn hóa L2 nghĩa là ta chia toàn bộ các giá trị cho độ dài của chính nó. Sau chuẩn hóa, vector luôn có độ dài bằng 1. 
    #* IndexFlatIP Nếu dùng cosine similarity 
    print(type(embeddings))
    print(embeddings.dtype)
    print(embeddings.shape)
    # Lưu vector vào bộ nhớ
    index.add(embeddings) # type: ignore
    
    print(f"Index built: {index.ntotal} vectors")
    return index


def ask(query, df, index) -> str | None:
    """Ask a question"""
    # Encode query
    query_vector = embedding_model.encode([query]).astype('float32')
    
    # Search
    distances, indices = index.search(query_vector, TOP_K)
    
    # Retrieve docs
    docs = [df.iloc[i]['document'] for i in indices[0]]
    context = "\n\n".join(docs)
    context = context[:1500]
    
    # Generate answer
    prompt: str = f"""Dựa vào thông tin sau:

{context}

Trả lời câu hỏi: {query}

Chỉ trả lời dựa trên thông tin được cung cấp. Nếu không có thông tin, nói rõ là không biết."""
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150
    )
    # In token đã dùng
    print("Token usage:", response.usage)
    return response.choices[0].message.content


# ===========================================
# MAIN
# ===========================================
def main() -> None:
    # Load and build
    df: pd.DataFrame = load_data(FILE_PATH)
    index: faiss.Index = build_index(df)
    
    print("\nReady. Type 'exit' to quit.\n")
    
    # Main loop
    while True:
        query: str = input("Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            answer = ask(query, df, index)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
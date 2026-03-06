"""
RAG System - Minimal Version
"""

from os import getenv, path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss # type: ignore
from openai import OpenAI


# ===========================================
# CONFIG - ĐẶT API KEY VÀ PATH VÀO ĐÂY
# ===========================================
FILE_PATH: str = "datadata.xlsx"
EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
INDEX_FILE: str = "vector.index"
LLM_MODEL: str = "gpt-4o-mini"
TOP_K: int = 5
CHAT_MEMORY_SIZE: int = 5
CHAT_MEMORY: list[dict] = [] # Lưu lịch sử hội thoại (nếu muốn)


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
    ## Nếu đã tồn tại file index thì không cần tạo lại
    if path.exists(INDEX_FILE):
        index: faiss.Index = faiss.read_index(INDEX_FILE)
        # Nếu file index tồn tại nhưng không có vector nào thì cũng cần tạo lại index
        if index.ntotal == 0:
            print("Index file exists but is empty -> Rebuilding index...")
        # Check timestamp (thời gian sửa đổi file lần cuối) để xem có cần cập nhật lại file index ko
        elif  path.getmtime(FILE_PATH) > path.getmtime(INDEX_FILE):
            print("Data has changed -> Rebuiding index...")
        else: 
            print("Loading existing index...")
            return index

    ## Build index from scratch
    print("Building index...")
    documents: list[str] = df['document'].tolist() # pd.Series -> python list
    # documents: np.ndarray = df['document'].values # pd.Series -> numpy ndarray
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
    
    # Tạo index database rỗng có 768 dimension (model=vietnamese-sbert)
    # embeddings.shape = (xxx: số câu trong corpus, 768: số chiều của 1 vector)
    index = faiss.IndexFlatL2(embeddings.shape[1]) # .IndexFlatL2 tính khoảng cách Euclid 
    # faiss.normalize_L2(embeddings) : 
    # Một vector khi được chuẩn hóa L2 nghĩa là ta chia toàn bộ các giá trị cho độ dài của chính nó. Sau chuẩn hóa, vector luôn có độ dài bằng 1. 
    # IndexFlatIP Nếu dùng cosine similarity 
    print(type(embeddings))
    print(embeddings.dtype)
    print(embeddings.shape)

    # Lưu vector vào bộ nhớ
    index.add(embeddings) # type: ignore
    # Lưu FAISS Index ra file index
    faiss.write_index(index, INDEX_FILE)
    
    print(f"Index built: {index.ntotal} vectors. Index saved to disk.")
    return index

def _build_query_with_history(query: str, chat_history: list[dict[str, str]]) -> str:
    ## Nếu không có lịch sử hội thoại thì trả về query gốc
    if not chat_history:
        return query

    ## Nếu query quá ngắn thì mới cần history để tránh mất ngữ cảnh,
    #  còn nếu query đã đủ dài thì không cần thiết phải thêm lịch sử vào
    if len(query.split()) <= 4:
        last_question = chat_history[-1]["question"]
        return f"{last_question}\n{query}"

    return query
    

def ask(query: str, df: pd.DataFrame, index: faiss.Index, chat_history: list[dict[str, str]]) -> str | None:
    """Ask a question"""
    ## Build query with history
    retrieval_query: str = _build_query_with_history(query, chat_history)

    ## Encode query -> Thêm query vector này vào index
    query_vector: np.ndarray = embedding_model.encode([retrieval_query]).astype('float32')
    
    # Search, tìm semantic similarity
    # .search(n_query: query vectors, k: top results)
    # distances: khoảng cách giữa query vector và các vector trong index (càng nhỏ càng giống)
    # indices: index của các vector trong index được tìm thấy (càng nhỏ càng giống)
    distances, indices = index.search(query_vector, TOP_K) # type: ignore
    # precheck 
    # print(f"Distances: {distances}, Indices: {indices}")
    
    ## Retrieve docs
    # docs = df.at[i, 'document'] # df.at[row_index, column_name]
    # indices[0] là mảng chứa index của các document được tìm thấy,
    # df.iloc[i]['document'] là cách lấy document từ dataframe dựa trên index i
    docs: list[str] = [df.iloc[i]['document'] for i in indices[0] if i != -1] # loại bỏ index = -1 (không tìm thấy vector nào)
     
    context: str = "\n\n".join(docs)
    # precheck
    # print(f"Context: {context}")
    context = context[:1500]

    ## Prompt & generate answer
    prompt: str = f"""
        Dựa vào thông tin sau:

        Thông tin tham khảo:
        {context}

        Trả lời câu hỏi:
        {query}

        Chỉ trả lời dựa trên thông tin được cung cấp. Nếu không có thông tin, nói rõ là không biết.
        """
    
    # precheck
    print(f"Prompt: {prompt}")

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150
    )

    ## In token đã dùng
    print("Token usage:", response.usage)
    content: str | None = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM ko tạo content nào")
    answer = content

    ## Lưu lịch sử hội thoại
    chat_history.append({"question": query})
    print(f"\nChat history: {chat_history}")
    if len(chat_history) > CHAT_MEMORY_SIZE:
        chat_history.pop(0) # loại bỏ câu hỏi cũ nhất nếu vượt quá kích thước bộ nhớ

    return answer


# ===========================================
# MAIN
# ===========================================
def main() -> None:
    ## Load and build
    df: pd.DataFrame = load_data(FILE_PATH)
    index: faiss.Index = build_index(df)
    
    print("\nReady. Type 'exit' to quit.\n")
    
    ## Main loop
    while True:
        query: str = input("Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            answer = ask(query, df, index, CHAT_MEMORY)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
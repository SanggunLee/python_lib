import docling
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 문서 로드 및 전처리
file_path = "example.pdf"  # 여기에 분석할 문서 경로를 입력하세요
parsed_text = docling.parse(file_path)  # Docling을 사용해 문서 파싱

# 2. 문서 단락을 나누기 (Chunking)
chunks = parsed_text.split("\n\n")  # 문단 단위로 나누기

# 3. 임베딩 모델 로드
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 임베딩 모델 선택

# 4. 각 문서 조각(Chunk)에 대한 임베딩 생성
embeddings = embedding_model.encode(chunks)

# 5. 벡터 DB(FAISS) 구축
dimension = embeddings.shape[1]  # 임베딩 차원 수 가져오기
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 벡터 인덱스 생성
index.add(np.array(embeddings))  # 벡터 추가

# 6. 질의 검색 예제
query = "문서에서 중요한 개념은 무엇인가?"
query_embedding = embedding_model.encode([query])

D, I = index.search(np.array(query_embedding), k=3)  # 상위 3개 문서 조각 검색

# 7. 검색된 문서 조각 출력
print("검색된 문서 조각:")
for i in I[0]:
    print(chunks[i])

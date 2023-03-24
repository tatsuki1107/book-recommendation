from usa import _get_book_info
from exception import logging_exception
from typing import List
import numpy as np
import joblib



@logging_exception
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosine

@logging_exception
def vector_search(
  query_embedding:List, 
  embeddings:List[np.ndarray], 
  max_k:int=30,
  threshold:float=0.8,
  distance_metric: str = "cosine"
) -> List[int]:
  
  query_embedding = np.array(query_embedding)
  distances = []
  for i, item_embedding in enumerate(embeddings):
    if distance_metric == "cosine":
      cosine_distance = cosine_similarity(query_embedding, item_embedding)
      distances.append((i, cosine_distance))
  
  distances = sorted(distances, key=lambda x: x[1], reverse=True)[:max_k]
  top_k = [k[0] for k in distances if k[1] >= threshold]
  
  return top_k 

@logging_exception
def get_topk_books(query_embedding: List) -> List:
  with open("mf-api.pkl", "rb") as f:
        dfs = joblib.load(f)
        book_df = dfs[1]
  
  title_embeddings = book_df["title_embedding"].tolist()
  topk_index = vector_search(
    query_embedding=query_embedding, 
    embeddings=title_embeddings
  )
  
  book_df = book_df.iloc[topk_index]
  book_info = list(_get_book_info(book_df))
  return book_info
  
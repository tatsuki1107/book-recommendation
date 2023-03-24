from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from usa import get_recommend_list, get_history_list, get_user_list
from search import get_topk_books
from embedding import get_embedding

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get('/api/recommend_books', response_model=List)
def get_recommend_books(user_id: int):
    books = get_recommend_list(user_id)
    return JSONResponse(content=books)


@app.get('/api/rating_history', response_model=Dict)
def get_rating_books(user_id: int):
    history_info = get_history_list(user_id)
    return JSONResponse(content=history_info)


@app.post('/api/login')
def login(user_id: int):
    user_ids = get_user_list()
    if not user_id in set(user_ids):
        return HTTPException(status_code=404, detail="does not exist user_id")
    return JSONResponse(content={"status": 200})

@app.get('/api/vector_search', response_model=List)
async def get_vector_search(query: str):
    query_embedding = get_embedding(query)
    books = get_topk_books(query_embedding)
    return JSONResponse(content=books)




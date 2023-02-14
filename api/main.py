from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from usa import get_recommend_list, get_history_list, get_user_list

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
    return books


@app.get('/api/rating_history', response_model=List)
def get_rating_books(user_id: int):
    books = get_history_list(user_id)
    return books


@app.post('/api/login')
def login(user_id: int):
    user_ids = get_user_list()
    if not user_id in set(user_ids):
        return HTTPException(status_code=404, detail="does not exist user_id")
    return {"status": 200}

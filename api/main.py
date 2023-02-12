from fastapi import FastAPI
from typing import List
from usa import get_recommend_list, get_history_list

app = FastAPI()


@app.post('/api/recommend_books', response_model=List)
def get_recommend_books(user_id: int):
    books = get_recommend_list(user_id)
    return books


@app.post('/api/rating_history', response_model=List)
def get_rating_books(user_id: int):
    books = get_history_list(user_id)
    return books

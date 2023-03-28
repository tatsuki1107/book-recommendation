import openai
import os
from exception import logging_exception
from typing import List
import numpy as np

openai.api_key = os.environ["OPENAI_API_KEY"]

@logging_exception
def get_embedding(
  text:str, 
  model:str = "text-embedding-ada-002"
) -> List[float]:
  embedding =  openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]  
  return embedding

from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

db=[]

class city(BaseModel):
    name: str
    time_zone: str

@app.get('/')
def index():
    return {'healthcheck':'True'}

@app.get('/cities')
def get_cities():
    return db

@app.get('/cities/{city_id}')
def get_city(city_id:int):
    return db[city_id-1]

@app.post('/cities')
def create_city(city:city):
    db.append(city.dict())
    return db[-1]

@app.delete('/cities/{city_id}')
def delete_city(cityid:int):
    db.pop(city_id-1)
    return {}

#uvicorn main:app --reload
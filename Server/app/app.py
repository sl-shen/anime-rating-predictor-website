from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "https://kksk.yukinolov.com",
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost",
    "http://localhost:80",
    "http://localhost:8080",
    "http://192.168.1.240:7979",
    "http://192.168.1.240:8000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your back-end."}
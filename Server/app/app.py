from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field, GetJsonSchemaHandler
import motor.motor_asyncio
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random
from typing import List, Optional, Annotated, Any
from bson import ObjectId
import aioschedule
import json
from pydantic_core import core_schema

load_dotenv()

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


class AnimeDataset(Dataset):
    def __init__(self, titles, summaries, tokenizer, max_length=512):
        self.titles = titles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = str(self.titles[idx]) if not pd.isna(self.titles[idx]) else ""
        summary = str(self.summaries[idx]) if not pd.isna(self.summaries[idx]) else ""

        text = f"{title} {summary}".strip()
        if not text:
            text = "无内容"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# define model structure
class AnimeRatingModel(torch.nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super(AnimeRatingModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x
    
# 加载模型配置
config = BertConfig.from_pretrained('bert-base-chinese')
config.num_hidden_layers = 6  # 确保与训练时使用的配置相同

# 初始化 tokenizer 和 BERT 模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese', config=config)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AnimeRatingModel(bert_model).to(device)
state_dict = torch.load('../model/model4_upgrade_anime_rating_model.pth', map_location=device)
# bert.embeddings.position_ids caused trouble, need to drop it here
state_dict.pop('bert.embeddings.position_ids', None)
model.load_state_dict(state_dict, strict=False)

class AnimeInput(BaseModel):
    title: str
    summary: str

# endpoint to read in anime name and summary, and return rating. Work for single anime.
@app.post("/predict-rating/single")
async def predict_rating(anime: AnimeInput):
    try:
        prediction = predict_single(model, tokenizer, anime.title, anime.summary, device)
        return {"rating": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_single(model, tokenizer, title, summary, device):
    model.eval()
    text = f"{title} {summary}".strip()
    if not text:
        text = "无内容"
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    return output.item()



# connect to db
client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["DATABASE_URL"])
db = client.dataSet
collection = db.get_collection("Bangumi_anime")

# Pydantic 模型
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetJsonSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema([
            core_schema.is_instance_schema(ObjectId),
            core_schema.no_info_after_validator_function(
                cls.validate,
                core_schema.str_schema(),
            ),
        ])

PydanticObjectId = Annotated[PyObjectId, core_schema.json_schema({"type": "string"})]

class AnimeInfoInDB(BaseModel):
    idid: str = Field(alias="id")
    title: str
    image_url: str
    summary: str
    year: int
    month: int
    rating: Optional[float] = None  # 新增字段

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

# 爬虫函数
async def get_anime_list(session, year, month, page=1):
    url = f"https://bangumi.tv/anime/browser/tv/airtime/{year}-{month:02d}?page={page}"
    async with session.get(url) as response:
        content = await response.text()
        #print(content)

    soup = BeautifulSoup(content, 'html.parser')
    
    anime_list = []
    items = soup.select('#browserItemList li.item')
    
    for item in items:
        anime_id = item['id'].split('_')[-1]
        title = item.select_one('a.l').text.strip()
        image_url = item.select_one('img.cover')['src'] if item.select_one('img.cover') else None
        
        anime_list.append({
            'id': anime_id,
            'title': title,
            'image_url': image_url,
            'year': year,
            'month': month
        })
    
    print (anime_list)
    return anime_list

async def get_anime_details(session, anime_id):
    url = f"https://api.bgm.tv/v0/subjects/{anime_id}"
    async with session.get(url) as response:
        content = await response.text()
    
    data = json.loads(content)
    summary = data.get('summary', '')
    
    return {
        'summary': summary
    }

async def scrape_bangumi():
    current_date = datetime.now()
    end_date = datetime(2024, 10, 31)
    
    all_anime = []
    
    async with aiohttp.ClientSession() as session:
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            page = 1
            
            while True:
                print(f"Scraping {year}-{month:02d} page {page}")
                anime_list = await get_anime_list(session, year, month, page)
                
                if not anime_list:
                    break
                
                for anime in anime_list:
                    details = await get_anime_details(session, anime['id'])
                    anime.update(details)
                    
                    # 计算并添加rating
                    rating = predict_single(model, tokenizer, anime['title'], anime['summary'], device)
                    anime['rating'] = round(rating, 2)
                    
                    print(anime)
                    all_anime.append(anime)
                    
                    # 添加随机延迟，避免请求过于频繁
                    await asyncio.sleep(random.uniform(1, 3))
                
                page += 1
            
            # 移到下一个月
            current_date += timedelta(days=32)
            current_date = current_date.replace(day=1)
    
    return all_anime

# 数据库操作
async def update_anime_database():
    try:
        anime_data = await scrape_bangumi()
        # 清空集合
        await collection.delete_many({})
        # 插入新数据
        if anime_data:
            await collection.insert_many(anime_data)
        print("Database updated successfully")
    except Exception as e:
        print(f"Error updating database: {e}")

# API 端点
@app.get("/api/current-anime", response_model=List[AnimeInfoInDB])
async def get_current_anime(skip: int = 0, limit: int = 100):
    anime_list = await collection.find().skip(skip).limit(limit).to_list(length=limit)
    return anime_list

@app.get("/api/anime/{year}/{month}", response_model=List[AnimeInfoInDB])
async def get_anime_by_month(year: int, month: int, skip: int = 0, limit: int = 100):
    anime_list = await collection.find({"year": year, "month": month}).skip(skip).limit(limit).to_list(length=limit)
    return anime_list

@app.get("/api/anime/search", response_model=List[AnimeInfoInDB])
async def search_anime(query: str, skip: int = 0, limit: int = 100):
    anime_list = await collection.find({"$text": {"$search": query}}).skip(skip).limit(limit).to_list(length=limit)
    return anime_list

@app.post("/api/update-anime")
async def update_anime(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_anime_database)
    return {"message": "Anime update task has been scheduled"}

async def check_and_update_anime_database():
    # 检查是否是每月的第一天
    if datetime.now().day == 1:
        print("Starting monthly anime database update...")
        await update_anime_database()
    else:
        print("Not the first day of the month, skipping update.")

# 定时任务
async def schedule_anime_update():
    aioschedule.every().day.at("00:00").do(check_and_update_anime_database)
    while True:
        await aioschedule.run_pending()
        await asyncio.sleep(3600)  # 每小时检查一次

@app.on_event("startup")
async def startup_event():
    # 创建文本索引
    await collection.create_index([("title", "text"), ("summary", "text")])
    # 首次运行爬虫
    #await update_anime_database()
    # 设置定时任务
    asyncio.create_task(schedule_anime_update())

# 关闭数据库连接
@app.on_event("shutdown")
async def shutdown_event():
    client.close()


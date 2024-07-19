from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

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
state_dict.pop('bert.embeddings.position_ids', None)
model.load_state_dict(state_dict, strict=False)

class AnimeInput(BaseModel):
    title: str
    summary: str

@app.post("/predict-rating")
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

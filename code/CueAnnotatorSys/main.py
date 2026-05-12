from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()

# 挂载静态文件服务（音频文件）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(__file__).resolve().parent
app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")
app.mount("/agent", StaticFiles(directory=str(PROJECT_ROOT / "agent" / "outputs")), name="agent")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 标注结果保存目录
OUTPUT_DIR = APP_ROOT / "annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class CueSpan(BaseModel):
    start: float
    end: float

class Cue(BaseModel):
    cue_id: int
    text: str
    original_span: CueSpan
    corrected_span: CueSpan
    status: str  # unchanged, modified, added, deleted

class AnnotationData(BaseModel):
    dataset: str
    sample_id: str
    cues: List[Cue]

@app.post("/api/save-annotation")
async def save_annotation(data: AnnotationData):
    try:
        # 构建文件名（使用固定名称，覆盖之前的内容）
        filename = f"annotation_{data.dataset}_{data.sample_id}.json"
        filepath = OUTPUT_DIR / filename
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data.dict(), f, ensure_ascii=False, indent=2)
        
        print(f"Annotation saved: {filepath}")
        
        return {
            "success": True,
            "message": "Annotation saved successfully",
            "filepath": str(filepath)
        }
    except Exception as e:
        print(f"Error saving annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/annotation/{dataset}/{sample_id}")
async def get_annotation(dataset: str, sample_id: str):
    """获取指定样本的标注内容（支持时间戳后缀）"""
    try:
        # 查找所有匹配的文件（支持有时间戳后缀的旧格式）
        prefix = f"annotation_{dataset}_{sample_id}"
        matching_files = []
        
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith(prefix) and filename.endswith('.json'):
                filepath = OUTPUT_DIR / filename
                # 获取文件修改时间
                mtime = os.path.getmtime(filepath)
                matching_files.append((filename, filepath, mtime))
        
        if matching_files:
            # 按修改时间排序，取最新的
            matching_files.sort(key=lambda x: x[2], reverse=True)
            latest_filepath = matching_files[0][1]
            
            with open(latest_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"exists": True, "data": data, "filename": matching_files[0][0]}
        else:
            return {"exists": False, "data": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/annotations")
async def get_annotations():
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
        return {"annotations": [{"filename": f, "filepath": str(OUTPUT_DIR / f)} for f in files]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "outputDir": str(OUTPUT_DIR)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)

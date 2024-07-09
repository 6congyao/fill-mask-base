from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
from typing import List


class SentenceList(BaseModel):
    sentences: List[str]


unmasker = pipeline("fill-mask", model="FacebookAI/roberta-base")

app = FastAPI()


@app.post("/sentences", response_model=SentenceList)
async def fill_mask_sentences(input: SentenceList):
    try:
        results = [unmasker(s)[0]['sequence'] for s in input.sentences]
        return SentenceList(sentences=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6000)

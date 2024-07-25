from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel
from typing import List
# from paraphraser import paraphrase
from roberta import unmask


class SentenceList(BaseModel):
    sentences: List[str]


app = FastAPI()


@app.post("/sentences", response_model=SentenceList)
async def fill_mask_sentences(input: SentenceList):
    try:
        results = [unmask(s)['sequence'] for s in input.sentences]
        # results = [paraphrase(s)[0] for s in input.sentences]
        # for prediction in results:
        #     print(prediction)
        return SentenceList(sentences=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6000)

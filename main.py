import numpy as np
import torch
from models.scoring_models import ReadingScoreModel, WritingScoreModel, MathScoreModel
import models.scoring_models
from fastapi import FastAPI

app = FastAPI()

writing_model: WritingScoreModel = WritingScoreModel(11, 1)
reading_model: ReadingScoreModel = ReadingScoreModel(11, 1)
math_model: MathScoreModel = MathScoreModel(11, 1)

if __name__ == '__main__':
    writing_model = torch.load('models/writing_score_model.pt')
    reading_model = torch.load('models/reading_score_model.pt')
    math_model = torch.load('models/math_score_model.pt')


model_input = torch.tensor([1., 2., 5., 0., 1., 2., 0., 1., 1., 1., 0.], dtype=torch.float32)
writing_model.eval()
reading_model.eval()
math_model.eval()

writing_output = round(float(writing_model(model_input)[0]), 1)
reading_output = round(float(reading_model(model_input)[0]), 1)
math_output = round(float(math_model(model_input)[0]), 1)

# print(f'predicted writing score: {writing_output}%')
# print(f'predicted reading score: {reading_output}%')
# print(f'predicted math score: {math_output}%')


@app.get('/')
async def root():
    return writing_output, reading_output, math_output


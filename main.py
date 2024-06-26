import torch
from fastapi import FastAPI
from pydantic import BaseModel
from models.scoring_models import ReadingScoreModel, WritingScoreModel, MathScoreModel

WRITING_MODEL_STATE_PATH = 'models/writing_score_model_state.pt'
READING_MODEL_STATE_PATH = 'models/reading_score_model_state.pt'
MATH_MODEL_STATE_PATH = 'models/math_score_model_state.pt'


class PredictionData(BaseModel):
    Gender: float
    EthnicGroup: float
    ParentEduc: float
    LunchType: float
    TestPrep: float
    ParentMaritalStatus: float
    PracticeSport: float
    IsFirstChild: float
    NrSiblings: float
    TransportMeans: float
    WklyStudyHours: float


class GenericInput(BaseModel):
    prediction: str
    data: PredictionData


app = FastAPI()

writing_model: WritingScoreModel = WritingScoreModel(11, 1)
writing_model.load_state_dict(torch.load(WRITING_MODEL_STATE_PATH))

reading_model: ReadingScoreModel = ReadingScoreModel(11, 1)
reading_model.load_state_dict(torch.load(READING_MODEL_STATE_PATH))

math_model: MathScoreModel = MathScoreModel(11, 1)
math_model.load_state_dict(torch.load(MATH_MODEL_STATE_PATH))

MODEL_CLASS_BY_PREDICTION_TYPE = {
    'writing': writing_model,
    'reading': reading_model,
    'math': math_model
}


def convert_prediction_data_to_tensor(prediction_data: PredictionData):
    return torch.tensor(data=[prediction_data.Gender, prediction_data.EthnicGroup, prediction_data.ParentEduc,
                              prediction_data.LunchType, prediction_data.TestPrep, prediction_data.ParentMaritalStatus,
                              prediction_data.PracticeSport, prediction_data.IsFirstChild, prediction_data.NrSiblings,
                              prediction_data.TransportMeans, prediction_data.WklyStudyHours],
                        dtype=torch.float32)


@app.get('/')
async def root():
    return 'SUML Project - Gr.1'


@app.post('/predictMerged')
async def predict(prediction_input: GenericInput):
    tensor_input = convert_prediction_data_to_tensor(prediction_input.data)
    output = (
        f'writing score: {round(float(writing_model(tensor_input)[0]), 1)}%',
        f'reading score: {round(float(reading_model(tensor_input)[0]), 1)}%',
        f'math score: {round(float(math_model(tensor_input)[0]), 1)}%'
    )
    return output


@app.post('/predictWriting')
async def predict(prediction_input: GenericInput):
    tensor_input = convert_prediction_data_to_tensor(prediction_input.data)
    output = (round(float(writing_model(tensor_input)[0]), 1))
    return output


@app.post('/predictReading')
async def predict(prediction_input: GenericInput):
    tensor_input = convert_prediction_data_to_tensor(prediction_input.data)
    output = (round(float(reading_model(tensor_input)[0]), 1))
    return output


@app.post('/predictMath')
async def predict(prediction_input: GenericInput):
    tensor_input = convert_prediction_data_to_tensor(prediction_input.data)
    output = (round(float(math_model(tensor_input)[0]), 1))
    return output

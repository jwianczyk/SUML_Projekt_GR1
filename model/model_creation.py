import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from data_preparation import prepare_data
import numpy as np


def model_creation(df: prepare_data()):
    num_rows = len(df)
    input_columns = list(df.drop(['WritingScore', 'MathScore', 'ReadingScore'], axis=1).columns)
    output_columns = df.columns[-1]

    def dataframe_to_arrays(dataframe: pd.DataFrame) -> list[np.ndarray]:
        df1 = dataframe.copy(deep=True)
        input_array = df1.drop(['WritingScore', 'MathScore', 'ReadingScore'], axis=1).values
        target_array = df1[['WritingScore']].values
        return [input_array, target_array]

    inputs_array = dataframe_to_arrays(df)[0]
    targets_array = dataframe_to_arrays(df)[1]

    inputs = torch.from_numpy(inputs_array).to(torch.float32)
    targets = torch.from_numpy(targets_array).to(torch.float32)

    dataset = TensorDataset(inputs, targets)

    val_percent = 0.1
    val_size = int(num_rows * val_percent)
    train_size = num_rows - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 64

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    input_size = len(input_columns)
    output_size = 1


if __name__ == '__main__':
    pass
import pandas as pd


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Expanded_data_with_more_features.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    for column in df.columns:
        df[column] = df[column].astype('category').cat.codes
        df[column] = df[column].fillna(df[column].mean())
    return df


if __name__ == '__main__':
    prepare_data()

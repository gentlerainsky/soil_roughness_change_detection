import pandas as pd
import numpy as np

def preprocess_ndvi(
        df,
        training_period_from,
        training_period_to,
        testing_period_from,
        testing_period_to
    ):
    df = df.copy()
    df.date = pd.to_datetime(df.date)
    # nir -> B5
    # red -> B4
    df['NDVI'] = (df['B5'] - df['B4']) / (df['B5'] + df['B4'])
    training_df = df[
        (df.date > pd.to_datetime(training_period_from))
        & (df.date <= pd.to_datetime(training_period_to))
    ].reset_index().set_index(['date', 'field_id'])
    testing_df = df[
        (df.date > pd.to_datetime(testing_period_from))
        & (df.date <= pd.to_datetime(testing_period_to))
    ].reset_index().set_index(['date', 'field_id'])
    return training_df, testing_df


def preprocess_precipitation(
        df,
        training_period_from,
        training_period_to,
        testing_period_from,
        testing_period_to):
    df = df.copy()
    df['precipitation_before'] = df['precipitation'].shift(1)
    training_df = df[
        (df.date > pd.to_datetime(training_period_from))
        & (df.date <= pd.to_datetime(training_period_to))
    ].reset_index().set_index(['date'])
    testing_df = df[
        (df.date > pd.to_datetime(testing_period_from))
        & (df.date <= pd.to_datetime(testing_period_to))
    ].reset_index().set_index(['date'])
    return training_df, testing_df


def calculate_backscatter_ratio(df):
    df = df.copy()
    # to avoid the case that np.log(0)
    VV = df['VV'] + 1
    VH = df['VH'] + 1
    df['VV_diff'] = VV - VV.shift(1)
    df['VH_diff'] = VH - VH.shift(1)
    df['from_date'] = df['date'].shift(1) + pd.DateOffset(1)
    df['VV_ratio'] = np.log(VV) - np.log(VV.shift(1))
    df['VH_ratio'] = np.log(VH) - np.log(VH.shift(1))
    df['VH_VV_ratio_diff'] = (np.log(VH) - np.log(VV)) - (np.log(VH).shift(1) - np.log(VV).shift(1))
    df = df.dropna()
    return df


def preprocess_backscatter(
        df,
        training_period_from,
        training_period_to,
        testing_period_from,
        testing_period_to
    ):
    df = df.reset_index()[['field_id', 'date', 'VV', 'VH']].groupby(['date', 'field_id']).mean().reset_index()
    df.date = pd.to_datetime(df.date)
    df = df.groupby(['field_id']).apply(calculate_backscatter_ratio, include_groups=False)
    df.index = df.index.droplevel(1)
    training_df = df[
        (df.date > pd.to_datetime(training_period_from))
        & (df.date <= pd.to_datetime(training_period_to))
    ].reset_index().set_index(['date', 'field_id'])
    testing_df = df[
        (df.date > pd.to_datetime(testing_period_from))
        & (df.date <= pd.to_datetime(testing_period_to))
    ].reset_index().set_index(['date', 'field_id'])
    return training_df, testing_df


def preprocess_harrysfarm_activity_log(
        tillage_df,
        training_period_from,
        training_period_to,
        testing_period_from,
        testing_period_to
    ):
    tillage_df = tillage_df.copy()
    tillage_df['Field'] = tillage_df['Field'] - 1
    tillage_df.Date = pd.to_datetime(tillage_df.Date)
    train_tillage_df = tillage_df[(tillage_df.Date > pd.to_datetime(training_period_from))
                            & (tillage_df.Date <= pd.to_datetime(training_period_to))]
    train_tillage_df = train_tillage_df.set_index(['Date', 'Field']).sort_index()

    test_tillage_df = tillage_df[
        (tillage_df.Date > pd.to_datetime(testing_period_from))
        & (tillage_df.Date <= pd.to_datetime(testing_period_to))
    ]
    test_tillage_df = test_tillage_df.set_index(['Date', 'Field']).sort_index()
    return train_tillage_df, test_tillage_df

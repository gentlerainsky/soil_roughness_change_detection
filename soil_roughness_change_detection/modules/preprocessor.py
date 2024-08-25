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
    df = df.sort_values('date')
    # to avoid the case that np.log(0)
    VV = df['VV']
    VH = df['VH']
    df['VV_diff'] = VV - VV.shift(1)
    df['VH_diff'] = VH - VH.shift(1)
    df['from_date'] = df['date'].shift(1) + pd.DateOffset(1)
    df['VV_ratio'] = VV / VV.shift(1)
    df['VH_ratio'] = VH / VH.shift(1)
    df['VH_VV_ratio_diff'] = (VH / VV) - (VH.shift(1) - VV.shift(1))
    df = df.dropna()
    return df


# Ref: https://www.nature.com/articles/s41597-021-01059-7
def normalize_to_38_degree(df, beta = -0.13):
    df = df.copy()
    df['VV'] = df['VV'] - beta*(df['angle'] - 38)
    df['VH'] = df['VH'] - beta*(df['angle'] - 38)
    return df


def preprocess_backscatter(
        df,
        training_period_from,
        training_period_to,
        testing_period_from,
        testing_period_to,
        is_normalized_to_38_degree=False,
        is_group_by_orbit=False
    ):

    df = df.reset_index()

    if is_normalized_to_38_degree:
        df = normalize_to_38_degree(df)

    if is_group_by_orbit:
        df = df[['field_id', 'date', 'VV', 'VH', 'orbit']]\
            .groupby(['date', 'field_id', 'orbit']).mean().reset_index()
        df.date = pd.to_datetime(df.date)
        df = df.groupby(['field_id', 'orbit']).apply(calculate_backscatter_ratio, include_groups=False)
        df.index = df.index.droplevel(2)
    else:
        df = df[['field_id', 'date', 'VV', 'VH']]\
            .groupby(['date', 'field_id']).mean().reset_index()
        df.date = pd.to_datetime(df.date)
        df = df.groupby(['field_id']).apply(calculate_backscatter_ratio, include_groups=False)
        df.index = df.index.droplevel(1)
    training_df = df[
        (df.date > pd.to_datetime(training_period_from))
        & (df.date <= pd.to_datetime(training_period_to))
    ].reset_index().set_index(['date', 'field_id']).sort_index()
    testing_df = df[
        (df.date > pd.to_datetime(testing_period_from))
        & (df.date <= pd.to_datetime(testing_period_to))
    ].reset_index().set_index(['date', 'field_id']).sort_index()
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

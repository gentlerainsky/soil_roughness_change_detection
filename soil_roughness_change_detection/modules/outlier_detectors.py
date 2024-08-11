import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


def dbscan_outlier_detector(
    sample_arr,
    model_args
):
    scaler = StandardScaler()
    scaler.fit(sample_arr)
    transformed_arr = scaler.transform(sample_arr)
    db = DBSCAN(**model_args).fit(transformed_arr)
    labels = db.labels_
    mask = labels == -1
    return mask


def mahalanobis_distance_outlier_detector(sample_arr, model_args=None, random_state=1234):
    alpha = model_args.get('p_values', 0.05)
    chi_sqrt = chi2.ppf(1 - alpha/2, sample_arr.shape[1])
    # robust location and spread estimation
    cov = MinCovDet(random_state=random_state).fit(sample_arr)
    C = cov.covariance_
    t = cov.location_
    c_inv = np.linalg.pinv(C)
    # find outliers
    mahalanobis_distance = np.diag(((sample_arr - t) @ (c_inv)) @ (sample_arr - t).T)
    mask = mahalanobis_distance > chi_sqrt
    return mask


def isolation_forest_outlier_detector(sample_arr, model_args):
    clf = IsolationForest(**model_args, warm_start=True)
    clf.fit(sample_arr)
    mask = clf.predict(sample_arr) == -1
    return mask


def outlier_detection(
        df,
        detector,
        params = ['VV_ratio', 'VH_ratio', 'VH_VV_ratio_diff']
    ):
    date_array = df.index.get_level_values(0).unique()
    all_outlier_df = pd.DataFrame([], columns=['id', 'date'])
    all_outlier_df.id = all_outlier_df.id.astype(np.int64)
    all_outlier_df.date = all_outlier_df.id.astype('datetime64[ns]')
    for date_idx in range(len(date_array)):
        sample_arr = df.xs(date_array[date_idx], level=0)[params].values
        mask = detector(sample_arr)
        outlier_df = pd.DataFrame({
            'id': np.where(mask)[0],
        }, columns=['id', 'date'])
        outlier_df.date = date_array[date_idx]
        all_outlier_df = pd.concat([all_outlier_df, outlier_df])
    return all_outlier_df

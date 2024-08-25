import datetime
from tqdm import tqdm
from soil_roughness_change_detection.modules.outlier_detectors import outlier_detection


def run_experiment(df, tillage_df, interval_df, feature, outlier_detector, parameter_combinations):
    results = []
    start_time = datetime.datetime.now()
    for parameter_combination in tqdm(parameter_combinations):
        outlier_df = outlier_detection(
            df,
            lambda x: outlier_detector(x, parameter_combination),
            params=feature
        )
        outlier_df = outlier_df.merge(interval_df)[['from_date', 'date', 'id']]
        true_positives, false_positives, false_negatives = evaluate_outliers(outlier_df, tillage_df)
        result = {
            'config': parameter_combination,
            'results': calculate_result(true_positives, false_positives, false_negatives),
        }
        output_outlier_df = outlier_df[outlier_df.id.isin([0, 1, 2, 3, 4])].copy()
        output_outlier_df['from_date'] = output_outlier_df['from_date'].dt.strftime('%Y-%m-%d')
        output_outlier_df['date'] = output_outlier_df['date'].dt.strftime('%Y-%m-%d')
        result['outlier'] = output_outlier_df.to_dict(orient='records')
        results.append(result)
    used_time = (datetime.datetime.now() - start_time).seconds
    print(f'Finish with {used_time // 3600:02d}:{used_time // 60:02d}:{used_time % 60:02d}')
    results = sorted(results, key=lambda x: -x['results']['f_score'])
    return results

def evaluate_outliers(outlier_df, tillage_df):
    test_vote_df = outlier_df[outlier_df.id.isin([0, 1, 2, 3, 4])].copy()
    test_vote_df['matched'] = False
    false_negatives = []
    true_positives = 0
    for field_idx in range(5):
        field_df = test_vote_df[test_vote_df.id == field_idx]
        field_tillage_df = tillage_df.xs(field_idx, level=1)
        for date, _ in field_tillage_df.iterrows():
            check_df = field_df[(date >= field_df.from_date) & (date <= field_df.date)]
            if check_df.shape[0] > 0:
                test_vote_df.loc[check_df.index, 'matched'] = True
                true_positives += 1
            else:
                false_negatives.append({
                    'date': date,
                    'field': field_idx
                })
    false_positives = int(test_vote_df[~test_vote_df['matched']].groupby('id').count()['date'].sum())
    false_negatives = len(false_negatives)
    return true_positives, false_positives, false_negatives

def calculate_result(true_positives, false_positives, false_negatives):
    result = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0
            else 0
        ),
        'recall': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0
            else 0,
    }
    a = 1 / result['precision'] if result['precision'] > 0 else 0
    b = 1 / result['recall'] if result['recall'] > 0 else 0
    
    result['f_score'] = 2 / (a + b) if (a + b) > 0 else 0
    return result

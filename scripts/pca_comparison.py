"""Supplementary PCA experiment; never evaluates or tunes against the holdout.

Run with the same dependencies as project.ipynb:
python scripts/pca_comparison.py --train data/train.csv --out pca_results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/train.csv'))
    parser.add_argument('--out', type=Path, default=Path('pca_results'))
    args = parser.parse_args()
    frame = pd.read_csv(args.train)
    features = frame.columns.drop(['id', 'Hardness']).tolist()
    X, y = frame[features], frame['Hardness']
    X_dev, _, y_dev, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    estimators = {
        'PCA 95% + Ridge': Ridge(alpha=10.0),
        'PCA 95% + Gradient boosting': HistGradientBoostingRegressor(
            loss='absolute_error', learning_rate=0.06, max_iter=250,
            max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=42
        ),
    }
    records = []
    diagnostics = {'sklearn_version': sklearn.__version__, 'development_rows': len(X_dev),
                   'features': features, 'holdout_evaluated': False, 'folds': {}}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver='full'), estimator)
        scores = cross_validate(pipeline, X_dev, y_dev, cv=cv, n_jobs=1,
            scoring={'medae': 'neg_median_absolute_error', 'mae': 'neg_mean_absolute_error'},
            return_estimator=True, error_score='raise')
        values = -scores['test_medae']
        components = [int(fit.named_steps['pca'].n_components_) for fit in scores['estimator']]
        records.append({'Model': name, 'CV MedAE': values.mean(),
                        'Fold SD': values.std(ddof=1), 'CV MAE': (-scores['test_mae']).mean(),
                        'Components per fold': ','.join(map(str, components))})
        diagnostics['folds'][name] = {'medae': values.tolist(), 'components': components}
    # Development-only descriptive spectrum. This fit is not used to score any fold.
    spectrum = make_pipeline(StandardScaler(), PCA(svd_solver='full')).fit(X_dev)
    ratios = spectrum.named_steps['pca'].explained_variance_ratio_
    diagnostics['development_variance_ratios'] = ratios.tolist()
    diagnostics['development_cumulative_variance'] = np.cumsum(ratios).tolist()
    diagnostics['components_for_95_percent'] = int(np.searchsorted(np.cumsum(ratios), 0.95) + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(args.out / 'Mohs_PCA_Comparison.csv', index=False)
    (args.out / 'Mohs_PCA_Diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print(table.to_string(index=False))
    print('Development cumulative variance:', np.cumsum(ratios))
    print('scikit-learn:', sklearn.__version__)


if __name__ == '__main__':
    main()

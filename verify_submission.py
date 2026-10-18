"""Quick checks for the bundled models and forecasts; no retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pipeline import Proxy, PARAMS

def main():
    base = Path(__file__).resolve().parent
    root = base / 'verified_run'
    lock = json.loads((root / 'selection_lock.json').read_text())
    assert hashlib.sha256((base / 'pipeline.py').read_bytes()).hexdigest() == lock['code_sha256']
    assert hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() == lock['config_sha256']
    proxy = Proxy(root)
    sim = pd.read_csv(root / 'Inputs/linked_parameters_production.csv', parse_dates=['Date'])
    sample = sim.iloc[[0]]
    rates = proxy.predict(sample[PARAMS].iloc[0].to_numpy(), sample.Date)
    assert np.isfinite(rates).all() and (rates >= 0).all()
    for shift in (0, 3):
        data = pd.read_csv(root / f'07_Export_Q{shift}/selected_10_forecast.csv', parse_dates=['Date'])
        assert data.Candidate_ID.nunique() == 10 and len(data) == 810
        assert np.isfinite(data.select_dtypes('number')).all().all()
        assert (data[['Oil_Rate', 'Gas_Rate', 'Water_Rate']] >= 0).all().all()
        assert data.Water_Cut.between(0, 1).all()
        assert np.allclose(data.Oil_Rate + data.Water_Rate, data.Total_Liquid_Rate)
        for _, group in data.groupby('Candidate_ID'):
            group = group.sort_values('Date')
            assert list(group.Date) == list(pd.date_range('2008-01-01', '2028-01-01', freq='QS'))
            assert (np.diff(group.Forecast_Cumulative_Oil_MMSTB) >= 0).all()
    print('PASS: model loading and prediction, code/config fingerprints, both 10-case exports.')

if __name__ == '__main__':
    main()

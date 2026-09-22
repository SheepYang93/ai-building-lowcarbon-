import pandas as pd
from src.pipeline import features,anomaly_table

def toy():
    dates=pd.date_range("2017-01-01",periods=70)
    return pd.DataFrame({"building_id":["B"]*70,"date":dates,"meter_reading":[100.0]*69+[300.0],"sub_primaryspaceusage":["Classroom"]*70,"sqm":[1000.0]*70,"site_id":["S1"]*70,"airTemperature":[20.0]*70,"cloudCoverage":[2.0]*70,"dewTemperature":[10.0]*70,"windSpeed":[2.0]*70})

def test_shifted_lag():
    d,f=features(toy()); assert d.loc[d.date.eq(pd.Timestamp("2017-01-29")),"lag1"].iloc[0]==100.0

def test_anomaly_output():
    out=anomaly_table(toy()); assert "priority_score" in out.columns and len(out)==1

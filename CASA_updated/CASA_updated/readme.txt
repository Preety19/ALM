Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,June-15-2023,Preety_tiwari


Header: Description
CASA model observes the behavior and distribution of a bank account in the future using time series modeling.


Header: Performence Kpi
[{'MAPE': 0.07204}]


Header: Input Parameters
[{name : date, type :datetime},
{name : amount, type :float64}]

Header: Output Parameters
[{name:Dist_df, type:float}]


Header: Libraries
[pandas, numpy, sklearn,statsmodels,pymannkendall,argparse,datetime]
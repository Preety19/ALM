Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V2,November-16-2023,Preety_tiwari

Header: Description
Forecasting model uses macro-economic variables to forecast the target end balance of Bank for each month.

Header: Performance Kpi
[{'MSE': 26734200669.496395, 'MAE': 158990.76314307423, 'MAPE': 0.06497371360793101}]

Header: Input Parameters
[
{name : Date, type :object},
{name : GDP, type :float64},
{name : Dollar, type :float64},
{name : Pound, type :float64},
{name : Yen, type :float64},
{name : Euro, type :float64},
{name : Inflation_Rate, type :float64},
{name : Target End Balance, type :float64}
]

Header: Output Parameters
[{name:modelOutput, type:Dataframe}]


Header: Libraries
[pandas,numpy,sklearn,argparse,datetime,sys,os,json,matplotlib]
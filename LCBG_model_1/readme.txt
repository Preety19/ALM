Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,August-10-2023,Preety_tiwari

Header: Description
LC/BG model uses macro-economic variables to forecast the Percentage of amount invoked during a certain period of time.

Header: Performence Kpi
[{'MSE': 1.093864337792761, 'MAE': 0.8253837028477116, 'MAPE': 0.7944108764371548}]

Header: Input Parameters
[{name : Account_No, type :object},
 {name : Header, type :object},
 {name : Currency_Code, type :object},
 {name : Issued_Amount, type :object},
 {name : Invoked_Devolved, type :object},
 {name : Issued_Date, type :object},
 {name : Devolvement_Date, type :object},
{name : Date, type :object},
{name : GDP, type :float64},
{name : CPI, type :float64},
{name : Inflation_Rate, type :float64},
{name : Dollar, type :float64},
{name : Pound, type :float64},
{name : Yen, type :float64},
{name : Euro, type :float64}
]

Header: Output Parameters
[{name:modelOutput, type:Dataframe}]


Header: Libraries
[pandas, numpy,sklearn,argparse,datetime,sys,os]
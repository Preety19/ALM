Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,July-13-2023,Preety_tiwari

Header: Description
Drawn Undrawn OD model observes the distribution of the Drawn and Undrawn amount over a given period of time.


Header: Performence Kpi
[{'MAPE': 0.27978101863087507}]


Header: Input Parameters
[{name : Date, type :datetime},
{name : Drawn_Amount, type :float64},
{name : Undrawn_Amount, type :float64}]

Header: Output Parameters
[{name:modelOutput, type:dictionary}]


Header: Libraries
[pandas, numpy, sklearn,statsmodels,pymannkendall,argparse,datetime]
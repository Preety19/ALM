Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,August-02-2023,Preety_tiwari

Header: Description
Prepayment TL model observes the distribution of the amount over various time buckets for the prepaid loan amounts.

Header: Performence Kpi
[{'MAPE': 0.07204}]

Header: Input Parameters
[{name : Exposure_No, type :object},
{name : Customer_ID, type :object},
{name : Product_Code, type :object},
{name : Currency_Code, type :object},
{name : Cashflow_date_before_prepayment_of_loan, type :object},
{name : Cashflow_Date_due_to_prepayment_of_loan, type :object},
{name : PCR_Amount_of_original_loan_before_prepayment, type :float64}]

Header: Output Parameters
[{name:modelOutput, type:dictionary}]


Header: Libraries
[pandas, numpy,argparse,datetime]
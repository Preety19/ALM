Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,August-02-2023,Preety_tiwari

Header: Description
Foreclosure TD model observes the distribution of the amount over various time buckets.

Header: Input Parameters
[{name : Exposure_No, type :object},
{name : Customer_ID, type :object},
{name : Product_Code, type :object},
{name : Currency_Code, type :object},
{name : Original_maturity_date_before_premature_withdrawal, type :object},
{name : Maturity_Date_due_to_premature_withdrawal, type :object},
{name : Balance_of_original_deposit_before_prepayment_withdrawal, type :float64},
{name : Initial_Deposit_Amount_of_original_deposit_before_premature_withdrawal, type :float64}]

Header: Output Parameters
[{name:modelOutput, type:Dictionary}]

Header: Libraries
[pandas, numpy,argparse,datetime]
Header: Model Version
<Model Version>,<Release Date>,<Author>,<Test Result>,<Validation Result>
V1,November-23-2023,Preety_tiwari

Header: Description
Deterministic model computes the difference between the present value of assets and present value of liabilities.


Header: Input Parameters
[{name : Maturity Yrs, type :float64},
{name : Day_1, type :float64},
{name : Day_2, type :float64},
{name : Description, type :object},
{name : Overnight, type :float64},
{name : Over Overnight and upto 1 month, type :float64},
{name : Over 1 month and upto 3 months,type :float64},
{name : Over 3 months and upto 6 months, type :float64},
{name : Over 6 months and upto 9 months, type :float64},
{name : Over 9 months and upto 1 year, type :float64},
{name : Over 1 year and upto 1_5 year, type :float64},
{name : Over 1_5 year and upto 2 years, type :float64},
{name : Over 2 years and upto 3 years, type :float64},
{name : Over 3 years and upto 4 years, type :float64},
{name : Over 4 years and up to 5 years, type :float64},
{name : Over 5 years and upto 6 years, type :float64},
{name : Over 6 years and upto 7 years, type :float64},
{name : Over 7 years and upto 8 years, type :float64},
{name : Over 8 years and upto 9 years, type :float64},
{name : Over 9 years and upto 10 years, type :float64},
{name : Over 10 years and upto 15 years, type :float64},
{name : Over 15 years and upto 20 years, type :float64},
{name : Over 20 years, type :float64},
{name : Non-Sensitive, type :float64},
{name : Total, type :float64},
{name : Sensitive Total, type :float64},
{name : Description, type :object},
{name : Overnight, type :float64},
{name : Over Overnight and upto 1 month, type :float64},
{name : Over 1 month and upto 3 months, type :float64},
{name : Over 3 months and upto 6 months, type :float64},
{name : Over 6 months and upto 9 months, type :float64},
{name : Over 9 months and upto 1 year, type :float64},
{name : Over 1 year and upto 1_5 year, type :float64},
{name : Over 1_5 year and upto 2 years, type :float64},
{name : Over 2 years and upto 3 years, type :float64},
{name : Over 3 years and upto 4 years, type :float64},
{name : Over 4 years and up to 5 years, type :float64},
{name : Over 5 years and upto 6 years, type :float64},
{name : Over 6 years and upto 7 years, type :float64},
{name : Over 7 years and upto 8 years, type :float64},
{name : Over 8 years and upto 9 years, type :float64},
{name : Over 9 years and upto 10 years, type :float64},
{name : Over 10 years and upto 15 years, type :float64},
{name : Over 15 years and upto 20 years, type :float64},
{name : Over 20 years, type :float64},
{name : Non-Sensitive, type :float64},
{name : Total, type :float64},
{name : Sensitive Total, type :float64}]

Header: Output Parameters
[{name:modelOutput, type:dictionary}]


Header: Libraries
[pandas, numpy, matplotlib,datetime]
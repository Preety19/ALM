import pandas as pd
import numpy as np
import sys
import os
import argparse
import warnings
import json
warnings.filterwarnings('ignore')

def data_load(path):
    # pylint: disable=no-member
    data = pd.read_json(os.path.join(path,'data.json'))
    date_cols = list(data.columns[4:6])
    for i in range (0,len(date_cols)):
        data[date_cols[i]] = data[date_cols[i]].astype('datetime64[ns]')
    data['Prepayment_Period'] = ((data['Cashflow_date_before_prepayment_of_loan'] -     data['Cashflow_Date_due_to_prepayment_of_loan'])/np.timedelta64(1, 'D'))    
    return data

def bins(data,bin_numbers, bucket):
    # pylint: disable=no-member
    data['Decile_Bin'] = pd.qcut(data['PCR_Amount_of_original_loan_before_prepayment'], bin_numbers, labels=False)
    data = data[data['Decile_Bin'] == bucket]
    data = data.reset_index(drop = True)
    return data

def Distribution_df():
    # pylint: disable=no-member
        End_value_in_days = [0, 1, 7, 14, 28, 90, 180, 360, 1080, 1800, 29970]
        Start_value_in_days = [0]
        for i in range (0, len(End_value_in_days)-1):
            a = End_value_in_days[i]+1
            Start_value_in_days.append(a)
        Bucket_Code = ['Error','PL: O/N', 'PL: 2-7D', 'PL: 8-14D', 'PL: 15-28D', 'PL: 29D-3M', 'PL: 3M-6M', 'PL: 6M-1Y',
                       'PL: 1Y-3Y', 'PL: 3Y-5Y', 'PL:5Y+']       
        Distribution_zip = list(zip(Start_value_in_days, End_value_in_days, Bucket_Code)) ## mapping the data
        Distribution_df = pd.DataFrame(Distribution_zip,columns = ['Start_value_in_days', 'End_value_in_days', 'Bucket_Code'])
        return Distribution_df
    
def df_Res_Maturity(Distribution_df, df):
        # pylint: disable=no-member
        ON = []
        for i in range (0, len(df)):
            if (Distribution_df['End_value_in_days'][1] - df['Prepayment_Period'][i] <= 0):
                a = Distribution_df['End_value_in_days'][1]
                ON.append(a)
            else:  
                a = Distribution_df['End_value_in_days'][1] - df['Prepayment_Period'][i]
                ON.append(a)

        df_Res_maturity = pd.DataFrame(ON, columns=['PL: O/N'])
        df_Res_maturity['PL: 2-7D'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 8-14D'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 15-28D'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 29D-3M'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 3M-6M'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 6M-1Y'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 1Y-3Y'] = pd.Series(dtype='int')
        df_Res_maturity['PL: 3Y-5Y'] = pd.Series(dtype='int')
        df_Res_maturity['PL:5Y+'] = pd.Series(dtype='int')

        for i in range (0, len(df)):
            if (Distribution_df['End_value_in_days'][2] - df['Prepayment_Period'][i]) <= 0:
                df_Res_maturity['PL: 2-7D'][i] = Distribution_df['End_value_in_days'][2]
            else:  
                df_Res_maturity['PL: 2-7D'][i] = Distribution_df['End_value_in_days'][2] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][3] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 8-14D'][i] = Distribution_df['End_value_in_days'][3]
            else:  
                df_Res_maturity['PL: 8-14D'][i] = Distribution_df['End_value_in_days'][3] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][4] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 15-28D'][i] = Distribution_df['End_value_in_days'][4]
            else:  
                df_Res_maturity['PL: 15-28D'][i] = Distribution_df['End_value_in_days'][4] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][5] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 29D-3M'][i] = Distribution_df['End_value_in_days'][5]
            else:  
                df_Res_maturity['PL: 29D-3M'][i] = Distribution_df['End_value_in_days'][5] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][6] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 3M-6M'][i] = Distribution_df['End_value_in_days'][6]
            else:  
                df_Res_maturity['PL: 3M-6M'][i] = Distribution_df['End_value_in_days'][6] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][7] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 6M-1Y'][i] = Distribution_df['End_value_in_days'][7]
            else:  
                df_Res_maturity['PL: 6M-1Y'][i] = Distribution_df['End_value_in_days'][7] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][8] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 1Y-3Y'][i] = Distribution_df['End_value_in_days'][8]
            else:  
                df_Res_maturity['PL: 1Y-3Y'][i] = Distribution_df['End_value_in_days'][8] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][9] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL: 3Y-5Y'][i] = Distribution_df['End_value_in_days'][9]
            else:  
                df_Res_maturity['PL: 3Y-5Y'][i] = Distribution_df['End_value_in_days'][9] - df['Prepayment_Period'][i]

            if Distribution_df['End_value_in_days'][10] - df['Prepayment_Period'][i] <= 0:
                df_Res_maturity['PL:5Y+'][i] = Distribution_df['End_value_in_days'][10]
            else:  
                df_Res_maturity['PL:5Y+'][i] = Distribution_df['End_value_in_days'][10] - df['Prepayment_Period'][i]
        return df_Res_maturity

def Buckets(df_Res_maturity,Distribution_df):
    # pylint: disable=no-member
    bucket_df = df_Res_maturity.copy() 
    bucket_1 = []
    for i in range (0,len(df_Res_maturity)):
        for j in range (0, len(Distribution_df)-1):
            if df_Res_maturity['PL: O/N'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: O/N'][i] = a

            if df_Res_maturity['PL: 2-7D'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 2-7D'][i] = a

            if df_Res_maturity['PL: 8-14D'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 8-14D'][i] = a

            if df_Res_maturity['PL: 15-28D'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 15-28D'][i] = a

            if df_Res_maturity['PL: 29D-3M'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1]  + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 29D-3M'][i] = a

            if df_Res_maturity['PL: 3M-6M'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 3M-6M'][i] = a

            if df_Res_maturity['PL: 6M-1Y'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 6M-1Y'][i] = a

            if df_Res_maturity['PL: 1Y-3Y'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 1Y-3Y'][i] = a

            if df_Res_maturity['PL: 3Y-5Y'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL: 3Y-5Y'][i] = a

            if df_Res_maturity['PL:5Y+'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1) or df_Res_maturity['PL:5Y+'][i] > 29970:
                a = Distribution_df['Bucket_Code'][j+1]
                bucket_df['PL:5Y+'][i] = a    
    return bucket_df
        
def probability_distribution(bucket_df):
        # pylint: disable=no-member
        aa = [bucket_df[c].value_counts() for c in list(bucket_df.select_dtypes(include=['O']).columns)]
        df_final = pd.DataFrame(columns = list(bucket_df.columns), index = list(bucket_df.columns))
        for i in range(0, len(aa)):
            mapping = aa[i].to_dict()
            df_final.iloc[:,i] = df_final.iloc[:,i].index.map(mapping)
        df_final = df_final.fillna(0)
        df_final = (df_final/len(bucket_df)) * 100    
        df_final.round(decimals = 2)
        return df_final 
    
def cashflow_pattern(Conventional_Maturity_Amount, Percentage_of_princal_prepaid):
    # pylint: disable=no-member
    Rollover = -1 * Conventional_Maturity_Amount * (Percentage_of_princal_prepaid/100)
    Matured_Amount_by_conventional_maturity = Rollover + Conventional_Maturity_Amount
    time_period = ['day_1', 'day2_7', 'day8_14','day15_28','1_3_month','3_6_month','6months_1year','1_3_years','3_5_years', '5_years+']
    Distribution_df = pd.DataFrame(time_period,columns = ['Time_period'])
    Distribution_df['PL: O/N'] = pd.Series(dtype='float')
    Distribution_df['PL: 2-7D'] = pd.Series(dtype='float')
    Distribution_df['PL: 8-14D'] = pd.Series(dtype='float')
    Distribution_df['PL: 15-28D'] = pd.Series(dtype='float')
    Distribution_df['PL: 29D-3M'] = pd.Series(dtype='float')
    Distribution_df['PL: 3M-6M'] = pd.Series(dtype='float')
    Distribution_df['PL: 6M-1Y'] = pd.Series(dtype='float')
    Distribution_df['PL: 1Y-3Y'] = pd.Series(dtype='float')
    Distribution_df['PL: 3Y-5Y'] = pd.Series(dtype='float')
    Distribution_df['PL:5Y+'] = pd.Series(dtype='float')
    #Rollover_amounts  = []
    for j in range (0, len(df_final.columns)):
        for i in range (0, len(df_final)):
            Distribution_df.iat[i,j+1] = Rollover * df_final.iat[i,j]/100        
    #Rollover_amounts.append(aa)
    #Distribution_df.to_csv('Distribution_table.csv')
    return Distribution_df
    
if __name__ == '__main__': 
    arguments = sys.argv[0] ###  Input File Path
    path = os.getcwd()
    parent = os.path.dirname(path) ### Fetching parent directory
    #here we are picking the model output - logic can be modified as per need
    df = data_load(path) 
    #df = data_load('/mnt/c/Users/preety.tiwari/Documents/ALM_TS/Prepayment/Datasets')
    parser = argparse.ArgumentParser()
    
#     parser.add_argument('filepath', type=argparse.FileType('r'),
#                         help='File that is supposed to be read')
    parser.add_argument('--bins', type=int, default= 10,help='Number of buckets')
    parser.add_argument('--Bucket', type=int, default= 5, help='Bucket number for which calculations should be done')                   
    parser.add_argument('--Conventional_Maturity_Amount', default= 100000, type=float,help='Maturity amount')
    parser.add_argument('--Percentage_of_princal_prepaid',default= 11, type = float,help='Percentage of the total prepaid loans')
    args = parser.parse_args()               
                   
    dt = bins(df,args.bins, args.Bucket)
    Distribution_df = Distribution_df()
    df_Res_Maturity = df_Res_Maturity(Distribution_df,dt)
    Bucket_df = Buckets(df_Res_Maturity,Distribution_df)
    df_final = probability_distribution(Bucket_df)
    modelOutput = cashflow_pattern(args.Conventional_Maturity_Amount,args.Percentage_of_princal_prepaid)
    #print(modelOutput)
    # Convert DataFrame to JSON
    json_output = modelOutput.to_json(orient='records')

    # Create a dictionary with the desired key
    #model_output = {"modelOutput": json_output}

    # Convert the dictionary to a JSON string
    #modelOutput = json.dumps(model_output, indent=4)
    print("modelOutput: ", json_output)
     

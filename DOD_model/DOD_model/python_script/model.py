## Importing all the required libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import sys
import os
import argparse
import datetime
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")
import json

# Load the data and set the Date column as the index
def data_load(path):
    data = pd.read_json(os.path.join(path,'data.json'))
    data['Date'] = data['Date'].astype('datetime64[ns]') #Converting date from object to datetime for further analysis
    data = (data.sort_values('Date')).reset_index(drop = True)
    data = data.set_index('Date')
    return data

def train_test_splitting(data):
    test_size = int(np.floor(len(data)/4))
    train_df = data[:-test_size] ##Storing 2 years data for training the dataset
    test_df = data[-test_size:]  ## Using 1 year data to check the performance of the model
    y_hat_avg = test_df.copy()
    return train_df, test_df, y_hat_avg,test_size

def generate_forecast(model, df, **params):
    if model == 'simple':
        #Code for Simple Exponential Smoothing
        model1 = SimpleExpSmoothing(df['Undrawn_Amount'])
        fit1 = model1.fit(smoothing_level=params['smoothing_level'])
        print("Running Simple Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['Undrawn_Amount'] = fit1.forecast(values)
        return y_hat_avg
        
    elif model == 'double':
        #Code for Double Exponential Smoothing
        model1 = Holt(df['Undrawn_Amount'])
        fit1 = model1.fit(smoothing_level=params['smoothing_level'], smoothing_trend=params['smoothing_trend'])
        print("Running Double Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['Undrawn_Amount'] = fit1.forecast(values)       
        return y_hat_avg
       
    elif model == 'triple':
        # Code for Triple Exponential Smoothing
        model1 = ExponentialSmoothing(df['Undrawn_Amount'], trend=params['trend'], seasonal=params['seasonal'], seasonal_periods=params['seasonal_periods'], freq = params['freq'], damped_trend = params['damped_trend'], initialization_method = params['initialization_method'])
        fit1 = model1.fit()
        print("Running Triple Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['Undrawn_Amount'] = fit1.forecast(values)        
        return y_hat_avg
        
    else:
        print("Invalid selection")
        return None
    
def drawn_amount_computed(y_hat_avg):
    df_drawn = pd.DataFrame()
    df_drawn['Drawn_Amount'] = 1000 - y_hat_avg['Undrawn_Amount']
    df_drawn['Date'] = df_drawn.index
    df_drawn.reset_index(drop = True, inplace = True)
    return df_drawn

def Mape_score(test_df,y_hat_avg):
    mape1 = mean_absolute_percentage_error(test_df['Undrawn_Amount'],y_hat_avg['Undrawn_Amount'])
    print (f'Mean Absolute Percentage error of this model is {mape1}')
    return mape1

def Distribution_df():
    End_value_in_days = [0, 1, 7, 14, 28, 90, 180, 360, 1080, 1800, 29970]
    Start_value_in_days = [0]
    for i in range (0, len(End_value_in_days)-1):
        a = End_value_in_days[i]+1
        Start_value_in_days.append(a)    
    Bucket_Code = ['Error','PL1: O/N', 'PL2: 2-7D', 'PL3: 8-14D', 'PL4: 15-28D', 'PL5: 29D-3M', 'PL6: 3M-6M', 'PL7: 6M-1Y',
                   'PL8: 1Y-3Y', 'PL9: 3Y-5Y', 'PL10:5Y+']     
    Distribution_zip = list(zip(Start_value_in_days, End_value_in_days, Bucket_Code)) ## mapping the data
    Distribution_df = pd.DataFrame(Distribution_zip,columns = ['Start_value_in_days', 'End_value_in_days', 'Bucket_Code'])
    return Distribution_df


def Time_bucket(df, Distribution_df):
    #df = data.copy()
    df['Date'] = df.index
    df = df.reset_index(drop = True)
    df['No_Days_from_Start_Date'] = df.index
    df['Drawdown'] = np.nan
    for i in range (0, len(df)):
        df['Drawdown'][i] = max((df['Undrawn_Amount'][0] - df['Undrawn_Amount'][i]),0)
    df['Bucket'] = ''
    for i in range (1,len(df)):
        for j in range (0, len(Distribution_df)-1):
            if df['No_Days_from_Start_Date'][i]  in range(Distribution_df['Start_value_in_days'][j+1], Distribution_df['End_value_in_days'][j+1] + 1):
                df['Bucket'][i] = Distribution_df['Bucket_Code'][j+1]
    return df               

def maximum_amount_cumulative(df):
    group = df.groupby(['Bucket'], as_index=False)
    buckets = group['Drawdown'].max()
    Maximum_value = max(buckets['Drawdown'])
    New_bucket = buckets.copy()
    New_bucket['Marginal_drawdown'] = np.nan
    New_bucket = New_bucket.iloc[1:, :]
    New_bucket = New_bucket.reset_index(drop = True)
    New_bucket['Marginal_drawdown'][0] = New_bucket['Drawdown'][0]
    for i in range(1, len(New_bucket)):
            cumulative_sum = New_bucket['Marginal_drawdown'].head(i).cumsum()
            cumulative_sum = cumulative_sum[i-1]    
        # Calculate the marginal drawdown for the current row
            a = New_bucket['Drawdown'][i] - cumulative_sum
            New_bucket['Marginal_drawdown'][i] = max((0,a))
    Maximum_Drawdown_for_entire_duration = max(New_bucket['Drawdown'])
    Undrawn_Amount_as_on_Start_Date = df['Undrawn_Amount'][0]
    Maximum_drawdown_percentage = Maximum_Drawdown_for_entire_duration/Undrawn_Amount_as_on_Start_Date
    Drawdown_bucket = pd.DataFrame()
    Drawdown_bucket['Time Bucket'] = New_bucket['Bucket']
    Drawdown_bucket['Max_Drawndown_Dist'] = (New_bucket['Marginal_drawdown']/Maximum_Drawdown_for_entire_duration)*100
    #Drawdown_bucket.to_excel('Maximum_Amount_Distribution.xlsx')     
    return Drawdown_bucket 

def simple_avg(df, column_name):
    df['sd1_day'] = np.nan
    for i in range (1, len(df)):
        df['sd1_day'][i] = ((df[column_name][i]-df[column_name][i-1])/df[column_name][i-1])
    df['sd2_7_day'] = np.nan
    for i in range (7, len(df)):
        if df.index[i]%7 == 0:
            df['sd2_7_day'][i] = ((df[column_name][i]-df[column_name][i-7])/df[column_name][i-7])
    df['sd8_14_day'] = np.nan
    for i in range (14, len(df)):
        if df.index[i]%14 == 0:
            df['sd8_14_day'][i] = ((df[column_name][i]-df[column_name][i-14])/df[column_name][i-14])
    return df    
            
def moving_avg(df, column_name):
    df['md1_day'] = np.nan
    for i in range (1, len(df)):
        df['md1_day'][i] = ((df[column_name][i]-df[column_name][i-1])/df[column_name][i-1])
    df['md2_7_day'] = np.nan
    for i in range (7, len(df)):
            df['md2_7_day'][i] = ((df[column_name][i]-df[column_name][i-7])/df[column_name][i-7])
    df['md8_14_day'] = np.nan
    for i in range (14, len(df)):
            df['md8_14_day'][i] = ((df[column_name][i]-df[column_name][i-14])/df[column_name][i-14])
    return df

def percentile(df, column_name):
    df['pd1_day'] = np.nan
    for i in range (1, len(df)):
        df['pd1_day'][i] = ((df[column_name][i]-df[column_name][i-1])/df[column_name][i-1])
    df['pd2_7_day'] = np.nan
    for i in range (7, len(df)):
            df['pd2_7_day'][i] = ((df[column_name][i]-df[column_name][i-7])/df[column_name][i-7])
    df['pd8_14_day'] = np.nan
    for i in range (14, len(df)):
            df['pd8_14_day'][i] = ((df[column_name][i]-df[column_name][i-14])/df[column_name][i-14])
    return df    

def moving_avg_incremental_bucket(df, column_name):
    df['mid1_day'] = np.nan
    for i in range (1, len(df)):
        df['mid1_day'][i] = ((df[column_name][i]-df[column_name][i-1])/df[column_name][i-1])
    df['mid2_7_day'] = np.nan
    for i in range (7, len(df)):
            df['mid2_7_day'][i] = ((df[column_name][i]-df[column_name][i-6])/df[column_name][i-6])
    df['mid8_14_day'] = np.nan
    for i in range (14, len(df)):
            df['mid8_14_day'][i] = ((df[column_name][i]-df[column_name][i-7])/df[column_name][i-7])
    df.to_excel('Returns.xlsx', sheet_name = 'Distribution')        
    return df

def Average_decrease(df1):
        cols = list(df1.columns)
        decrease = []
        variable = []
        for i in range (2, len(df1.columns)):
            variable.append(cols[i])
            a = df1[df1[cols[i]].lt(0)][cols[i]].mean()
            decrease.append(a)
        Distribution_zip = list(zip(variable, decrease)) ## mapping the data
        Dist_df = pd.DataFrame(Distribution_zip,columns = ['Time Period', 'Average Decrease'])
        Dist_df['Average Decrease'][6] = df1.pd1_day.quantile(0.75)
        Dist_df['Average Decrease'][7] = df1.pd2_7_day.quantile(0.75)
        Dist_df['Average Decrease'][8] = df1.pd8_14_day.quantile(0.75)
        Dist_df.to_excel('Decrease_df.xlsx')       
        return Dist_df

if __name__ == '__main__': 
    arguments = sys.argv[1]                  ###  Input File Path
    ### Fetching parent directory
    path = os.getcwd()
    parent = os.path.dirname(path)    
    parser = argparse.ArgumentParser(description='Defining the model parameters')

    # Add the model name argument
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('--autolag', type=str, default= 'AIC',help='Method to use when automatically determining the lag length')
    parser.add_argument('--Forecast_values',type = int, default= 365,help='Specify the initialization method of data (estimated, Heuristic, etc)')
    parser.add_argument('--smoothing_level', type=float, default= 0.5,help='Method to use when automatically determining the lag length')
    parser.add_argument('--smoothing_trend', type=float, default= 0.5,help='Method to use when automatically determining the lag length')
    
    parser.add_argument('--seasonal_periods', type=int,help='What should be the seasonality of the data under consideration')
    parser.add_argument('--trend',type = str, default= 'additive',help='Specify the type of trend present in the data additive/multiplicative')
    parser.add_argument('--seasonal',type = str, default= 'additive',help='Specify the type of seasonality present in the data additive/multiplicative')
    parser.add_argument('--freq',type = str, default= 'D',help='Specify the frequency of data (daily:D, Weekly:W, Monthly:M, etc)')
    parser.add_argument('--damped_trend',type = bool, default= False,help='Should the trend be damped or not')
    parser.add_argument('--initialization_method',type = str, default= 'estimated',help='Specify the initialization method of data (estimated, Heuristic, etc)')
    args = parser.parse_args()
    
    df = data_load(path)    
    train_df,test_df,y_hat_avg,test_df_size = train_test_splitting(df)
    
    train_data_forecast = generate_forecast(args.model_name, train_df, **vars(args))
    Mape = Mape_score(test_df,train_data_forecast)
    forecasting = generate_forecast(args.model_name, df, **vars(args))
    
    drawn_df = drawn_amount_computed(forecasting)
    Dist_df = Distribution_df()
    df_new = Time_bucket(forecasting, Dist_df)
    df1 = simple_avg(drawn_df, 'Drawn_Amount')
    df2 = moving_avg(drawn_df, 'Drawn_Amount')
    df4 = percentile(drawn_df, 'Drawn_Amount')
    df3 = moving_avg_incremental_bucket(drawn_df, 'Drawn_Amount')
    Decrease_df = Average_decrease(df3)

    modelOutput = maximum_amount_cumulative(df_new)
    # Convert DataFrame to JSON
    json_output = modelOutput.to_json(orient='records')
    # Create a dictionary with the desired key
    # model_output = {"modelOutput": json_output}
    # Convert the dictionary to a JSON string
    # modelOutput = json.dumps(model_output, indent=4)    
    print('modelOutput: ', json_output)
    
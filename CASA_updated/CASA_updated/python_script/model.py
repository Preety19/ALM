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

# Load the data and set the date column as the index
def data_load(path):
    # pylint: disable=no-member
    data = pd.read_json('data.json')
    data['date'] = data['date'].astype('datetime64[ns]') # Converting date from object to datetime for further analysis
    data = (data.sort_values('date')).reset_index(drop = True)
    data = data.set_index('date')
    return data

# Divide the dataset into train and test
def train_test_splitting(data):
    test_size = np.ceil(len(data)/5)
    test_size = int(test_size)
    train_df = df[:-test_size] 
    test_df = df[-test_size:]  
    y_hat_avg = test_df.copy()
    return train_df, test_df, y_hat_avg, test_size

# Generate forecasts based on the selected model
def generate_forecast(model, df, **params):
    if model == 'simple':
        #Code for Simple Exponential Smoothing
        model1 = SimpleExpSmoothing(df['amount'])
        fit1 = model1.fit(smoothing_level=params['smoothing_level'])
        print("Running Simple Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['amount'] = fit1.forecast(values)
        return y_hat_avg
        #forecast = fit1.forecast(steps=params['forecast_steps'])
        
    elif model == 'double':
        #Code for Double Exponential Smoothing
        model1 = Holt(df['amount'])
        fit1 = model1.fit(smoothing_level=params['smoothing_level'], smoothing_trend=params['smoothing_trend'])
        print("Running Double Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['amount'] = fit1.forecast(values)       
        return y_hat_avg
        #forecast = fit.forecast(steps=params['forecast_steps'])        

    elif model == 'triple':
        # Code for Triple Exponential Smoothing
        model1 = ExponentialSmoothing(df['amount'], trend=params['trend'], seasonal=params['seasonal'], seasonal_periods=params['seasonal_periods'], freq = params['freq'], damped_trend = params['damped_trend'], initialization_method = params['initialization_method'])
        fit1 = model1.fit()
        print("Running Triple Exponential Smoothing")
        values = test_df_size
        y_hat_avg = pd.DataFrame()
        y_hat_avg['amount'] = fit1.forecast(values)        
        return y_hat_avg
        #forecast = fit.forecast(steps=params['forecast_steps'])
        
    else:
        print("Invalid selection")
        return None

# Calculate the Mean Absolute Percentage Error (MAPE) score    
def Mape_score(test_df,y_hat_avg):
    mape1 = mean_absolute_percentage_error(test_df['amount'],y_hat_avg['amount'])
    print (f'Mean Absolute Percentage error of this model is {mape1}')
    return mape1

# Compute various distribution-related metrics    
def final_computations(final_forecasted_df):
    final_forecasted_df['date'] = final_forecasted_df.index
    final_forecasted_df.reset_index(drop = True, inplace = True)
    # Define the new column order using column indices
    new_column_order = [1,0]

    final_forecasted_df = final_forecasted_df.iloc[:, new_column_order]

    final_forecasted_df['Lambda'] = 0.96 
    final_forecasted_df['diff_lambda'] = 1 - final_forecasted_df['Lambda']
    final_forecasted_df.reset_index(drop = True, inplace = True)
    final_forecasted_df['Power'] = final_forecasted_df.index
    final_forecasted_df['Weight'] = ((final_forecasted_df['diff_lambda'] * 
                                     final_forecasted_df['Lambda']**final_forecasted_df['Power'])).round(decimals = 4)
    
    final_forecasted_df['Squared_Ui_7d'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-6):
        final_forecasted_df['Squared_Ui_7d'][i] = ((np.log(final_forecasted_df['amount'][i]/final_forecasted_df['amount'][i+6]))**2).round(decimals = 8)    
    final_forecasted_df['Squared_Ui_7d'] = (final_forecasted_df['Squared_Ui_7d'].fillna(0)).round(decimals = 8)     
    final_forecasted_df['EWMA_7D'] = (final_forecasted_df['Squared_Ui_7d']*final_forecasted_df['Weight']).round(decimals = 8)
    
    ####8-30days EWMA
    final_forecasted_df['Squared_Ui_8_30d'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-29):
        final_forecasted_df['Squared_Ui_8_30d'][i] = ((np.log(final_forecasted_df['amount'][i + 7]/final_forecasted_df['amount'][i+29]))**2).round(decimals = 8)
    final_forecasted_df['Squared_Ui_8_30d'] = (final_forecasted_df['Squared_Ui_8_30d'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_8_30D'] = (final_forecasted_df['Squared_Ui_8_30d']*final_forecasted_df['Weight']).round(decimals = 8) 
    
    ####1m-3m EWMA
    final_forecasted_df['Squared_Ui_1_3m'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-89):
        final_forecasted_df['Squared_Ui_1_3m'][i] = ((np.log(final_forecasted_df['amount'][i+30]/final_forecasted_df['amount'][i+89]))**2).round(decimals = 8)
    final_forecasted_df['Squared_Ui_1_3m'] = (final_forecasted_df['Squared_Ui_1_3m'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_1_3m'] = (final_forecasted_df['Squared_Ui_1_3m']*final_forecasted_df['Weight']).round(decimals = 8) 

####3m-6m EWMA
    final_forecasted_df['Squared_Ui_3_6m'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-179):
        final_forecasted_df['Squared_Ui_3_6m'][i] = ((np.log(final_forecasted_df['amount'][i+90]/final_forecasted_df['amount'][i+179]))**2).round(decimals = 8)
    final_forecasted_df['Squared_Ui_3_6m'] = (final_forecasted_df['Squared_Ui_3_6m'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_3_6m'] = (final_forecasted_df['Squared_Ui_3_6m']*final_forecasted_df['Weight']).round(decimals = 8) 
    
####6m-9m EWMA
    final_forecasted_df['Squared_Ui_6_9m'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-269):
        final_forecasted_df['Squared_Ui_6_9m'][i] = ((np.log(final_forecasted_df['amount'][i+180]/final_forecasted_df['amount'][i+269]))**2).round(decimals = 8)

    final_forecasted_df['Squared_Ui_6_9m'] = (final_forecasted_df['Squared_Ui_6_9m'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_6_9m'] = (final_forecasted_df['Squared_Ui_6_9m']*final_forecasted_df['Weight']).round(decimals = 8) 

    ####9m-1y EWMA
    final_forecasted_df['Squared_Ui_9m_1y'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-359):
        final_forecasted_df['Squared_Ui_9m_1y'][i] = ((np.log(final_forecasted_df['amount'][i+270]/final_forecasted_df['amount'][i+359]))**2).round(decimals = 8)

    final_forecasted_df['Squared_Ui_9m_1y'] = (final_forecasted_df['Squared_Ui_9m_1y'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_9m_1y'] = (final_forecasted_df['Squared_Ui_9m_1y']*final_forecasted_df['Weight']).round(decimals = 8) 

    ####1y3y EWMA
    final_forecasted_df['Squared_Ui_1y_3y'] = pd.Series(dtype='float')
    for i in range(0,len(final_forecasted_df)-1079):
        final_forecasted_df['Squared_Ui_1y_3y'][i] = ((np.log(final_forecasted_df['amount'][i+360]/final_forecasted_df['amount'][i+1079]))**2).round(decimals = 8)
    final_forecasted_df['Squared_Ui_1y_3y'] = (final_forecasted_df['Squared_Ui_1y_3y'].fillna(0)).round(decimals = 8)        
    final_forecasted_df['EWMA_1y_3y'] = (final_forecasted_df['Squared_Ui_1y_3y']*final_forecasted_df['Weight']).round(decimals = 8)    
    final_forecasted_df.to_csv('Forecasting.csv')
    
    Distribution = []
    h = 0
    for i in range (7, len(final_forecasted_df.columns)):
        if i%2 != 0:
            a = (np.sqrt(sum(final_forecasted_df.iloc[:,i])))
            h = h + a
            Distribution.append(a)
        else:
            pass    
    m = (1 - (h))*0.5
    Distribution.append(m)
    Distribution.append(m)
    dist = (h + (2*m))*100
    
    Bucket_Code = ['RL 1-7D', 'RL 8-30D', 'RL 1m-3m', 'RL 3m-6m', 'RL 6m-9m', 'RL 9m-1Y', 'RL 1Y-3Y', 'RL 3Y-5Y', 'RL>5Y'] ## Buckets
    Description = ['RL 1-7D', 'RL 8-30D', 'RL 1m-3m', 'RL 3m-6m', 'RL 6m-9m', 'RL 9m-1Y', 'RL 1Y-3Y', 'RL 3Y-5Y', 'RL>5Y'] ## Distribution of the buckets
    start_value_days = [0, 8, 31, 91, 181, 271, 361, 1081, 1801]  ## Starting day of each bucket
    end_value_days = [7, 30, 90, 180, 270, 360, 1080,1800, 29970] ## Ending day of each bucket
    #Distribution = values ## Distribution of each bucket
    Distribution_zip = list(zip(Bucket_Code, Description, Distribution, start_value_days, end_value_days)) ## mapping the data
    Distribution_df = pd.DataFrame(Distribution_zip,columns = ['Bucket_Code', 'Description', 'Distribution','start_value_days', 'end_value_days']) ## Creating a table for the same
    Distribution_df.to_csv('Distribution_table.csv') ## Exporting the forecasted values
       
    return Distribution_df

# Main execution when the script is run
if __name__ == '__main__': 
    # Parse command line arguments
    arguments = sys.argv[0]
    path = os.getcwd()
    parent = os.path.dirname(path)
    parser = argparse.ArgumentParser(description='Defining the model parameters for CASA model')

    # Add the model name argument
    parser.add_argument('model_name', type=str, help='Name of the model')
    #parser.add_argument('--autolag', type=str, default= 'AIC',help='Method to use when automatically determining the lag length')
    parser.add_argument('--Forecast_values',type = int, default= 365,help='Specify the number of values that should be further forecasted eg. 30')
    parser.add_argument('--smoothing_level', type=float, default= 0.5,help='Set the alpha value for the model i.e. the smoothening parameter. It ranges between 0-1')
    parser.add_argument('--smoothing_trend', type=float, default= 0.5,help='Set the beta value for the model i.e. the trend parameter')
    
    parser.add_argument('--seasonal_periods', type=int,help='What should be the seasonality of the data under consideration eg.(7, 30, 365)')
    parser.add_argument('--trend',type = str, default= 'additive',help='Specify the type of trend present in the data additive/multiplicative')
    parser.add_argument('--seasonal',type = str, default= 'additive',help='Specify the type of seasonality present in the data additive/multiplicative')
    parser.add_argument('--freq',type = str, default= 'D',help='Specify the frequency of data (daily:D, Weekly:W, Monthly:M, etc)')
    parser.add_argument('--damped_trend',type = bool, default= False,help='Should the trend be damped or not')
    parser.add_argument('--initialization_method',type = str, default= 'estimated',help='Specify the initialization method of data (estimated, Heuristic, etc)')
    args = parser.parse_args()

    df = data_load(os.path.join(parent,"Datasets"))              
    train_df,test_df,y_hat_avg,test_df_size = train_test_splitting(df)
    
    train_data_forecast = generate_forecast(args.model_name, train_df, **vars(args))
    Mape = Mape_score(test_df,train_data_forecast)
    forecasting = generate_forecast(args.model_name, df, **vars(args))
    df_final_dist= final_computations(forecasting)
    print("modelOutput: ", df_final_dist.to_json(orient='records'))


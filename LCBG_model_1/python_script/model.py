import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import argparse

def data_load(path):
    data = pd.read_json(os.path.join(path, 'data1.json'))
    data['Issued_Amount'] = data['Issued_Amount'].str.replace(',', '').astype(float)
    data['Invoked_Devolved'] = data['Invoked_Devolved'].str.replace(',', '').astype(float)
    # Convert the 'date' column to a datetime data type if it's not already in the proper format
    data['Issued_Date'] = pd.to_datetime(data['Issued_Date'])
    # Create new columns for month and year to use for grouping
    data['month'] = data['Issued_Date'].dt.month
    data['year'] = data['Issued_Date'].dt.year
    # Group the data by month and year, and apply the desired aggregation function to 'value'
    # In this example, we'll calculate the sum of 'value' for each month
    result1 = data.groupby(['year', 'month'])['Issued_Amount'].sum().reset_index()
    result2 = data.groupby(['year', 'month'])['Invoked_Devolved'].sum().reset_index()
    result2.drop(['year', 'month'], axis = 1, inplace = True)
    result1 = pd.concat([result1, result2], axis = 1)
    result1['Date'] = pd.to_datetime(result1[['year', 'month']].assign(day=1),format="%d-%m-%Y")
    result1.drop(['year', 'month'], inplace=True, axis=1)
    result1['Percentage_invoked'] = (result1['Invoked_Devolved']/result1['Issued_Amount'])*100
    result1.drop(['Issued_Amount', 'Invoked_Devolved'], inplace=True, axis=1)    
    result1.set_index('Date', inplace=True)  # Set 'Date' as the index
    return result1

def fetch_records_within_date_range(data, start_date, end_date):
    # Convert start_date and end_date to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)    
    # Filter the DataFrame using the .loc method
    selected_records = data.loc[(data.index >= start_date) & (data.index <= end_date)]    
    return selected_records

def load_macro_data(path):
    # Load the CSV file into a pandas DataFrame
    mev_data = pd.read_json(os.path.join(path, 'data2.json'))
    mev_data['Date'] = pd.to_datetime(mev_data['Date'], format="%d-%b-%y")  # Convert the 'Date' column to datetime format
    mev_data.sort_values('Date', inplace=True)  # Sort the DataFrame by 'Date' in ascending order
    mev_data.reset_index(drop=True, inplace=True)  # Reset the index after sorting
    mev_data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return mev_data

def columnselector(mev_df, *args : str):
    my_args = locals()
    cols = list(my_args["args"])
    mev_df = mev_df[cols]
    return mev_df

def perform_pca(data):
    """
    Perform Principal Component Analysis (PCA) on selected columns of a DataFrame with 2 components.
    
    Parameters:
        data (pandas DataFrame): The input DataFrame with date as the index.
        cols (list): A list of column names to be used for PCA.
    
    Returns:
        pca_df (pandas DataFrame): DataFrame containing the PCA results with date as the index.
    """
    # Select the columns of interest
    #data_selected = data[cols]
    
    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_standardized)
    
    # Create a DataFrame to store the PCA results
    pca_df = pd.DataFrame(pca_result, columns=["C1", "C2"], index=data.index)
    
    return pca_df

def perform_tsne(data):
    """
    Perform t-distributed Stochastic Neighbor Embedding (t-SNE) on selected columns of a DataFrame with 2 components.
    
    Parameters:
        data (pandas DataFrame): The input DataFrame with date as the index.
        cols (list): A list of column names to be used for t-SNE.
    
    Returns:
        tsne_df (pandas DataFrame): DataFrame containing the t-SNE results with date as the index.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)   
    # Perform t-SNE with 2 components
    tsne = TSNE(n_components=2,  random_state=42, perplexity=14)
    tsne_result = tsne.fit_transform(data_standardized)   
    # Create a DataFrame to store the t-SNE results
    tsne_df = pd.DataFrame(tsne_result, columns=["C1", "C2"], index=data.index)    
    return tsne_df

def perform_isomap(data):
    """
    Perform Isometric Mapping (ISOMAP) on selected columns of a DataFrame with 2 components.
    
    Parameters:
        data (pandas DataFrame): The input DataFrame with date as the index.
        cols (list): A list of column names to be used for ISOMAP.
    
    Returns:
        isomap_df (pandas DataFrame): DataFrame containing the ISOMAP results with date as the index.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)    
    # Perform ISOMAP with 2 components
    isomap = Isomap(n_components=2)
    isomap_result = isomap.fit_transform(data_standardized)    
    # Create a DataFrame to store the ISOMAP results
    isomap_df = pd.DataFrame(isomap_result, columns=["C1", "C2"], index=data.index)    
    return isomap_df

def merge_dataframes_by_common_index(df1, df2):
    """
    Merge two dataframes based on the common index using an inner join and store the next 'n' observations from df2.

    Parameters:
        df1 (pandas DataFrame): First dataframe.
        df2 (pandas DataFrame): Second dataframe.
        n (int): Number of records to store from df2 after merging with df1.

    Returns:
        merged_df (pandas DataFrame): Merged dataframe with only the rows having matching indices in both dataframes.
        df2_next_n_records (pandas DataFrame): Next 'n' records from df2 after merging with df1.
    """
    # Merge the dataframes based on the common index using an inner join
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    # Get the index of the last row of df1
    last_row_index = df1.index[-1]
    # Get the index of the next 'n' records in df2
    n = round(len(df1) / 4)
    next_n_records_index = df2.index[df2.index > last_row_index][:n]
    # Select the next 'n' records from df2
    df2_next_n_records = df2.loc[next_n_records_index]
    return merged_df, df2_next_n_records

def train_test_split_and_fit_lrr(data, mev_df):
    # Split the data into features (X) and target variable (y)
    X = data.drop(columns=['Percentage_invoked'])
    y = data['Percentage_invoked']
    
    # Split the data into training and testing sets
    n = round(len(data) / 4)
    X_train = X.iloc[:n*3]  # Use 'iloc' to access rows by integer index
    y_train = y.iloc[:n*3]
    X_test = X.iloc[3*n:]
    y_test = y.iloc[3*n:]
    
    # Create and fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    
    pred_df = pd.DataFrame(y_pred, columns=['Predicted'])
    pred_df.index = y_test.index
    plt.rcParams["figure.figsize"] = [8,4]
    plt.plot(y, label='Target')
    plt.plot(pred_df['Predicted'], label='Predicted', c='red')
    plt.legend(loc='best')
    #plt.savefig('Actual_versus_Predicted_values.png')   
    #plt.show()
    
    # Make predictions for the forecasted data
    y_forecast = model.predict(mev_df)
    # Create a new DataFrame with the forecasted values and index from 'mev_df'
    forecast_result_df = pd.DataFrame({'Percentage_invoked': y_forecast}, index=mev_df.index)
    forecast_result_df['Percentage_invoked'] = forecast_result_df['Percentage_invoked'].apply(lambda x: max(0, x))
    plt.plot(y, label='Actual')
    plt.plot(forecast_result_df['Percentage_invoked'], label='Forecasted', c='red')
    plt.legend(loc='best')
    #plt.savefig('Forecast.png')   
    #plt.show()    
    # Return the trained model and evaluation metrics
    return mse, mae, mape, forecast_result_df

if __name__ == "__main__":
    arguments = sys.argv[0]                 ###  Input File Path
    ### Fetching parent directory
    path = os.getcwd()
    parent = os.path.dirname(path)
    df = data_load(path)
    
        # Create the argument parser
    parser = argparse.ArgumentParser(description='Dimension reduction using PCA, t-SNE, or ISOMAP.')
    # Add an argument for each column name with nargs='+' to accept multiple arguments
    parser.add_argument('--start_date', type=str, default= '2015-01-01',help='Starting date of the data')
    parser.add_argument('--end_date',type = str, default= '2020-12-01',help='Last date till which data should be considered')
    parser.add_argument('model', choices=['PCA', 't-SNE', 'ISOMAP'], help='Choose the dimension reduction model (PCA, t-SNE, or ISOMAP)')
    parser.add_argument("columns", metavar="column_name", type=str, nargs="+", help="Column names to select.")    
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    #selected_records = fetch_records_within_date_range(df, '2014-01-04', '2015-01-03')
    selected_records = fetch_records_within_date_range(df, args.start_date, args.end_date)   
    data_mev = load_macro_data(path)
    mv_df = columnselector(data_mev, *args.columns)
    
    # Perform dimension reduction based on the user's choice
    if args.model == 'PCA':
        result_df = perform_pca(mv_df)
    elif args.model == 't-SNE':
        result_df = perform_tsne(mv_df)
    elif args.model == 'ISOMAP':
        result_df = perform_isomap(mv_df)
    else:
        print("Invalid model choice. Please choose either PCA, t-SNE, or ISOMAP.")
        exit(1)
        
    # print(result_df)    
    merged_df, forecasting_input = merge_dataframes_by_common_index(selected_records, result_df)
    mse, mae, mape, forecast_result_df = train_test_split_and_fit_lrr(merged_df, forecasting_input)
    #print(forecast_result_df.reset_index().astype(str))
    forecast_result_df = forecast_result_df.reset_index()
    forecast_result_df['Date'] = forecast_result_df['Date'].astype(str)
    finalOutput = forecast_result_df.to_json(orient='records', double_precision=2)
    print('modelOutput: ',finalOutput)

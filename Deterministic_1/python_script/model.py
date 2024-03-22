import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import json

def midpoints():
    Mid_point = {"Overnight" : 0.028,
"Over Overnight and upto 1 month" : 0.0417,
"Over 1 month and upto 3 months" : 0.1667,
"Over 3 months and upto 6 months" : 0.375,
"Over 6 months and upto 9 months" : 0.625,
"Over 9 months and upto 1 year" : 0.875,
"Over 1 year and upto 1_5 year" : 1.25,
"Over 1_5 year and upto 2 years" : 1.75,
"Over 2 years and upto 3 years" : 2.5,
"Over 3 years and upto 4 years" : 3.5,
"Over 4 years and up to 5 years" : 4.5,
"Over 5 years and upto 6 years" : 5.5,
"Over 6 years and upto 7 years" : 6.5,
"Over 7 years and upto 8 years" : 7.5,
"Over 8 years and upto 9 years" : 8.5,
"Over 9 years and upto 10 years" : 9.5,
"Over 10 years and upto 15 years" : 12.5,
"Over 15 years and upto 20 years" : 17.5,
"Over 20 years" : 25}
    return Mid_point

def columns():
    columns = ['Overnight',
 'Over Overnight and upto 1 month',
 'Over 1 month and upto 3 months',
 'Over 3 months and upto 6 months',
 'Over 6 months and upto 9 months',
 'Over 9 months and upto 1 year',
 'Over 1 year and upto 1_5 year',
 'Over 1_5 year and upto 2 years',
 'Over 2 years and upto 3 years',
 'Over 3 years and upto 4 years',
 'Over 4 years and up to 5 years',
 'Over 5 years and upto 6 years',
 'Over 6 years and upto 7 years',
 'Over 7 years and upto 8 years',
 'Over 8 years and upto 9 years',
 'Over 9 years and upto 10 years',
 'Over 10 years and upto 15 years',
 'Over 15 years and upto 20 years',
 'Over 20 years',
 'Non-Sensitive',
 'Total',
 'Sensitive Total']
    return columns

def multiplier_dict():
    multiplier_dict = {"Interest Rate Shock - Short" : 300,
"Interest Rate Shock - Long" : 200,
"Steepner Multiplier for short" : -0.65,
"Steepner Multiplier for long" : 0.9,
"Flattener Multiplier for short" : 0.8,
"Flattener Multiplier for long" : -0.6}
    return multiplier_dict

def data_load(path):
    df = pd.read_json(os.path.join(path, 'data1.json'))
    df.rename(columns = {'Maturity Yrs' : 'Maturity', '2023-08-25 00:00:00' : 'Day1', '2023-08-28 00:00:00' : 'Day2'}, inplace = True)
    return df

def total_gap(path):
    assets_df = pd.read_json(os.path.join(path, 'data2.json'))
    forma_df = assets_df.style.format('{:.2f}')
    assets_df = assets_df.round(0)
    total_assets_row = assets_df[assets_df['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
    total_assets_row = total_assets_row.iloc[0]
    # Convert the last row to dictionary    
    total_assets_row_dict = total_assets_row.to_dict()
    asset_total_df = pd.DataFrame([total_assets_row_dict])
    liabilities_df = pd.read_json(os.path.join(path, 'data3.json'))
    forma_df = liabilities_df.style.format('{:.2f}')
    liabilities_df = liabilities_df.round(0)
    # Fetch the last row
    last_row_liabilities = liabilities_df[liabilities_df['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
    total_liabilities_row = last_row_liabilities.iloc[0]

    # Convert the last row to dictionary
    last_row_dict_liabilities = total_liabilities_row.to_dict()
    liabilities_total_df = pd.DataFrame([last_row_dict_liabilities])
    total_df = pd.concat([asset_total_df,liabilities_total_df], axis= 0)
    total_df.reset_index(drop=True, inplace = True)
    total_df2 = total_df.transpose()
    total_df2.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
    total_df2.drop(['Description'], inplace=True)
    total_df2['Total_GAP'] = total_df2['Assets'] - total_df2['Liabilities']
    total_df2 = total_df2.round(0)
    
    ############################################################### NII calculation ######################################
    assets_df_nii = assets_df.copy(deep=True)
    assets_df_nii['Sensitive Total'] = assets_df_nii.iloc[:, 1:19].sum(axis = 1)
    total_assets_index = assets_df_nii.index[assets_df_nii['Description'] == 'TOTAL ASSETS'][0]
    sum_values = assets_df_nii[assets_df_nii['Description'] != 'TOTAL ASSETS'].iloc[:, 1:].sum(axis=0)
    loan_advances_value = assets_df_nii.loc[assets_df_nii['Description'] == 'Loans & Advances (Interest Cashflow)', assets_df_nii.columns[1:]].values[0]
    investments_value = assets_df_nii.loc[assets_df_nii['Description'] == 'Investments (Interest Cashflow)', assets_df_nii.columns[1:]].values[0]
    sum_values = sum_values - loan_advances_value - investments_value
    assets_df_nii.loc[total_assets_index, assets_df_nii.columns[1:]] = sum_values.values
    
    filtered_df = assets_df_nii[assets_df_nii['Description'] != 'TOTAL ASSETS']
    sum_sensitive_total = filtered_df['Sensitive Total'].sum()
    assets_df_nii.loc[assets_df_nii['Description'] == 'TOTAL ASSETS', 'Sensitive Total'] = sum_sensitive_total
        
    liabilities_df_nii = liabilities_df.copy(deep=True)
    liabilities_df_nii['Sensitive Total'] = liabilities_df_nii.iloc[:, 1:19].sum(axis = 1)
    total_liabilities_index = liabilities_df_nii.index[liabilities_df_nii['Description'] == 'TOTAL LIABILITIES'][0]
    sum_values = liabilities_df_nii[liabilities_df_nii['Description'] != 'TOTAL LIABILITIES'].iloc[:, 1:].sum(axis=0)
    liabilities_df_nii.loc[total_liabilities_index, liabilities_df_nii.columns[1:]] = sum_values.values
    
    filtered_df = liabilities_df_nii[liabilities_df_nii['Description'] != 'TOTAL LIABILITIES']
    sum_sensitive_total = filtered_df['Sensitive Total'].sum()
    liabilities_df_nii.loc[liabilities_df_nii['Description'] == 'TOTAL LIABILITIES', 'Sensitive Total'] = sum_sensitive_total
    
    total_assets_row_nii = assets_df_nii[assets_df_nii['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
    total_assets_row_nii = total_assets_row_nii.iloc[0]
    # Convert the last row to dictionary    
    total_assets_row_dict_nii = total_assets_row_nii.to_dict()
    asset_total_df_nii = pd.DataFrame([total_assets_row_dict_nii])
    
    last_row_liabilities_nii = liabilities_df_nii[liabilities_df_nii['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
    total_liabilities_row_nii = last_row_liabilities_nii.iloc[0]
    # Convert the last row to dictionary
    last_row_dict_liabilities_nii = total_liabilities_row_nii.to_dict()
    liabilities_total_df_nii = pd.DataFrame([last_row_dict_liabilities_nii])
    
    total_df_nii = pd.concat([asset_total_df_nii,liabilities_total_df_nii], axis= 0)
    total_df_nii.reset_index(drop=True, inplace = True)
    total_df2_nii = total_df_nii.transpose()
    total_df2_nii.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
    total_df2_nii.drop(['Description'], inplace=True)
    total_df2_nii['Total_GAP'] = total_df2_nii['Assets'] - total_df2_nii['Liabilities']
    total_df2_nii = total_df2_nii.round(0)
    total_df2_nii_t = total_df2_nii.transpose()
    total_df2_nii_t = total_df2_nii_t.iloc[:, :6]

    parallel_up = 250
    parallel_down = -250
    parallel_up_nii = []
    parallel_down_nii = []
    for col in total_df2_nii_t.columns:
        multiplier_value = Mid_point[col]
        parallel_up_nii.append(total_df2_nii_t[col]['Total_GAP'] * (parallel_up/10000)* (1-multiplier_value))
        parallel_down_nii.append(total_df2_nii_t[col]['Total_GAP'] * (parallel_down/10000)* (1-multiplier_value))
    parallel_up_sum = sum(parallel_up_nii)
    parallel_down_sum = sum(parallel_down_nii)
    nii_values_list = [parallel_up_sum, parallel_down_sum,0,0,0,0]        
    return assets_df, liabilities_df, total_df2, nii_values_list, assets_df_nii, liabilities_df_nii,total_df2_nii


def nii_sensitive_total_calculation(assets_df_nii,liabilities_df_nii):
        df1 = pd.DataFrame(columns=column)
        df2 = df1.transpose()
        df2['Description'] = df2.index
        df2.reset_index(drop = True, inplace = True)
        mids = list(Mid_point.values())
        mids.extend([0,0,0])
        df2['Mid_point'] = mids
        zcyc = []
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
                        (Mid_point['Overnight'] - df['Maturity'][0])/df['Maturity'][1] - df['Maturity'][0]),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over Overnight and upto 1 month'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over 1 month and upto 3 months'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over 3 months and upto 6 months'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))

        a = zcyc.append(round(df['Day_2'][1]+((df['Day_2'][2]-df['Day_2'][1]) * 
            (Mid_point['Over 6 months and upto 9 months'] - df['Maturity'][1])/(df['Maturity'][2] - df['Maturity'][1])),2))
        a = zcyc.append(round(df['Day_2'][1]+((df['Day_2'][2]-df['Day_2'][1]) * 
            (Mid_point['Over 9 months and upto 1 year'] - df['Maturity'][1])/(df['Maturity'][2] - df['Maturity'][1])),2))

        a = zcyc.append(round(df['Day_2'][2]+((df['Day_2'][3]-df['Day_2'][2]) * 
            (Mid_point['Over 1 year and upto 1_5 year'] - df['Maturity'][2])/(df['Maturity'][3] - df['Maturity'][2])),2))

        a = zcyc.append(round(df['Day_2'][3]+((df['Day_2'][4]-df['Day_2'][3]) * 
            (Mid_point['Over 1_5 year and upto 2 years'] - df['Maturity'][3])/(df['Maturity'][4] - df['Maturity'][3])),2))
        mid_values = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25]
        for j in mid_values:
            a = df.loc[df['Maturity'] == j, 'Day_2'].values[0] if 0 in df['Maturity'].values else None
            zcyc.append(a)
        zcyc.extend([0,0,0])
        df2['ZCYC'] = zcyc    
        
        df_parallel_up_nii = df2.copy()
        
        assets_df_pv_parallel_up_nii = assets_df_nii.copy(deep = True)
        df2_t_parallel_up_nii = df_parallel_up_nii.transpose()
        # Set the first row as header
        df2_t_parallel_up_nii.columns = df2_t_parallel_up_nii.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_up_nii = df2_t_parallel_up_nii[1:]
        # Assuming df2_t is already defined
        ZCYC_nii = df2_t_parallel_up_nii.loc['ZCYC']
        mid_point = df2_t_parallel_up_nii.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_parallel_up_nii[col] = assets_df_nii[col] / ((1 + (ZCYC_nii[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_parallel_up_nii['Sensitive Total'] = assets_df_pv_parallel_up_nii.iloc[:, 1:19].sum(axis = 1)
        #print(assets_df_pv_parallel_up_nii)
        assets_df_pv_parallel_up_nii['Non-Sensitive'] = 0
        assets_df_pv_parallel_up_nii['Total'] = 0
        forma_df_parallel_up_nii = assets_df_pv_parallel_up_nii.style.format('{:.2f}')
        assets_df_pv_parallel_up_nii = assets_df_pv_parallel_up_nii.round(0)
        
        assets_df_pv_parallel_up_nii['Sensitive Total'] = assets_df_pv_parallel_up_nii.iloc[:, 1:19].sum(axis = 1)
        total_assets_index = assets_df_pv_parallel_up_nii.index[assets_df_pv_parallel_up_nii['Description'] == 'TOTAL ASSETS'][0]
        sum_values = assets_df_pv_parallel_up_nii[assets_df_pv_parallel_up_nii['Description'] != 'TOTAL ASSETS'].iloc[:, 1:].sum(axis=0)
        assets_df_pv_parallel_up_nii.loc[total_assets_index, assets_df_pv_parallel_up_nii.columns[1:]] = sum_values.values
    

        liabilities_df_pv_parallel_up_nii = liabilities_df_nii.copy(deep = True)
        df2_t_parallel_up_nii = df_parallel_up_nii.transpose()
        # Set the first row as header
        df2_t_parallel_up_nii.columns = df2_t_parallel_up_nii.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_up_nii = df2_t_parallel_up_nii[1:]
        # Assuming df2_t is already defined
        ZCYC_nii = df2_t_parallel_up_nii.loc['ZCYC']
        mid_point = df2_t_parallel_up_nii.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_parallel_up_nii[col] = liabilities_df_nii[col] / ((1 + (ZCYC_nii[f'{col}']/100)) ** mid_point[f'{col}'])
        liabilities_df_pv_parallel_up_nii['Sensitive Total'] = liabilities_df_pv_parallel_up_nii.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_parallel_up_nii['Non-Sensitive'] = 0
        liabilities_df_pv_parallel_up_nii['Total'] = 0
        forma_df_parallel_up_nii = liabilities_df_pv_parallel_up_nii.style.format('{:.2f}')
        liabilities_df_pv_parallel_up_nii = liabilities_df_pv_parallel_up_nii.round(0)    
        # Fetch the last row        
        last_row_assets_pv_parallel_up_nii = assets_df_pv_parallel_up_nii[assets_df_pv_parallel_up_nii['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_parallel_up_nii = last_row_assets_pv_parallel_up_nii.iloc[0]
       
        # Convert the last row to dictionary
        last_row_dict_assets_pv_parallel_up_nii = last_row_assets_pv_parallel_up_nii.to_dict()
        asset_total_df_pv_parallel_up_nii = pd.DataFrame([last_row_dict_assets_pv_parallel_up_nii])

        # Fetch the last row
        last_row_liabilities_pv_parallel_up_nii = liabilities_df_pv_parallel_up_nii[liabilities_df_pv_parallel_up_nii['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_parallel_up_nii = last_row_liabilities_pv_parallel_up_nii.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_parallel_up_nii = last_row_liabilities_pv_parallel_up_nii.to_dict()
    
        liabilities_total_df_pv_parallel_up_nii = pd.DataFrame([last_row_dict_liabilities_pv_parallel_up_nii])

        total_df_pv_parallel_up_nii = pd.concat([asset_total_df_pv_parallel_up_nii,liabilities_total_df_pv_parallel_up_nii], axis= 0)
        total_df_pv_parallel_up_nii.reset_index(drop=True, inplace = True)
        total_df2_pv_parallel_up_nii = total_df_pv_parallel_up_nii.transpose()
        total_df2_pv_parallel_up_nii.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_parallel_up_nii.drop(['Description'], inplace=True)
        total_df2_pv_parallel_up_nii['Total_GAP'] = total_df2_pv_parallel_up_nii['Assets'] - total_df2_pv_parallel_up_nii['Liabilities']
        total_df2_pv_parallel_up_nii = total_df2_pv_parallel_up_nii.round(0)
        #print(total_df2_pv_parallel_up_nii)
        normal_scenario_sensitive_total = total_df2_pv_parallel_up_nii['Total_GAP'][-1]
        return normal_scenario_sensitive_total

def final_total_difference(df, assets_df, liabilities_df, multiplier_dict):
        total_pv_difference_parallel_up = []
        total_pv_difference_parallel_down = []
        total_pv_difference_Short_rate_Shock_up = []
        total_pv_difference_Short_rate_Shock_down = []
        total_pv_difference_steepener = []
        total_pv_difference_flattener = []
    
        # Create an empty DataFrame
        df1 = pd.DataFrame(columns=column)
        df2 = df1.transpose()
        df2['Description'] = df2.index
        df2.reset_index(drop = True, inplace = True)
        mids = list(Mid_point.values())
        mids.extend([0,0,0])
        df2['Mid_point'] = mids
        zcyc = []
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
                        (Mid_point['Overnight'] - df['Maturity'][0])/df['Maturity'][1] - df['Maturity'][0]),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over Overnight and upto 1 month'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over 1 month and upto 3 months'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))
        a = zcyc.append(round(df['Day_2'][0]+((df['Day_2'][1]-df['Day_2'][0]) * 
            (Mid_point['Over 3 months and upto 6 months'] - df['Maturity'][0])/(df['Maturity'][1] - df['Maturity'][0])),2))

        a = zcyc.append(round(df['Day_2'][1]+((df['Day_2'][2]-df['Day_2'][1]) * 
            (Mid_point['Over 6 months and upto 9 months'] - df['Maturity'][1])/(df['Maturity'][2] - df['Maturity'][1])),2))
        a = zcyc.append(round(df['Day_2'][1]+((df['Day_2'][2]-df['Day_2'][1]) * 
            (Mid_point['Over 9 months and upto 1 year'] - df['Maturity'][1])/(df['Maturity'][2] - df['Maturity'][1])),2))

        a = zcyc.append(round(df['Day_2'][2]+((df['Day_2'][3]-df['Day_2'][2]) * 
            (Mid_point['Over 1 year and upto 1_5 year'] - df['Maturity'][2])/(df['Maturity'][3] - df['Maturity'][2])),2))

        a = zcyc.append(round(df['Day_2'][3]+((df['Day_2'][4]-df['Day_2'][3]) * 
            (Mid_point['Over 1_5 year and upto 2 years'] - df['Maturity'][3])/(df['Maturity'][4] - df['Maturity'][3])),2))
        mid_values = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25]
        for j in mid_values:
            a = df.loc[df['Maturity'] == j, 'Day_2'].values[0] if 0 in df['Maturity'].values else None
            zcyc.append(a)
        zcyc.extend([0,0,0])
        df2['ZCYC'] = zcyc    
        
        ################### Parallel up #######################################
        df_parallel_up = df2.copy()
        df_parallel_up['Parallel_Shock'] = 250
        df_parallel_up['Parallel_Shock_up'] = round(df_parallel_up['ZCYC'] + df_parallel_up['Parallel_Shock']/100, 2)
        
        assets_df_pv_parallel_up = assets_df.copy(deep = True)
        df2_t_parallel_up = df_parallel_up.transpose()
        # Set the first row as header
        df2_t_parallel_up.columns = df2_t_parallel_up.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_up = df2_t_parallel_up[1:]
        # Assuming df2_t is already defined
        parallel_shock_up = df2_t_parallel_up.loc['Parallel_Shock_up']
        mid_point = df2_t_parallel_up.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_parallel_up[col] = assets_df[col] / ((1 + (parallel_shock_up[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_parallel_up['Sensitive Total'] = assets_df_pv_parallel_up.iloc[:, 1:19].sum(axis = 1)
        #print(assets_df_pv_parallel_up)
        assets_df_pv_parallel_up['Non-Sensitive'] = 0
        assets_df_pv_parallel_up['Total'] = 0
        forma_df_parallel_up = assets_df_pv_parallel_up.style.format('{:.2f}')
        assets_df_pv_parallel_up = assets_df_pv_parallel_up.round(0)

        liabilities_df_pv_parallel_up = liabilities_df.copy(deep = True)
        df2_t_parallel_up = df_parallel_up.transpose()
        # Set the first row as header
        df2_t_parallel_up.columns = df2_t_parallel_up.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_up = df2_t_parallel_up[1:]
        # Assuming df2_t is already defined
        parallel_shock_up = df2_t_parallel_up.loc['Parallel_Shock_up']
        mid_point = df2_t_parallel_up.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_parallel_up[col] = liabilities_df[col] / ((1 + (parallel_shock_up[f'{col}']/100)) ** mid_point[f'{col}'])
        liabilities_df_pv_parallel_up['Sensitive Total'] = liabilities_df_pv_parallel_up.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_parallel_up['Non-Sensitive'] = 0
        liabilities_df_pv_parallel_up['Total'] = 0
        forma_df_parallel_up = liabilities_df_pv_parallel_up.style.format('{:.2f}')
        liabilities_df_pv_parallel_up = liabilities_df_pv_parallel_up.round(0)    
        # Fetch the last row        
        last_row_assets_pv_parallel_up = assets_df_pv_parallel_up[assets_df_pv_parallel_up['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_parallel_up = last_row_assets_pv_parallel_up.iloc[0]
       
        # Convert the last row to dictionary
        last_row_dict_assets_pv_parallel_up = last_row_assets_pv_parallel_up.to_dict()
        asset_total_df_pv_parallel_up = pd.DataFrame([last_row_dict_assets_pv_parallel_up])

        # Fetch the last row
        last_row_liabilities_pv_parallel_up = liabilities_df_pv_parallel_up[liabilities_df_pv_parallel_up['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_parallel_up = last_row_liabilities_pv_parallel_up.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_parallel_up = last_row_liabilities_pv_parallel_up.to_dict()
    
        
        #last_row_liabilities_pv_parallel_up = liabilities_df_pv_parallel_up.iloc[-1]
        # Convert the last row to dictionary
        #last_row_dict_liabilities_pv_parallel_up = last_row_liabilities_pv_parallel_up.to_dict()
        liabilities_total_df_pv_parallel_up = pd.DataFrame([last_row_dict_liabilities_pv_parallel_up])

        total_df_pv_parallel_up = pd.concat([asset_total_df_pv_parallel_up,liabilities_total_df_pv_parallel_up], axis= 0)
        total_df_pv_parallel_up.reset_index(drop=True, inplace = True)
        total_df2_pv_parallel_up = total_df_pv_parallel_up.transpose()
        total_df2_pv_parallel_up.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_parallel_up.drop(['Description'], inplace=True)
        total_df2_pv_parallel_up['Total_GAP'] = total_df2_pv_parallel_up['Assets'] - total_df2_pv_parallel_up['Liabilities']
        total_df2_pv_parallel_up = total_df2_pv_parallel_up.round(0)
        a = total_df2_pv_parallel_up['Total_GAP'][-1] 
        total_pv_difference_parallel_up.append(a)
        maximum_value_parallel_up = max(total_pv_difference_parallel_up)
        
        
        ################### Parallel down #######################################
        df_parallel_down = df2.copy()
        df_parallel_down['Parallel_Shock'] = -250
        df_parallel_down['Parallel_Shock_down'] = round(df_parallel_down['ZCYC'] + df_parallel_down['Parallel_Shock']/100, 2)
        
        assets_df_pv_parallel_down = assets_df.copy(deep = True)
        df2_t_parallel_down = df_parallel_down.transpose()
        # Set the first row as header
        df2_t_parallel_down.columns = df2_t_parallel_down.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_down = df2_t_parallel_down[1:]
        # Assuming df2_t is already defined
        parallel_shock_down = df2_t_parallel_down.loc['Parallel_Shock_down']
        mid_point = df2_t_parallel_down.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_parallel_down[col] = assets_df[col] / ((1 + (parallel_shock_down[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_parallel_down['Sensitive Total'] = assets_df_pv_parallel_down.iloc[:, 1:19].sum(axis = 1)
        assets_df_pv_parallel_down['Non-Sensitive'] = 0
        assets_df_pv_parallel_down['Total'] = 0
        forma_df_parallel_down = assets_df_pv_parallel_down.style.format('{:.2f}')
        assets_df_pv_parallel_down = assets_df_pv_parallel_down.round(0)

        liabilities_df_pv_parallel_down = liabilities_df.copy(deep = True)
        df2_t_parallel_down = df_parallel_down.transpose()
        # Set the first row as header
        df2_t_parallel_down.columns = df2_t_parallel_down.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_parallel_down = df2_t_parallel_down[1:]
        # Assuming df2_t is already defined
        parallel_shock_down = df2_t_parallel_down.loc['Parallel_Shock_down']
        mid_point = df2_t_parallel_down.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_parallel_down[col] = liabilities_df[col] / ((1 + (parallel_shock_down[f'{col}']/100)) ** mid_point[f'{col}'])

        liabilities_df_pv_parallel_down['Sensitive Total'] = liabilities_df_pv_parallel_down.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_parallel_down['Non-Sensitive'] = 0
        liabilities_df_pv_parallel_down['Total'] = 0
        forma_df_parallel_down = liabilities_df_pv_parallel_down.style.format('{:.2f}')
        liabilities_df_pv_parallel_down = liabilities_df_pv_parallel_down.round(0)    
        # Fetch the last row
        last_row_assets_pv_parallel_down = assets_df_pv_parallel_down[assets_df_pv_parallel_down['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_parallel_down = last_row_assets_pv_parallel_down.iloc[0]
       
        # Convert the last row to dictionary
        last_row_dict_assets_pv_parallel_down = last_row_assets_pv_parallel_down.to_dict()
        asset_total_df_pv_parallel_down = pd.DataFrame([last_row_dict_assets_pv_parallel_down])

        # Fetch the last row
        last_row_liabilities_pv_parallel_down = liabilities_df_pv_parallel_up[liabilities_df_pv_parallel_down['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_parallel_down = last_row_liabilities_pv_parallel_down.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_parallel_down = last_row_liabilities_pv_parallel_down.to_dict()
        liabilities_total_df_pv_parallel_down = pd.DataFrame([last_row_dict_liabilities_pv_parallel_down])

        total_df_pv_parallel_down = pd.concat([asset_total_df_pv_parallel_down,liabilities_total_df_pv_parallel_down], axis= 0)
        total_df_pv_parallel_down.reset_index(drop=True, inplace = True)
        total_df2_pv_parallel_down = total_df_pv_parallel_down.transpose()
        total_df2_pv_parallel_down.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_parallel_down.drop(['Description'], inplace=True)
        total_df2_pv_parallel_down['Total_GAP'] = total_df2_pv_parallel_down['Assets'] - total_df2_pv_parallel_down['Liabilities']
        total_df2_pv_parallel_down = total_df2_pv_parallel_down.round(0)

        a = total_df2_pv_parallel_down['Total_GAP'][-1] 
        total_pv_difference_parallel_down.append(a)
        maximum_value_parallel_down = max(total_pv_difference_parallel_down)
        
        
        
        ################### Short_rate_shock_up #######################################
        mid_points = [-0.028,-0.0417,-0.1667,-0.375,-0.625,-0.875,-1.25,-1.75,-2.5,-3.5,-4.5,-5.5,-6.5,-7.5,-8.5,-9.5,-12.5,-17.5,-25]
        df_short_rate_shock_up = df2.copy()
        #df_short_rate_shock_up['Parallel_Shock'] = -250
        short = []
        long = []
        for j in mid_points:
            a = round(np.exp(j/4),4)
            short.append(a)
            long.append(1-a)

        short = short + [0, 0, 0]    
        long = long + [0, 0, 0]

        short_rate_shock_up = []
        short_rate_shock_down = []
        for k,l in zip(short,long):
            short_rate_shock_up.append(k*multiplier_dict["Interest Rate Shock - Short"]) 
            short_rate_shock_down.append(l* multiplier_dict['Interest Rate Shock - Long'])
            
        steepener_shock = []
        flattener_shock = []
        for m, n in zip(short_rate_shock_up, short_rate_shock_down):
            steepener_shock.append((m*multiplier_dict['Steepner Multiplier for short']) + (n* multiplier_dict['Steepner Multiplier for long']))
            flattener_shock.append((m*multiplier_dict['Flattener Multiplier for short']) + (n* multiplier_dict['Flattener Multiplier for long']))    
        
        #print(flattener_shock)
        #print(steepener_shock)
        df_short_rate_shock_up['Short_rate_Shock_up'] = short_rate_shock_up
        df_short_rate_shock_up['Revised_ZCYC_Short_rate_Shock_up'] = round(df_short_rate_shock_up['ZCYC'] + df_short_rate_shock_up['Short_rate_Shock_up']/100, 2)
        
        assets_df_pv_Short_rate_Shock_up = assets_df.copy(deep = True)
        df2_t_Short_rate_Shock_up = df_short_rate_shock_up.transpose()
        # Set the first row as header
        df2_t_Short_rate_Shock_up.columns = df2_t_Short_rate_Shock_up.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_Short_rate_Shock_up = df2_t_Short_rate_Shock_up[1:]
        # Assuming df2_t is already defined
        Short_rate_Shock_up = df2_t_Short_rate_Shock_up.loc['Revised_ZCYC_Short_rate_Shock_up']
        mid_point = df2_t_Short_rate_Shock_up.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_Short_rate_Shock_up[col] = assets_df[col] / ((1 + (Short_rate_Shock_up[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_Short_rate_Shock_up['Sensitive Total'] = assets_df_pv_Short_rate_Shock_up.iloc[:, 1:19].sum(axis = 1)
        assets_df_pv_Short_rate_Shock_up['Non-Sensitive'] = 0
        assets_df_pv_Short_rate_Shock_up['Total'] = 0
        forma_df_Short_rate_Shock_up = assets_df_pv_Short_rate_Shock_up.style.format('{:.2f}')
        assets_df_pv_Short_rate_Shock_up = assets_df_pv_Short_rate_Shock_up.round(0)

        liabilities_df_pv_Short_rate_Shock_up = liabilities_df.copy(deep = True)
        df2_t_Short_rate_Shock_up = df_short_rate_shock_up.transpose()
        # Set the first row as header
        df2_t_Short_rate_Shock_up.columns = df2_t_Short_rate_Shock_up.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_Short_rate_Shock_up = df2_t_Short_rate_Shock_up[1:]
        # Assuming df2_t is already defined
        Short_rate_Shock_up = df2_t_Short_rate_Shock_up.loc['Short_rate_Shock_up']
        mid_point = df2_t_Short_rate_Shock_up.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_Short_rate_Shock_up[col] = liabilities_df[col] / ((1 + (Short_rate_Shock_up[f'{col}']/100)) ** mid_point[f'{col}'])

        liabilities_df_pv_Short_rate_Shock_up['Sensitive Total'] = liabilities_df_pv_Short_rate_Shock_up.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_Short_rate_Shock_up['Non-Sensitive'] = 0
        liabilities_df_pv_Short_rate_Shock_up['Total'] = 0
        forma_df_Short_rate_Shock_up = liabilities_df_pv_Short_rate_Shock_up.style.format('{:.2f}')
        liabilities_df_pv_Short_rate_Shock_up = liabilities_df_pv_Short_rate_Shock_up.round(0)    
        # Fetch the last row
        last_row_assets_pv_Short_rate_Shock_up = assets_df_pv_Short_rate_Shock_up[assets_df_pv_Short_rate_Shock_up['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_Short_rate_Shock_up = last_row_assets_pv_Short_rate_Shock_up.iloc[0]
       
        # Convert the last row to dictionary
        last_row_dict_assets_pv_Short_rate_Shock_up = last_row_assets_pv_Short_rate_Shock_up.to_dict()
        asset_total_df_pv_Short_rate_Shock_up = pd.DataFrame([last_row_dict_assets_pv_Short_rate_Shock_up])

        # Fetch the last row
        last_row_liabilities_pv_Short_rate_Shock_up = liabilities_df_pv_Short_rate_Shock_up[liabilities_df_pv_Short_rate_Shock_up['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_Short_rate_Shock_up = last_row_liabilities_pv_Short_rate_Shock_up.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_Short_rate_Shock_up = last_row_liabilities_pv_Short_rate_Shock_up.to_dict()
        liabilities_total_df_pv_Short_rate_Shock_up = pd.DataFrame([last_row_dict_liabilities_pv_Short_rate_Shock_up])

        total_df_pv_Short_rate_Shock_up = pd.concat([asset_total_df_pv_Short_rate_Shock_up,liabilities_total_df_pv_Short_rate_Shock_up], axis= 0)
        total_df_pv_Short_rate_Shock_up.reset_index(drop=True, inplace = True)
        total_df2_pv_Short_rate_Shock_up = total_df_pv_Short_rate_Shock_up.transpose()
        total_df2_pv_Short_rate_Shock_up.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_Short_rate_Shock_up.drop(['Description'], inplace=True)
        total_df2_pv_Short_rate_Shock_up['Total_GAP'] = total_df2_pv_Short_rate_Shock_up['Assets'] - total_df2_pv_Short_rate_Shock_up['Liabilities']
        total_df2_pv_Short_rate_Shock_up = total_df2_pv_Short_rate_Shock_up.round(0)

        a = total_df2_pv_Short_rate_Shock_up['Total_GAP'][-1] 
        total_pv_difference_Short_rate_Shock_up.append(a)
        maximum_value_Short_rate_Shock_up = max(total_pv_difference_Short_rate_Shock_up)
        
        
        
        
        ################### Short_rate_shock_down ####################################### 
        df_short_rate_shock_down = df2.copy()
        df_short_rate_shock_down['Short_rate_Shock_down'] = short_rate_shock_down
        df_short_rate_shock_down['Revised_ZCYC_Short_rate_Shock_down'] = round(df_short_rate_shock_down['ZCYC'] + df_short_rate_shock_down['Short_rate_Shock_down']/100, 2)
        
        assets_df_pv_Short_rate_Shock_down = assets_df.copy(deep = True)
        df2_t_Short_rate_Shock_down = df_short_rate_shock_down.transpose()
        # Set the first row as header
        df2_t_Short_rate_Shock_down.columns = df2_t_Short_rate_Shock_down.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_Short_rate_Shock_down = df2_t_Short_rate_Shock_down[1:]
        # Assuming df2_t is already defined
        Short_rate_Shock_down = df2_t_Short_rate_Shock_down.loc['Revised_ZCYC_Short_rate_Shock_down']
        mid_point = df2_t_Short_rate_Shock_down.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_Short_rate_Shock_down[col] = assets_df[col] / ((1 + (Short_rate_Shock_down[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_Short_rate_Shock_down['Sensitive Total'] = assets_df_pv_Short_rate_Shock_down.iloc[:, 1:19].sum(axis = 1)
        assets_df_pv_Short_rate_Shock_down['Non-Sensitive'] = 0
        assets_df_pv_Short_rate_Shock_down['Total'] = 0
        forma_df_Short_rate_Shock_down = assets_df_pv_Short_rate_Shock_down.style.format('{:.2f}')
        assets_df_pv_Short_rate_Shock_down = assets_df_pv_Short_rate_Shock_down.round(0)

        liabilities_df_pv_Short_rate_Shock_down = liabilities_df.copy(deep = True)
        df2_t_Short_rate_Shock_down = df_short_rate_shock_down.transpose()
        # Set the first row as header
        df2_t_Short_rate_Shock_down.columns = df2_t_Short_rate_Shock_down.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_Short_rate_Shock_down = df2_t_Short_rate_Shock_down[1:]
        # Assuming df2_t is already defined
        Short_rate_Shock_down = df2_t_Short_rate_Shock_down.loc['Short_rate_Shock_down']
        mid_point = df2_t_Short_rate_Shock_down.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_Short_rate_Shock_down[col] = liabilities_df[col] / ((1 + (Short_rate_Shock_down[f'{col}']/100)) ** mid_point[f'{col}'])

        liabilities_df_pv_Short_rate_Shock_down['Sensitive Total'] = liabilities_df_pv_Short_rate_Shock_down.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_Short_rate_Shock_down['Non-Sensitive'] = 0
        liabilities_df_pv_Short_rate_Shock_down['Total'] = 0
        forma_df_Short_rate_Shock_down = liabilities_df_pv_Short_rate_Shock_down.style.format('{:.2f}')
        liabilities_df_pv_Short_rate_Shock_down = liabilities_df_pv_Short_rate_Shock_down.round(0)    
        # Fetch the last row
        last_row_assets_pv_Short_rate_Shock_down = assets_df_pv_Short_rate_Shock_down[assets_df_pv_Short_rate_Shock_down['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_Short_rate_Shock_down = last_row_assets_pv_Short_rate_Shock_down.iloc[0]
       
        # Convert the last row to dictionary
        last_row_dict_assets_pv_Short_rate_Shock_down = last_row_assets_pv_Short_rate_Shock_down.to_dict()
        asset_total_df_pv_Short_rate_Shock_down = pd.DataFrame([last_row_dict_assets_pv_Short_rate_Shock_down])

        # Fetch the last row
        last_row_liabilities_pv_Short_rate_Shock_down = liabilities_df_pv_Short_rate_Shock_down[liabilities_df_pv_Short_rate_Shock_down['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_Short_rate_Shock_down = last_row_liabilities_pv_Short_rate_Shock_down.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_Short_rate_Shock_down = last_row_liabilities_pv_Short_rate_Shock_down.to_dict()
        liabilities_total_df_pv_Short_rate_Shock_down = pd.DataFrame([last_row_dict_liabilities_pv_Short_rate_Shock_down])
        
        total_df_pv_Short_rate_Shock_down = pd.concat([asset_total_df_pv_Short_rate_Shock_down,liabilities_total_df_pv_Short_rate_Shock_down], axis= 0)
        total_df_pv_Short_rate_Shock_down.reset_index(drop=True, inplace = True)
        total_df2_pv_Short_rate_Shock_down = total_df_pv_Short_rate_Shock_down.transpose()
        total_df2_pv_Short_rate_Shock_down.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_Short_rate_Shock_down.drop(['Description'], inplace=True)
        total_df2_pv_Short_rate_Shock_down['Total_GAP'] = total_df2_pv_Short_rate_Shock_down['Assets'] - total_df2_pv_Short_rate_Shock_down['Liabilities']
        total_df2_pv_Short_rate_Shock_down = total_df2_pv_Short_rate_Shock_down.round(0)

        a = total_df2_pv_Short_rate_Shock_down['Total_GAP'][-1] 
        total_pv_difference_Short_rate_Shock_down.append(a)
        maximum_value_Short_rate_Shock_down = max(total_pv_difference_Short_rate_Shock_down)
              
        
        
        
        ################### Steepener #######################################         
        df_steepener = df2.copy()
        df_steepener['steepener_shock'] = steepener_shock
        df_steepener['Revised_ZCYC_steepener'] = round(df_steepener['ZCYC'] + df_steepener['steepener_shock']/100, 2)
        
        assets_df_pv_steepener = assets_df.copy(deep = True)
        df2_t_steepener = df_steepener.transpose()
        # Set the first row as header
        df2_t_steepener.columns = df2_t_steepener.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_steepener = df2_t_steepener[1:]
        # Assuming df2_t is already defined
        steepener = df2_t_steepener.loc['Revised_ZCYC_steepener']
        mid_point = df2_t_steepener.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_steepener[col] = assets_df[col] / ((1 + (steepener[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_steepener['Sensitive Total'] = assets_df_pv_steepener.iloc[:, 1:19].sum(axis = 1)
        assets_df_pv_steepener['Non-Sensitive'] = 0
        assets_df_pv_steepener['Total'] = 0
        forma_df_steepener = assets_df_pv_steepener.style.format('{:.2f}')
        assets_df_pv_steepener = assets_df_pv_steepener.round(0)

        liabilities_df_pv_steepener = liabilities_df.copy(deep = True)
        df2_t_steepener = df_steepener.transpose()
        # Set the first row as header
        df2_t_steepener.columns = df2_t_steepener.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_steepener = df2_t_steepener[1:]
        # Assuming df2_t is already defined
        steepener = df2_t_steepener.loc['Revised_ZCYC_steepener']
        mid_point = df2_t_steepener.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_steepener[col] = liabilities_df[col] / ((1 + (steepener[f'{col}']/100)) ** mid_point[f'{col}'])

        liabilities_df_pv_steepener['Sensitive Total'] = liabilities_df_pv_steepener.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_steepener['Non-Sensitive'] = 0
        liabilities_df_pv_steepener['Total'] = 0
        forma_df_steepener = liabilities_df_pv_steepener.style.format('{:.2f}')
        liabilities_df_pv_steepener = liabilities_df_pv_steepener.round(0)    
        # Fetch the last row
        last_row_assets_pv_steepener = assets_df_pv_steepener[assets_df_pv_steepener['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_steepener = last_row_assets_pv_steepener.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_assets_pv_steepener = last_row_assets_pv_steepener.to_dict()
    
        
        #last_row_assets_pv_steepener = assets_df_pv_steepener.iloc[-1]
        # Convert the last row to dictionary
        #last_row_dict_assets_pv_steepener = last_row_assets_pv_steepener.to_dict()
        asset_total_df_pv_steepener = pd.DataFrame([last_row_dict_assets_pv_steepener])

        # Fetch the last row
        last_row_liabilities_pv_steepener = liabilities_df_pv_steepener[liabilities_df_pv_steepener['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_steepener = last_row_liabilities_pv_steepener.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_steepener = last_row_liabilities_pv_steepener.to_dict()
    
        
        #last_row_liabilities_pv_steepener = liabilities_df_pv_steepener.iloc[-1]
        # Convert the last row to dictionary
        #last_row_dict_liabilities_pv_steepener = last_row_liabilities_pv_steepener.to_dict()
        liabilities_total_df_pv_steepener = pd.DataFrame([last_row_dict_liabilities_pv_steepener])

        total_df_pv_steepener = pd.concat([asset_total_df_pv_steepener,liabilities_total_df_pv_steepener], axis= 0)
        total_df_pv_steepener.reset_index(drop=True, inplace = True)
        total_df2_pv_steepener = total_df_pv_steepener.transpose()
        total_df2_pv_steepener.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_steepener.drop(['Description'], inplace=True)
        total_df2_pv_steepener['Total_GAP'] = total_df2_pv_steepener['Assets'] - total_df2_pv_steepener['Liabilities']
        total_df2_pv_steepener = total_df2_pv_steepener.round(0)

        a = total_df2_pv_steepener['Total_GAP'][-1] 
        total_pv_difference_steepener.append(a)
        maximum_value_steepener = max(total_pv_difference_steepener)
        
        
        
        ##################################################### Flattener ########################################################         
        df_flattener = df2.copy()
        df_flattener['flattener_shock'] = flattener_shock
        df_flattener['Revised_ZCYC_flattener'] = round(df_flattener['ZCYC'] + df_flattener['flattener_shock']/100, 2)
        
        assets_df_pv_flattener = assets_df.copy(deep = True)
        df2_t_flattener = df_flattener.transpose()
        # Set the first row as header
        df2_t_flattener.columns = df2_t_flattener.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_flattener = df2_t_flattener[1:]
        # Assuming df2_t is already defined
        flattener = df2_t_flattener.loc['Revised_ZCYC_flattener']
        mid_point = df2_t_flattener.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            assets_df_pv_flattener[col] = assets_df[col] / ((1 + (flattener[f'{col}']/100)) ** mid_point[f'{col}'])

        assets_df_pv_flattener['Sensitive Total'] = assets_df_pv_flattener.iloc[:, 1:19].sum(axis = 1)
        assets_df_pv_flattener['Non-Sensitive'] = 0
        assets_df_pv_flattener['Total'] = 0
        forma_df_flattener = assets_df_pv_flattener.style.format('{:.2f}')
        assets_df_pv_flattener = assets_df_pv_flattener.round(0)

        liabilities_df_pv_flattener = liabilities_df.copy(deep = True)
        df2_t_flattener = df_flattener.transpose()
        # Set the first row as header
        df2_t_flattener.columns = df2_t_flattener.iloc[0]
        # Drop the first row (since it's now the header)
        df2_t_flattener = df2_t_flattener[1:]
        # Assuming df2_t is already defined
        flattener = df2_t_flattener.loc['Revised_ZCYC_flattener']
        mid_point = df2_t_flattener.loc['Mid_point']
        # Loop through the list of columns
        for col in column:
            liabilities_df_pv_flattener[col] = liabilities_df[col] / ((1 + (flattener[f'{col}']/100)) ** mid_point[f'{col}'])

        liabilities_df_pv_flattener['Sensitive Total'] = liabilities_df_pv_flattener.iloc[:, 1:19].sum(axis = 1)
        liabilities_df_pv_flattener['Non-Sensitive'] = 0
        liabilities_df_pv_flattener['Total'] = 0
        forma_df_flattener = liabilities_df_pv_flattener.style.format('{:.2f}')
        liabilities_df_pv_flattener = liabilities_df_pv_flattener.round(0)    
        # Fetch the last row
        last_row_assets_pv_flattener = assets_df_pv_flattener[assets_df_pv_flattener['Description'] == 'TOTAL ASSETS'].reset_index(drop=True)
        last_row_assets_pv_flattener = last_row_assets_pv_flattener.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_assets_pv_flattener = last_row_assets_pv_flattener.to_dict()
    
        #last_row_assets_pv_flattener = assets_df_pv_flattener.iloc[-1]
        # Convert the last row to dictionary
        #last_row_dict_assets_pv_flattener = last_row_assets_pv_flattener.to_dict()
        asset_total_df_pv_flattener = pd.DataFrame([last_row_dict_assets_pv_flattener])

        # Fetch the last row
        last_row_liabilities_pv_flattener = liabilities_df_pv_flattener[liabilities_df_pv_flattener['Description'] == 'TOTAL LIABILITIES'].reset_index(drop=True)
        last_row_liabilities_pv_flattener = last_row_liabilities_pv_flattener.iloc[0]
        # Convert the last row to dictionary
        last_row_dict_liabilities_pv_flattener = last_row_liabilities_pv_flattener.to_dict()
    
        #last_row_liabilities_pv_flattener = liabilities_df_pv_flattener.iloc[-1]
        # Convert the last row to dictionary
        #last_row_dict_liabilities_pv_flattener = last_row_liabilities_pv_flattener.to_dict()
        liabilities_total_df_pv_flattener = pd.DataFrame([last_row_dict_liabilities_pv_flattener])

        total_df_pv_flattener = pd.concat([asset_total_df_pv_flattener,liabilities_total_df_pv_flattener], axis= 0)
        total_df_pv_flattener.reset_index(drop=True, inplace = True)
        total_df2_pv_flattener = total_df_pv_flattener.transpose()
        total_df2_pv_flattener.rename(columns={0: 'Assets', 1: 'Liabilities'}, inplace=True)
        total_df2_pv_flattener.drop(['Description'], inplace=True)
        total_df2_pv_flattener['Total_GAP'] = total_df2_pv_flattener['Assets'] - total_df2_pv_flattener['Liabilities']
        total_df2_pv_flattener = total_df2_pv_flattener.round(0)

        a = total_df2_pv_flattener['Total_GAP'][-1] 
        total_pv_difference_flattener.append(a)
        maximum_value_flattener = max(total_pv_difference_flattener)

        #print(i)
        return maximum_value_parallel_up, maximum_value_parallel_down, maximum_value_Short_rate_Shock_up, maximum_value_Short_rate_Shock_down, maximum_value_steepener, maximum_value_flattener   

if __name__ == '__main__': 
    arguments = sys.argv[0]
    path = os.getcwd()
    parent = os.path.dirname(path)
    
    column = columns()
    Mid_point = midpoints()
    multiplier_dict = multiplier_dict()
    df = data_load(path)
    assets_df, liabilities_df, total_df2, nii_values_list, assets_df_nii, liabilities_df_nii,total_df2_nii = total_gap(path)
    maximum_value_parallel_up, maximum_value_parallel_down, maximum_value_Short_rate_Shock_up, maximum_value_Short_rate_Shock_down, maximum_value_steepener, maximum_value_flattener = final_total_difference(df,assets_df, liabilities_df, multiplier_dict)
    naa = nii_sensitive_total_calculation(assets_df_nii,liabilities_df_nii)
    shocks = ["Parallel up", "Parallel down", "ShortRate Shock up", "ShortRate Shock down", "Steepener shock", "Flattener shock"]
    max_loss_values = [maximum_value_parallel_up, maximum_value_parallel_down, maximum_value_Short_rate_Shock_up, maximum_value_Short_rate_Shock_down, maximum_value_steepener, maximum_value_flattener]
    
    print(naa)
    result = [naa - value for value in max_loss_values]
    
    final_df = df = pd.DataFrame({'Period': shocks,'GAP': result,'NII': nii_values_list})
    #final_df['GAP'] = naa - final_df['GAP']
    # Convert DataFrame to JSON
    json_output = final_df.to_json(orient='records')
    # Create a dictionary with the desired key
    # model_output = {"modelOutput": json_output}
    # # Convert the dictionary to a JSON string
    # modelOutput = json.dumps(model_output, indent=4)    
    print('modelOutput: ',json_output)
    

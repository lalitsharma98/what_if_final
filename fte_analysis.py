import pandas as pd
import numpy as np
import streamlit as st
import re

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def calculate_average_q2_q4_time(df, staffing_change, demand_change, occupancy_assumption):
    # Adjust Staffing by the given amount using the mean of Staffing (ACTUAL)
    df['Adjusted_Staffing'] = df['Staffing'].mean() + staffing_change
    
    # Adjust Demand by the given amount using the mean of Demand multiplied by (1 + demand_change)
    df['Adjusted_Demand'] = df['Demand'].mean() * (1 + demand_change)    
    
    # Change occupancy assumption to the given value
    df['Adjusted_Occupancy_Assumption'] = occupancy_assumption
    
    # Filter entries with Staffing within ±10% and Demand within ±10% and occupancy assumption within ±10%
    staffing_lower_bound = df['Adjusted_Staffing'] * 0.9
    staffing_upper_bound = df['Adjusted_Staffing'] * 1.1
    demand_lower_bound = df['Adjusted_Demand'] * 0.9
    demand_upper_bound = df['Adjusted_Demand'] * 1.1    
    occupancy_lower_bound = df['Adjusted_Occupancy_Assumption'] * 0.9
    occupancy_upper_bound = df['Adjusted_Occupancy_Assumption'] * 1.1
    
    filtered_df = df[(df['Staffing'] >= staffing_lower_bound) & (df['Staffing'] <= staffing_upper_bound) & 
        (df['Demand'] >= demand_lower_bound) & (df['Demand'] <= demand_upper_bound) &
        (df['Occupancy Assumptions'] >= occupancy_lower_bound) & (df['Occupancy Assumptions'] <= occupancy_upper_bound)]
    
    # Check if "Q4" column is present
    if 'Q4' in df.columns:
        mean_value = filtered_df['Q4'].mean()
    else:
        mean_value = filtered_df['Q2'].mean()
    
    return mean_value

def calculate_average_occupancy_rate(df, staffing_change, demand_change, occupancy_assumption):
    # Adjust Staffing by the given amount using the mean of Staffing
    df['Adjusted_Staffing'] = df['Staffing'].mean() + staffing_change
    
    # Adjust Demand by the given amount using the mean of Demand multiplied by (1 + demand_change)
    df['Adjusted_Demand'] = df['Demand'].mean() * (1 + demand_change)     
    
    # Change occupancy assumption to the given value
    df['Adjusted_Occupancy_Assumption'] = occupancy_assumption
    
    # Filter entries with Staffing within ±10% and Demand within ±10% and occupancy assumption within ±10%
    staffing_lower_bound = df['Adjusted_Staffing'] * 0.9
    staffing_upper_bound = df['Adjusted_Staffing'] * 1.1
    demand_lower_bound = df['Adjusted_Demand'] * 0.9
    demand_upper_bound = df['Adjusted_Demand'] * 1.1    
    occupancy_lower_bound = df['Adjusted_Occupancy_Assumption'] * 0.9
    occupancy_upper_bound = df['Adjusted_Occupancy_Assumption'] * 1.1
    
    filtered_df = df[(df['Staffing'] >= staffing_lower_bound) & (df['Staffing'] <= staffing_upper_bound) & 
        (df['Demand'] >= demand_lower_bound) & (df['Demand'] <= demand_upper_bound) &
        (df['Occupancy Assumptions'] >= occupancy_lower_bound) & (df['Occupancy Assumptions'] <= occupancy_upper_bound)]
    
    # Calculate average Occupancy Rate for the filtered entries
    average_occupancy_rate = filtered_df['Occupancy Rate'].mean()
    
    return average_occupancy_rate

def group_data(df, min_value, max_value):
    
    # Calculate the range and determine the bin width
    range_value = max_value - min_value
    bin_width = max(10, range_value // 10)
    # Create bins with a minimum difference of 10 and a maximum of 10 bins
    bins = np.arange(min_value, max_value + bin_width, bin_width)

    # Check if "Q4" column is present
    if 'Q4' in df.columns:
        q_column = 'Q4'
    else:
        q_column = 'Q2'
    
    # Ensure columns are numeric
    df['Occupancy Rate'] = pd.to_numeric(df['Occupancy Rate'], errors='coerce')
    df['Occupancy Assumptions'] = pd.to_numeric(df['Occupancy Assumptions'], errors='coerce')
    df[q_column] = pd.to_numeric(df[q_column], errors='coerce')        
    
    df['Requirement_Range'] = pd.cut(df['Staffing'], bins=bins)

    df['Requirement_Range'] = df['Requirement_Range'].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
    
    grouped_df = df.groupby('Requirement_Range').agg({
        'Demand': 'mean',        
        'Occupancy Rate': ['mean', 'count'],
        'Occupancy Assumptions': 'mean',
        q_column : 'mean'
    }).reset_index()
    grouped_df.columns = ['Requirement_Range', 'Demand', 'Occupancy Rate', 'Frequency', 'Occupancy Assumptions', 'Q2_Q4 Time']
    
    
    
    # Round all numeric columns to 2 decimal places
    grouped_df['Demand'] = grouped_df['Demand'].fillna(0).astype(int)
    grouped_df['Occupancy Rate'] = grouped_df['Occupancy Rate'].round(2)
    grouped_df['Occupancy Assumptions'] = grouped_df['Occupancy Assumptions'].round(2)
    grouped_df['Q2_Q4 Time'] = grouped_df['Q2_Q4 Time'].round(2)

    return grouped_df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Handling NaN and infinite values for derived variables
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df

def run_fte_analysis():
    st.title("Staffing Prediction with Variable Analysis")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        
        df = load_data(uploaded_file)
        st.write("Data loaded successfully!")
        
        # Extract the text with "Volume (ACTUAL)" columns
        volume_columns = [col for col in df.columns if "Occupancy Rate" in col]
        for col in volume_columns:
            text_between = col.split("Occupancy Rate")[0].strip()
               
        if text_between.upper().startswith(('OLY','BOA')):            
            selected_option_part = text_between.split(' ', 0)
            selected_columns = [col for col in df.columns if selected_option_part[0] in col]
            column_mapping = {
                'Week Of:': 'Date',
                f'Combined {selected_option_part[0]} Volume (ACTUAL)': 'Calls',
                f'{text_between} STAFFING DIFFERENTIAL': 'Staffing Diff',
                f'Combined {selected_option_part[0]} AHT (ACTUAL)': 'AHT',
                f'{text_between} Occupancy Rate': 'Occupancy Rate',
                f'{text_between} Staffing Requirement (ACTUAL)': 'FTE Requirement',
                f'{text_between} Staffing (ACTUAL/FORECASTED)': 'Staffing',
                f'{text_between} Q4 Time': 'Q4',
                f'OCC Assumptions ({selected_option_part[0]})': 'Occupancy Assumptions'
            }
        else:
            file_name = uploaded_file.name    
            if 'CSA' in file_name.upper():
                selected_option_part = text_between.split(' ', 1)
                selected_columns = [col for col in df.columns if selected_option_part[0] in col]
                
                column_mapping = {
                    'Week Of:': 'Date',
                    f'Combined {selected_option_part[0]} Volume (ACTUAL)': 'Calls',
                    f'{text_between} STAFFING DIFFERENTIAL': 'Staffing Diff',
                    f'Combined {selected_option_part[0]} AHT (ACTUAL)': 'AHT',
                    f'{text_between} Occupancy Rate': 'Occupancy Rate',
                    f'{text_between} Staffing Requirement (ACTUAL)': 'FTE Requirement',
                    f'{text_between} Staffing (ACTUAL/FORECASTED)': 'Staffing',
                    f'{text_between} Q4 Time': 'Q4',
                    f'{text_between} OCC Assumptions ({selected_option_part[1]}):': 'Occupancy Assumptions'
                }
            else:
                
                selected_option_part = text_between.split(' ', 1)
                selected_columns = [col for col in df.columns if selected_option_part[1] in col]
                column_mapping = {
                    'Week Of:': 'Date',
                    f'{text_between} Volume (ACTUAL)': 'Calls',
                    f'{text_between} STAFFING DIFFERENTIAL': 'Staffing Diff',
                    f'{text_between} AHT (ACTUAL)': 'AHT',
                    f'{text_between} Occupancy Rate': 'Occupancy Rate',
                    f'{text_between} Staffing Requirement (ACTUAL)': 'FTE Requirement',
                    f'{text_between} Staffing (ACTUAL/FORECASTED)': 'Staffing',
                    f'{text_between} Q2 Time': 'Q2',
                    f'{selected_option_part[0]} OCC Assumptions ({selected_option_part[1]}):': 'Occupancy Assumptions'
                }          
        
        # Renaming the columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Check for duplicate columns and handle them
        df = df.loc[:, ~df.columns.duplicated()]

        
        df = df[df['Staffing'] != 0]
        df = df[df['Staffing'] != '']
        
        df = df[df['Occupancy Rate'] != 0]
        df = df[df['Occupancy Rate'] != '']
        df = df[df['Occupancy Rate'] != 'None']
        df = df[df['Occupancy Rate'] != '-']
        
        df['Demand'] = df['Calls'] * df['AHT']

        df = preprocess_data(df)
        
        # User inputs for Staffing change and occupancy assumption
        staffing_change_1 = st.number_input("Enter the change in Staffing for scenario 1:", value=0)
        demand_change_1 = st.number_input("Enter the change in demand for scenario 1 (%):", value=10) / 100       
        occupancy_assumption_1 = st.number_input("Enter the occupancy assumption for scenario 1 (as a percentage):", value=80) / 100
        
        staffing_change_2 = st.number_input("Enter the change in Staffing for scenario 2:", value=0)
        demand_change_2 = st.number_input("Enter the change in demand for scenario 2 (%):", value=10) / 100        
        occupancy_assumption_2 = st.number_input("Enter the occupancy assumption for scenario 2 (as a percentage):", value=80) / 100       
        
        # Calculate average Q4 time and occupancy rate with Staffing reduced by 50 and occupancy assumption of 60%
        average_q2_q4_time_1 = calculate_average_q2_q4_time(df, staffing_change_1, demand_change_1, occupancy_assumption_1)
        average_occupancy_rate_1 = calculate_average_occupancy_rate(df, staffing_change_1, demand_change_1, occupancy_assumption_1)

        # Calculate average Q4 time and occupancy rate with Staffing reduced by 80 and occupancy assumption of 70%
        average_q2_q4_time_2 = calculate_average_q2_q4_time(df, staffing_change_2, demand_change_2, occupancy_assumption_2)
        average_occupancy_rate_2 = calculate_average_occupancy_rate(df, staffing_change_2, demand_change_2, occupancy_assumption_2)

        # Display results in specified format
        st.write("Whatif Results------------>")
        results = pd.DataFrame({
            'Staffing': [int(df['Staffing'].mean() + staffing_change_1),
                                              int(df['Staffing'].mean() + staffing_change_2)],
            'Demand': [int(df['Demand'].mean()*(1 + demand_change_1)),
                                              int(df['Demand'].mean()*(1 + demand_change_2))],                                              
            'OCC Assumptions': [occupancy_assumption_1 * 100, occupancy_assumption_2 * 100],
            'Q2_Q4 Time': [round(average_q2_q4_time_1,2), round(average_q2_q4_time_2,2)],
            'Occupancy Rate': [round(average_occupancy_rate_1 * 100,2), round(average_occupancy_rate_2 * 100,2)]
        }, index=[f'Change by {staffing_change_1} Staffing', f'Change by {staffing_change_2} Staffing'])
        

        st.write(results)
        
               
        min_value = int(df['Staffing'].min())
        max_value = int(df['Staffing'].max())
        grouped_results = group_data(df, min_value, max_value)
        st.write("Data distribution based on Staffing buckets-------->")
        st.write(grouped_results)


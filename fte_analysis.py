import pandas as pd
import numpy as np
import streamlit as st
import re

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def calculate_average_q2_q4_time(df, staffing_change, occupancy_assumption):
    # Adjust Staffing by the given amount using the mean of Staffing (ACTUAL)
    df['Adjusted_Staffing'] = df['Staffing'].mean() + staffing_change
    
    # Change occupancy assumption to the given value
    df['Adjusted_Occupancy_Assumption'] = occupancy_assumption
    
    # Filter entries with Staffing within ±10% and occupancy assumption within ±10%
    staffing_lower_bound = df['Adjusted_Staffing'] * 0.9
    staffing_upper_bound = df['Adjusted_Staffing'] * 1.1
    occupancy_lower_bound = df['Adjusted_Occupancy_Assumption'] * 0.9
    occupancy_upper_bound = df['Adjusted_Occupancy_Assumption'] * 1.1
    
    filtered_df = df[(df['Staffing'] >= staffing_lower_bound) & (df['Staffing'] <= staffing_upper_bound) & 
                     (df['Occupancy Assumptions'] >= occupancy_lower_bound) & (df['Occupancy Assumptions'] <= occupancy_upper_bound)]
    
    # Check if "Q4" column is present
    if 'Q4' in df.columns:
        mean_value = filtered_df['Q4'].mean()
    else:
        mean_value = filtered_df['Q2'].mean()
    
    return mean_value

def calculate_average_occupancy_rate(df, staffing_change, occupancy_assumption):
    # Adjust Staffing by the given amount using the mean of Staffing
    df['Adjusted_Staffing'] = df['Staffing'].mean() + staffing_change
    
    # Change occupancy assumption to the given value
    df['Adjusted_Occupancy_Assumption'] = occupancy_assumption
    
    # Filter entries with Staffing within ±10% and occupancy assumption within ±10%
    staffing_lower_bound = df['Adjusted_Staffing'] * 0.9
    staffing_upper_bound = df['Adjusted_Staffing'] * 1.1
    occupancy_lower_bound = df['Adjusted_Occupancy_Assumption'] * 0.9
    occupancy_upper_bound = df['Adjusted_Occupancy_Assumption'] * 1.1
    
    filtered_df = df[(df['Staffing'] >= staffing_lower_bound) & (df['Staffing'] <= staffing_upper_bound) & 
                     (df['Occupancy Assumptions'] >= occupancy_lower_bound) & (df['Occupancy Assumptions'] <= occupancy_upper_bound)]
    
    # Calculate average Occupancy Rate for the filtered entries
    average_occupancy_rate = filtered_df['Occupancy Rate'].mean()
    
    return average_occupancy_rate

def group_data(df, min_value, max_value):
    bins = np.arange(min_value, max_value + 10, 10)

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
    grouped_df = df.groupby('Requirement_Range').agg({
        'Occupancy Rate': ['mean', 'count'],
        'Occupancy Assumptions': 'mean',
        q_column : 'mean'
    }).reset_index()
    grouped_df.columns = ['Requirement_Range', 'Occupancy Rate', 'Frequency', 'Occupancy Assumptions', 'Q2_Q4 Time']
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

        df = preprocess_data(df)
        
        # User inputs for Staffing change and occupancy assumption
        staffing_change_1 = st.number_input("Enter the change in Staffing for scenario 1:", value=0)
        occupancy_assumption_1 = st.number_input("Enter the occupancy assumption for scenario 1 (as a percentage):", value=100) / 100
        
        staffing_change_2 = st.number_input("Enter the change in Staffing for scenario 2:", value=0)
        occupancy_assumption_2 = st.number_input("Enter the occupancy assumption for scenario 2 (as a percentage):", value=100) / 100       
        
        # Calculate average Q4 time and occupancy rate with Staffing reduced by 50 and occupancy assumption of 60%
        average_q2_q4_time_1 = calculate_average_q2_q4_time(df, staffing_change_1, occupancy_assumption_1)
        average_occupancy_rate_1 = calculate_average_occupancy_rate(df, staffing_change_1, occupancy_assumption_1)

        # Calculate average Q4 time and occupancy rate with Staffing reduced by 80 and occupancy assumption of 70%
        average_q2_q4_time_2 = calculate_average_q2_q4_time(df, staffing_change_2, occupancy_assumption_2)
        average_occupancy_rate_2 = calculate_average_occupancy_rate(df, staffing_change_2, occupancy_assumption_2)
        

        # Display results in specified format
        st.write("Whatif Results------------>")
        results = pd.DataFrame({
            'Staffing': [df['Staffing'].mean() + staffing_change_1,
                                              df['Staffing'].mean() + staffing_change_2],
            'OCC Assumptions': [occupancy_assumption_1 * 100, occupancy_assumption_2 * 100],
            'Q2_Q4 Time': [average_q2_q4_time_1, average_q2_q4_time_2],
            'Occupancy Rate': [average_occupancy_rate_1 * 100, average_occupancy_rate_2 * 100]
        }, index=[f'Change by {staffing_change_1} Staffing', f'Change by {staffing_change_2} Staffing'])

        st.write(results)
        
        # st.write(df)
               
        min_value = int(df['Staffing'].min())
        max_value = int(df['Staffing'].max())
        grouped_results = group_data(df, min_value, max_value)
        st.write("Data distribution based on Staffing buckets-------->")
        st.write(grouped_results)

# if __name__ == '__main__':
#     main()

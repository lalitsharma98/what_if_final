import streamlit as st
import pandas as pd
from io import BytesIO

# Function to process the uploaded file
def process_file(file, sheets):
    occ_assumptions_dataframe = pd.DataFrame()
    for sheet in sheets:
        df = pd.read_excel(file, sheet_name=sheet, header=None)
        df = df.iloc[3:, 3:].reset_index(drop=True)
        df = df.T.reset_index(drop=True)
        
        df.columns = df.iloc[0].astype(str) + " " + df.iloc[1].astype(str) + " " + df.iloc[2].astype(str)
        df.columns = df.columns.str.replace("nan", "").str.strip()
        df = df[3:].reset_index(drop=True)
        df["Week Of:"] = pd.to_datetime(df["Week Of:"], errors="coerce").dt.date

        occ_columns = [col for col in df.columns if "OCC Assumptions" in col]
        volume_columns = [col for col in df.columns if "Volume (ACTUAL)" in col]
        staff_diff_columns = [col for col in df.columns if "STAFFING DIFFERENTIAL" in col]
        aht_columns = [col for col in df.columns if "AHT (ACTUAL)" in col]
        q4_columns = [col for col in df.columns if "Q4 Time" in col]
        q2_columns = [col for col in df.columns if "Q2 Time" in col]
        or_columns = [col for col in df.columns if "Occupancy Rate" in col]
        sr_columns = [col for col in df.columns if "Staffing Requirement (ACTUAL)" in col]
        staff_columns = [col for col in df.columns if "Staffing (ACTUAL/FORECASTED)" in col]

        combined_columns = occ_columns + volume_columns + staff_diff_columns + aht_columns + q4_columns + or_columns + sr_columns + staff_columns + q2_columns
        occ_df = df[["Week Of:"] + combined_columns]
        occ_assumptions_dataframe = pd.concat([occ_assumptions_dataframe, occ_df], axis=1)

    return occ_assumptions_dataframe, aht_columns, occ_columns

# Function to convert DataFrame to Excel and provide download link

def convert_df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Processed Data')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# Main function to run the Streamlit app
def run_data_extractor():

    # Streamlit app
    st.title("Excel File Processor")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsm", "xlsx"])
    

    if uploaded_file:
        # planner_type selection
        planner_type = st.multiselect("Select sheets to process", ["1. Weekly Planner OPI USD", "2. Weekly Planner OPI Global", 
                                                             "3. Weekly Planner VRI", "1. Weekly Planner OPI", "2. Weekly Planner VRI", 
                                                                   "3. UKD"])
        if planner_type:
            # Process the file
            final_df, aht_columns, occ_columns = process_file(uploaded_file, planner_type)

            # Extract column names for dropdown
            file_name = uploaded_file.name
            if file_name.lower().startswith('csa'):
                dropdown_options = [col.split("Combined")[1].split("AHT (ACTUAL)")[0].strip() for col in aht_columns]
            else:
                dropdown_options = [col.replace("AHT (ACTUAL)", "").strip() for col in aht_columns]
            
            # User selection from dropdown
            selected_option = st.selectbox("Select the level", dropdown_options)
            
            if selected_option:

                if file_name.lower().startswith('csa'):
                    # Filter columns based on user selection
                    selected_columns = [col for col in final_df.columns if selected_option in col]
                    final_df = final_df[["Week Of:"] + selected_columns]
                else:
                    selected_option_part = selected_option.split(' ', 1)
                    selected_columns = [col for col in final_df.columns if selected_option_part[0] in col 
                                        and selected_option_part[1] in col]                    
                    final_df = final_df[["Week Of:"] + selected_columns]
                                
                # Convert DataFrame to Excel
                processed_data = convert_df_to_excel(final_df)            
            
            # Display the final table
            st.write(final_df)
            
            # Convert DataFrame to Excel
            processed_data = convert_df_to_excel(final_df)
            
            # Provide download link
            st.download_button(label="Download data as Excel file", data=processed_data, 
                               file_name="processed_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")            
if __name__ == '__main__':
    main()              

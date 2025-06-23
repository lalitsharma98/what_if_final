# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Function to load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # Convert necessary columns to numeric
    columns_to_convert = ["Q2", "ABN %", "Loaded AHT", "Met", "Missed"]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Handle missing values
    df = df.dropna(subset=["Requirement", "Staffing", "Demand"])
    df = df.fillna(0)
    
    return df

# Function to train regression model
def train_regression_model(df):
    features = ["Q2", "ABN %", "Occ Assumption", "Staffing", "Calls", "Occupancy Rate"]
    target = "Requirement"
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Store feature names inside the model for consistency
    model.feature_names_in_ = features

    return model

# Function for Scenario Analysis
def analyze_scenarios(df, model, q2, abn, occ):
    test_data = pd.DataFrame({
        "Q2": [q2],
        "ABN %": [abn],
        "Occ Assumption": [occ],
        "Staffing": [df["Staffing"].median()],
        "Calls": [df["Calls"].median()],
        "Occupancy Rate": [df["Occupancy Rate"].median()],
    })

    # Ensure the test data matches the model's trained feature order
    test_data = test_data[model.feature_names_in_]

    # Predict FTE Requirement
    fte_prediction = model.predict(test_data)[0]
    return int(fte_prediction)

# Scenario 1: Calls per FTE Analysis
def calls_per_fte_scenario(df, percentage_changes):
    results = []
    for change in percentage_changes:
        adjusted_calls = df["Calls"].median()
        adjusted_fte = df["Staffing"].median() * (1 + change / 100)
        calls_per_fte = adjusted_calls / adjusted_fte if adjusted_fte != 0 else 0
        results.append({"FTE Change %": change, "Predicted Calls per FTE": calls_per_fte})
    return pd.DataFrame(results)

# Scenario 2: KPI Impact due to Calls per FTE Changes
def kpi_impact_scenario(df, model, percentage_changes):
    results = []
    for change in percentage_changes:
        adjusted_calls = df["Calls"].median() * (1 + change / 100)
        adjusted_fte = df["Staffing"].median()
        calls_per_fte = adjusted_calls / adjusted_fte if adjusted_fte != 0 else 0

        test_scenario = pd.DataFrame({
            "Q2": [df["Q2"].median() * (1 + change / 100)],
            "ABN %": [df["ABN %"].median() * (1 + change / 100)],
            "Occ Assumption": [df["Occ Assumption"].median()],
            "Staffing": [adjusted_fte],
            "Calls": [adjusted_calls],
            "Occupancy Rate": [df["Occupancy Rate"].median()],
        })

        # Ensure the test data matches the model's trained feature order
        test_scenario = test_scenario[model.feature_names_in_]

        fte_pred = model.predict(test_scenario)[0]

        results.append({
            "Calls per FTE Change %": change,
            "Predicted FTE Requirement": fte_pred,
            "Adjusted Q2 Time": df["Q2"].median() * (1 + change / 100),
            "Adjusted Abandonment Rate": df["ABN %"].median() * (1 + change / 100),
            "Predicted Calls per FTE": calls_per_fte
        })
    return pd.DataFrame(results)

# Scenario 3: FTE Impact of Demand Increase
def fte_impact_of_demand_increase(df, model, q2, abn, occ):
    demand_changes = [5, 10, 15, 20, 25]
    predictions = []
    for change in demand_changes:
        call_scenario = df["Calls"].median() * (1 + change / 100)
        test_scenario = pd.DataFrame({
            "Q2": [q2],
            "ABN %": [abn],
            "Occ Assumption": [occ],
            "Staffing": [df["Staffing"].median()],
            "Calls": [call_scenario],
            "Occupancy Rate": [df["Occupancy Rate"].median()],
        })

        test_scenario = test_scenario[model.feature_names_in_]

        fte_pred = model.predict(test_scenario)[0]
        predictions.append((change, int(fte_pred)))

    return predictions

# Scenario 4: Impact of OCC Assumption Change
def impact_of_occ_assumption_change(df, model, q2, abn):
    occ_changes = [-10, -5, 0, 5, 10]
    predictions = []
    for change in occ_changes:
        occ_scenario = df["Occ Assumption"].median() * (1 + change / 100)
        test_scenario = pd.DataFrame({
            "Q2": [q2],
            "ABN %": [abn],
            "Occ Assumption": [occ_scenario],
            "Staffing": [df["Staffing"].median()],
            "Calls": [df["Calls"].median()],
            "Occupancy Rate": [df["Occupancy Rate"].median()],
        })

        test_scenario = test_scenario[model.feature_names_in_]

        fte_pred = model.predict(test_scenario)[0]
        predictions.append((change, int(fte_pred)))

    return predictions

# Streamlit UI
st.title("What-If Analysis for FTE Requirements")
st.sidebar.header("User Inputs")

def scn6_7():
    file_path = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if file_path is not None:
        df = load_and_prepare_data(file_path)
        
        st.header("Filter Data ------------------------->")
        # Dropdown filters based on columns
        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())
        usd = st.selectbox("Select USD", df['USD'].unique())
        level = st.selectbox("Select Level", df['Level'].unique())

        # Filter DataFrame based on selected values
        df = df[(df['Language'] == language) & (df['Req Media'] == req_media) & 
                (df['USD'] == usd) & (df['Level'] == level)]
        
        st.header("Output ------------------------------>")        
        
        model = train_regression_model(df)
    
        q2 = st.sidebar.slider("Set Q2 Time", min_value=0, max_value=100, value=20, step=1)
        abn = st.sidebar.slider("Set Abandon Rate (%)", min_value=0, max_value=10, value=2, step=1)
        occ = st.sidebar.slider("Set Occupancy Rate (%)", min_value=0, max_value=100, value=80, step=1)
    
        fte_prediction = analyze_scenarios(df, model, q2, abn, occ)
        st.write(f"Predicted FTE based on user inputs: {fte_prediction}")
    
        st.header("Scenario 2: KPI Impact due to Calls per FTE Changes")
        percentage_changes = np.array([-10, -5, 0, 5, 10])
        kpi_impact_df = kpi_impact_scenario(df, model, percentage_changes)
        st.dataframe(kpi_impact_df) 

# Uncomment to run locally
if __name__ == '__main__':
    scn6_7()

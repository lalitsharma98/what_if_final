from modules.utils import st
from modules.whatif_scn1 import scn1
from modules.whatif_scn2 import scn2
# from modules.whatif_scn3 import scn3
from modules.whatif_scn4 import scn_4
from modules.whatif_scn5 import scn_5
from modules.whatif_scn6_7 import scn6_7



# Dictionary mapping scenarios to functions
SCENARIOS = {
    "Scenario 1": scn1,
    "Scenario 2": scn2,
    # "Scenario 3": scn3,
    "Scenario 4": scn_4,
    "Scenario 5": scn_5,
    "Scenario 6": scn6_7,
}

def main():
    st.title("What-If Analysis Tool")
    
    # Dropdown for selecting a scenario
    selected_scenario = st.selectbox("Select a Scenario", list(SCENARIOS.keys()))
    
    st.write(f"**You selected:** {selected_scenario}")
    
    # Run the corresponding function
    if selected_scenario in SCENARIOS:
        SCENARIOS[selected_scenario]()
    else:
        st.error("Invalid Scenario Selected")
    
if __name__ == "__main__":
    main()
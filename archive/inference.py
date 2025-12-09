import streamlit as st
import pandas as pd
import joblib
import lifelines
import matplotlib.pyplot as plt  # Corrected import
from sklearn.metrics import accuracy_score

# Title of the app
st.title("ARIA-Guard Inference Environment")

# Sidebar for settings
st.sidebar.title("Settings")

# File uploader for the user to upload their data
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Model Card Def
@st.dialog("Model Cards")
def models():
    tab1, tab2, tab3, tab4 = st.tabs(["M2", "M1","Forcast","Survivor Analysis"])
    with tab1:
        st.header("M2 Model")
        st.subheader("RandomForest Classifier Trained with SMOTE")

        st.write("Utilizes all available Clinical + Demographic + Biomarker Features.")
        with st.expander("Required Columns"):
            st.write(['RID','SCANDATE','NOFINDINGS',"ptau_pos_csf", "amyloid_pos_csf", "ptau_ab_ratio_csf", "ad_pathology_pos_csf", "PLASMA_NFL", "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"])  # List of required columns
        st.subheader("Stats")
        st.image("models/model_data/full_model_stats.png")
    with tab2:
        st.header("M1 Model")
        st.subheader("RandomForest Classifier Trained with SMOTE")

        st.write("Utilizes only  Clinical + Demographic Features.")
        with st.expander("Required Columns"):
            st.write(['RID','SCANDATE','NOFINDINGS', "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"])

        st.subheader("Stats")
        st.image("models/model_data/min_model_stats.png")
    with tab3:
        st.header("Forecast Model")
        st.subheader("ExtraTrees Regressor")
        st.write("Utilizes only  Clinical + Demographic Features.")
        with st.expander("Required Columns"):
            st.write(['RID','SCANDATE','NOFINDINGS', "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"])

        st.subheader("Stats")
        st.image("models/model_data/forcast_stats.png")
    with tab4:
        st.header("Survivor Analysis Model")
        st.subheader("CoxPH Regression Model from Lifelines Library")
        st.write("Utilizes only  Clinical + Demographic Features.")
        with st.expander("Required Columns"):
            st.write(['RID','SCANDATE','NOFINDINGS', "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"])

        st.subheader("Stats")
        st.image("models/model_data/sa_stats.png")
        

        
        
# Additional settings
st.sidebar.subheader("Model Settings")
modeltype = st.sidebar.selectbox("Select Model Type", options=["Full Model", "Minimal Model"])
st.sidebar.caption("Select the type of model you want to use for prediction of microbleed status.")
if st.sidebar.button("Model Card"):
    models()

        
# Survivor Analysis
survivability_prediction = st.sidebar.checkbox("Enable Survivor Analysis", value=False)
df_preview = st.sidebar.checkbox("Enable Dataset Preview", value=False)
inferencetype = st.sidebar.selectbox("Select Inference Type", options=["Test", "Predict"])


# Load the appropriate model based on the selected model type
if modeltype == "Full Model":
    model = joblib.load("models/md_classifier.pkl")
else:
    model = joblib.load("models/sm_classifier.pkl")

# Load Survivor Model
if survivability_prediction:
    survmodel = joblib.load("models/cph.pkl")
    regmodel = joblib.load("models/reg.pkl")
else:
    survmodel = None

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    # Assuming the model requires specific features for prediction
    features = st.multiselect("Select RID for prediction", options=data['RID'].tolist())
    # Display the data
    if df_preview:
        st.write("Data Preview:")
        st.dataframe(data)
    if modeltype == "Full Model":
        required_columns = ['RID','SCANDATE','NOFINDINGS',"ptau_pos_csf", "amyloid_pos_csf", "ptau_ab_ratio_csf", "ad_pathology_pos_csf", "PLASMA_NFL", "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"]  # List of required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        data = data[required_columns]
        if missing_columns:
            st.error(f"The following required columns are missing from the uploaded data: {', '.join(missing_columns)}")
    else:
        required_columns = ['RID','SCANDATE','NOFINDINGS', "MED_0", "MED_1", "MED_2", "MED_3", "MED_4", "MED_5", "MED_6", "MED_7", "MED_8", "MED_9", "MED_10", "MED_11", "MED_12", "MED_13", "MED_14", "MED_15", "MED_16", "MED_17", "MED_18", "MED_19", "MED_20", "MED_21", "MED_22", "MED_23", "MED_24", "MED_25", "MED_26", "MED_27", "MED_28", "MED_29", "MED_30", "MED_31", "PTGENDER", "PTEDUCAT", "RACE_ETHNICITY", "PTAGE", "GENOTYPE_encoded", "NORM_WMH", "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH"]  # List of required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        data = data[required_columns]
        if missing_columns:
            st.error(f"The following required columns are missing from the uploaded data: {', '.join(missing_columns)}")
    if inferencetype == "Test":
        # Extract NOFINDINGS column and save it to another dataframe
        nofindings_df = data[['RID', 'NOFINDINGS']].copy()
        # Optionally, drop the NOFINDINGS column from the original data
        preddf = data.drop(columns=['RID','SCANDATE','NOFINDINGS'])
    else:
        preddf = data.drop(columns=['RID','SCANDATE','NOFINDINGS'])      
    

    if st.button("Predict"):
        if features:
            results = pd.DataFrame()
            
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))  # Create a grid of subplots
            for idx, i in enumerate(features):
                # Make Predictions
                y_pred = model.predict(preddf[data['RID'] == i])
                # Get the predicted probabilities
                y_pred_proba = model.predict_proba(preddf[data['RID'] == i])
                # Get the confidence of the model on the predictions
                confidence = y_pred_proba.max(axis=1).mean()
                results = pd.concat([results, pd.DataFrame({'RID': [i], 'Prediction': [y_pred[0]], 'Confidence': [confidence]})], ignore_index=True)

            st.header("Results:", divider="gray")
            if inferencetype == "Test":
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Predictions:")
                    st.dataframe(results.sort_values(['RID']))
                with col2:
                    st.write("GT:")
                    st.dataframe(nofindings_df[nofindings_df['RID'].isin(features)].sort_values(['RID']))
                st.subheader("Accuracy:")
                results['NOFINDINGS'] = results['Prediction'].copy()
                results = results.sort_values(['RID'])
                gt = nofindings_df[nofindings_df['RID'].isin(features)].sort_values(['RID'])
                gt = gt['NOFINDINGS']
                accuracy = accuracy_score(gt, results['NOFINDINGS'])
                st.write(f"Accuracy Achieved: {accuracy}")
            else:
                st.write("Predictions:")
                st.dataframe(results)
            if survivability_prediction:  # Check if graphs should be shown
                st.header("Survival Function Plot:", divider="gray")
                columns_to_remove = ['PLASMA_NFL', 'ad_pathology_pos_csf', 'amyloid_pos_csf', 'ptau_ab_ratio_csf', 'ptau_pos_csf']
                try:
                    preddf = preddf.drop(columns=[col for col in columns_to_remove if col in preddf.columns])
                except KeyError as e:
                    pass
                for i in features:    
                    reg = regmodel.predict(preddf[data['RID'] == i])  # Plot the output of the regmodel on the same axis
                    surv_func = survmodel.predict_survival_function(preddf[data['RID'] == i])
                    surv_func.plot(ax=axes, label=i)  # Plot on the corresponding axis
                    labels = {i: f'RID {i}' for i in features}
                    axes.legend([labels[i] for i in features])
                    color = axes.get_lines()[-1].get_color()  # Get the color of the last plotted line
                    for value in reg:
                        axes.axvline(x=value, color=color, linestyle='--', label='_nolegend_')
                    axes.set_title(f'Survival Function for RIDs')  # Set title for each plot
                    axes.set_xlabel('Time (3 Month Intervals)')
                    axes.set_ylabel('Survival Probability (No ARIA Effect)')
                st.pyplot(fig)  # Use st.pyplot to display the figure correctly
                plt.close(fig)  # Close the figure to avoid display issues
                st.caption("Each solid line shows how the likelihood of conversion to microbleed status changes over 3 month intervals. The dotted vertical lines show the predicted period of microbleed conversion for their respective lines.")
            

                    
        else:
            st.warning("Please select at least one RID for prediction.")

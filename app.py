import streamlit as st
from src.model import load_model, predict
from src.utils import load_data
import src.train as trainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title='Student Performance Predictor v4', layout='centered')

st.title('Student Performance Predictor ')
st.write('Clean dashboard: enter student details on the sidebar and click Predict. No raw data shown.')

# Sidebar inputs
st.sidebar.header('Student inputs (clean & minimal)')
gender = st.sidebar.selectbox('Gender', ['male','female'])
study_hours = st.sidebar.slider('Study hours per day', 1, 8, 3)
failed_subjects = st.sidebar.number_input('Number of failed subjects', min_value=0, max_value=10, value=0)
absences = st.sidebar.number_input('Absences per month', min_value=0, max_value=31, value=2)
wifi = st.sidebar.selectbox('Wi‑Fi at home', ['yes','no'])
hours_wasted = st.sidebar.slider('Hours wasted on games/internet per day', 0.0, 10.0, 2.0, step=0.5)
grasp = st.sidebar.slider('Ability to grasp knowledge (1-10)', 1.0, 10.0, 6.0, step=0.1)
parental_support = st.sidebar.selectbox('Parental support', ['low','medium','high'])
sleep = st.sidebar.slider('Sleep hours per day', 3.0, 10.0, 7.0, step=0.5)
motivation = st.sidebar.slider('Motivation level (1-10)', 1.0, 10.0, 6.0, step=0.1)
extra_classes = st.sidebar.selectbox('Attends extra classes/tutoring', ['no','yes'])

inputs = {
    'gender': gender,
    'study_hours': study_hours,
    'failed_subjects': failed_subjects,
    'absences': absences,
    'wifi': wifi,
    'hours_wasted': hours_wasted,
    'grasp': grasp,
    'parental_support': parental_support,
    'sleep': sleep,
    'motivation': motivation,
    'extra_classes': extra_classes
}

# Tips panel (minimal)
with st.expander('Quick tips (optional)'):
    st.markdown("""
- Increase **study hours** and **graspability** to improve passing chances.
- Reduce **hours wasted** and **absences**.
- **Parental support** and **extra classes** can help in low-probability cases.
""")

# Predict button (main UI has only this and tips)
if st.button('Predict pass probability'):
    try:
        model = load_model()
    except FileNotFoundError:
        st.info('Training a quick model now (only because none was found).')
        model = trainer.train_and_save()
    res = predict(inputs, model=model)
    prob = res['probability_pass']
    st.metric('Probability of passing', f"{prob} %")
    if prob >= 80:
        st.success('High chance of passing. Keep doing what you are doing.')
    elif prob >= 50:
        st.warning('Moderate chance. Target study habits for improvement.')
    else:
        st.error('Low chance. Consider interventions: tutoring, reduce wasted hours, increase study time.')

    # After prediction -> show visualizations (aggregated; no raw data)
    df = load_data()
    st.markdown('---')
    st.header('Post-prediction analytics (aggregated)')
    # Heatmap of numeric correlations
    numeric = df.select_dtypes(include=['int64','float64'])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Feature importance (simple permutation via model.feature_importances_ if available)
    st.subheader('Model feature importances (approx)')
    try:
        # get importances from underlying RandomForest if pipeline used
        import numpy as np
        clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else None
        if clf is not None and hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            # map to feature names from preprocessor
            pre = model.named_steps['pre']
            # numeric feature names
            num_feats = pre.transformers_[0][2]
            # categorical feature names after OneHotEncoder
            cat_transformer = pre.transformers_[1][1]
            try:
                cat_names = pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2]).tolist()
            except Exception:
                # fallback names
                cat_names = pre.transformers_[1][1].get_feature_names().tolist() if hasattr(pre.transformers_[1][1], 'get_feature_names') else []
            feat_names = list(num_feats) + cat_names
            imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(12)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            sns.barplot(x='importance', y='feature', data=imp_df, ax=ax2)
            st.pyplot(fig2)
        else:
            st.info('Feature importances not available for the loaded model.')
    except Exception as e:
        st.info(f'Could not compute feature importances: {e}')

else:
    # show only minimal dashboard components (no data, no visuals)
    st.write('Enter details in the sidebar and click **Predict pass probability**.')

st.caption('App designed to avoid exposing raw sample data. Replace data/student_data.csv with your dataset for real predictions.')
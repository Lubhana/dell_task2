import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Data Loading ---
# Using @st.cache_data to speed up app by loading data only once
@st.cache_data
def load_data():
    try:
        attendance_df = pd.read_csv("attendance_logs.csv")
        events_df = pd.read_csv("event_participation.csv")
        lms_df = pd.read_csv("lms_usage.csv")
        return attendance_df, events_df, lms_df
    except FileNotFoundError as e:
        st.error(f"Error: Missing data file - {e.filename}")
        st.info("Please make sure `attendance_logs.csv`, `event_participation.csv`, and `lms_usage.csv` are in the same folder as your app.")
        return None, None, None

attendance_df, events_df, lms_df = load_data()

# Stop the app if data couldn't be loaded
if attendance_df is None:
    st.stop()

# --- App Layout ---
st.title("📊 Smart Campus Insights")
st.sidebar.header("🔍 Filters")

# --- REWRITTEN SIDEBAR LOGIC ---
# This new logic prevents the sidebar from overflowing

students = attendance_df['StudentID'].unique()

# 1. Add a "Select All" checkbox
select_all = st.sidebar.checkbox("Select All Students", value=True)

# 2. Update the selected_students logic
if select_all:
    # If "Select All" is checked, use all students
    selected_students = students
else:
    # If unchecked, show the multiselect box (now empty by default)
    selected_students = st.sidebar.multiselect("Select Students", students, default=[])

# --- End of rewritten logic ---


# --- Main Page Content ---

# Check if any students are selected before trying to filter
if len(selected_students) == 0:
    st.warning("Please select at least one student from the sidebar to view insights.")
    st.stop()

# Filter data based on selection
filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

# --- Visualizations ---

st.subheader("📋 Attendance Trends")
if not filtered_attendance.empty:
    attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
    st.line_chart(attendance_summary)
else:
    st.info("No attendance data for selected students.")

st.subheader("🎓 Event Participation")
if not filtered_events.empty:
    event_counts = filtered_events['EventName'].value_counts()
    st.bar_chart(event_counts)
else:
    st.info("No event data for selected students.")

st.subheader("💻 LMS Usage Patterns")
if not filtered_lms.empty:
    lms_summary = filtered_lms.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean()
    st.dataframe(lms_summary)
else:
    st.info("No LMS data for selected students.")

# --- Machine Learning Model ---
st.subheader("🤖 Predict Student Engagement Risk")

try:
    # Prepare data for ML
    ml_data = pd.merge(attendance_df.groupby('StudentID')['Status'].apply(lambda x: (x == 'Absent').mean()).reset_index(name='AbsenceRate'),
                       lms_df.groupby('StudentID')[['SessionDuration', 'PagesViewed']].mean().reset_index(),
                       on='StudentID')

    ml_data['Engagement'] = (ml_data['AbsenceRate'] < 0.2).astype(int)

    X = ml_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
    y = ml_data['Engagement']
    
    # Check if we have enough data to train
    if len(X) < 10 or len(y.unique()) < 2:
        st.warning("Not enough data to build a predictive model. Please check your data sources.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.text("Model Performance (based on test data):")
        st.text(classification_report(y_test, y_pred))

        # --- Prediction UI ---
        st.subheader("📈 Predict Engagement for New Student")
        absence_rate = st.number_input("Absence Rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.1)
        session_duration = st.number_input("Average Session Duration (minutes)", min_value=0.0, value=30.0)
        pages_viewed = st.number_input("Average Pages Viewed", min_value=0.0, value=10.0)

        if st.button("Predict Engagement"):
            prediction = model.predict([[absence_rate, session_duration, pages_viewed]])
            result = "Engaged" if prediction[0] == 1 else "At Risk"
            if result == "Engaged":
                st.success(f"Predicted Engagement Status: {result}")
            else:
                st.error(f"Predicted Engagement Status: {result}")

except Exception as e:
    st.error(f"An error occurred while building the model: {e}")
    st.info("This usually happens if there is not enough data (e.g., no LMS data).")

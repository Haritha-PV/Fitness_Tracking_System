import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page title and other configuration options
st.set_page_config(
    page_title="Fitness Tracker App",
    page_icon=":bar_chart:",  # Emoji for the page icon
    layout="wide"  # Optional: Use "wide" or "centered" layout
)

# Load data
# Load the dataset into a DataFrame
df = pd.read_csv('C:/Users/harit/OneDrive/Desktop/Meta Scifor technologies/Haritha-P-V_Scifor/Haritha-P-V_Scifor/Main_Project/Fitness_tracker_dataset.csv')

# About Page Function
def about():
    st.title("About the Project")

    st.write(
        """
        ## Project Overview
        This Fitness Tracker App aims to help users track their fitness activities, including their daily steps, BMI, and progress towards fitness goals. It includes features for calculating BMI, setting and tracking weekly step goals, and exploring a dataset.

        ## Key Features
        - **BMI Calculator:** Allows users to calculate their Body Mass Index based on weight and height and provides an interpretation of the BMI category.
        - **Goal Setting & Progress Tracking:** Users can set weekly step goals and track their progress towards these goals with visualizations.
        - **Dataset Overview:** Provides an overview of the dataset used in the app, including its shape, a few rows, and descriptive statistics.

        ## Models and Predictions
        The app uses machine learning models to predict fitness-related metrics based on user inputs. The models include:
        - **Polynomial Regression Model:** Provides predictions using polynomial features of the input data.
        - **Random Forest Model:** Uses an ensemble of decision trees to make predictions based on input features.
      
        ## Outputs and Findings
        - Predictions are made based on the input features.
        - Visualizations and user interactions are provided to enhance the fitness tracking experience.

        ## Future Enhancements
        - Integrate more advanced models and algorithms.
        - Add features for tracking different types of fitness activities.
        - Enhance data visualization and user interaction.
        """
    )

# BMI Calculator Function
def bmi_calculator():
    st.header("BMI Calculator")
    st.image("bmi.jpg")

    # Input for BMI calculation
    weight = st.number_input("Enter your weight (kg):", min_value=0.0, format="%.2f")
    height = st.number_input("Enter your height (cm):", min_value=0.0, format="%.2f")
    
    if weight > 0 and height > 0:
        # Convert height from cm to meters
        height_m = height / 100
        bmi = weight / (height_m ** 2)
          
        # Determine the BMI category
        if bmi < 18.5:
            category = "Underweight"
            color = "blue"
        elif 18.5 <= bmi < 24.9:
            category = "Normal weight"
            color = "green"
        elif 25 <= bmi < 29.9:
            category = "Overweight"
            color = "red"
        else:
            category = "Obese"
            color = "red"

        st.markdown(
            f"<h3 style='font-size:28px; color:{color}; text-align:center;'>Category: {category}</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='font-size:36px; text-align:center;'>Please enter valid weight and height.</h2>",
            unsafe_allow_html=True
        )    

# Goal Tracking Function
def goal_tracking():
    st.header("Goal Setting & Progress Tracking")
    
    # User inputs for setting goals
    goal_steps = st.number_input("Set your weekly steps goal:", min_value=0, step=1000)
    
    # Placeholder for historical steps data
    st.write("### Enter your weekly steps")
    steps_data = []
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_of_week:
        steps = st.number_input(f"Steps on {day}:", min_value=0, format="%d", key=day)
        steps_data.append(steps)
    
    # Calculate total steps and progress
    total_steps = sum(steps_data)
    progress = min(total_steps / goal_steps * 100, 100) if goal_steps > 0 else 0

    # Display progress
    st.write(f"### Total Steps: {total_steps}")
    st.write(f"### Progress towards Goal: {progress:.2f}%")
    
    # Display progress bar
    st.progress(progress / 100)
    
    # Display steps data as a bar chart with correct order
    st.write("### Steps Data")
    steps_df = pd.DataFrame({'Day': days_of_week, 'Steps': steps_data})
    steps_df['Day'] = pd.Categorical(steps_df['Day'], categories=days_of_week, ordered=True)
    steps_df = steps_df.sort_values('Day')
    st.bar_chart(steps_df.set_index('Day'))


# Main Function
def main():
    st.sidebar.title("Navigation 🌟")
    st.sidebar.subheader("📋 Go to")
    page = st.sidebar.radio("Select a Page:", ["🏠 Home", "ℹ️ About", "📊 Dataset", "📉 BMI Calculator", "🎯 Goal Setting & Progress"])
    
    # Add a motivational quote
    st.sidebar.markdown("✨ **Make your health a priority!** ✨")
    st.sidebar.image("fit.gif", use_column_width=True, caption="Stay Fit!")
    
    if page == "🏠 Home":
        st.markdown("<h1 style='font-size:50px; color:darkblue; text-align:center;'>Welcome to the Fitness Tracker App!</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; color:green;'>✨ Explore the Dataset, use the BMI calculator, and Track your goals! 🏃‍♂️📈</h2>",
            unsafe_allow_html=True
        )
        st.image("fitnessi.jpg")
        st.markdown(
            "<h2 style='text-align:center;'>App by: HARITHA P V</h2>",
            unsafe_allow_html=True
        )

    elif page == "ℹ️ About":
        about()

    elif page == "📊 Dataset":
        st.title("Dataset Overview")
        st.write("### Shape of the Dataset")
        st.markdown(
            f"<p style='font-size:24px;'>The dataset contains <b>{df.shape[0]}</b> rows and <b>{df.shape[1]}</b> columns.</p>",
            unsafe_allow_html=True
        )
        st.write("### Data Information")
        st.image("info.png")

        st.write("### First Few Rows of the Dataset")
        st.write(df.head())

        st.write("### Descriptive Statistics")
        st.write(df.describe())
  
    elif page == "📉 BMI Calculator":
        bmi_calculator()      
   
    elif page == "🎯 Goal Setting & Progress":
        goal_tracking()

# Run the app
if __name__ == "__main__":
    main()

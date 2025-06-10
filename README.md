# Telco Customer Churn Prediction App 📊

This repository hosts a Streamlit web application that utilizes a Machine Learning model to predict customer churn for a telecommunications company, based on the Telco Customer Churn dataset.

## 🌟 Key Features

* **Intuitive User Interface:** A user-friendly Streamlit platform for inputting customer data.
* **Churn Prediction:** Employs an optimized Logistic Regression model to forecast whether a customer is likely to churn.
* **Probability Output:** Displays the probability of a customer churning.
* **Actionable Insights:** Provides basic recommendations to assist in retaining high-risk customers.

## 🛠️ Technologies Used

* **Python:** The primary programming language.
* **Streamlit:** For building the interactive web application.
* **scikit-learn:** Utilized for machine learning model development and preprocessing (Logistic Regression, StandardScaler).
* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.

## 📁 Project Structure

* `app.py`: The main Streamlit application script, responsible for building the UI and making predictions.
* `optimized_logistic_regression_model.pkl`: The trained and saved Logistic Regression model.
* `scaler.pkl`: The fitted StandardScaler object, used to transform new input data consistently with the training data.
* `median_total_charges_train.pkl`: The median value of `TotalCharges` from the training data, used for imputing missing values during preprocessing.
* `feature_columns.pkl`: A list of all feature names (columns) in the correct order that the model expects after one-hot encoding.
* `requirements.txt`: A list of all Python libraries and their versions required to run the application.
* `.gitignore`: Specifies files and folders that Git should ignore (e.g., `venv/`).

## 🚀 How to Run Locally

Follow these steps to set up and run the application on your local machine:

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone this project:
    ```bash
    git clone [https://github.com/palmyz000/streamlit_churn_prediction.git](https://github.com/palmyz000/streamlit_churn_prediction.git)
    cd streamlit_churn_prediction
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies:
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    With your virtual environment activated (you should see `(venv)` prefixing your command line), install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    Finally, launch the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    The application will automatically open in your web browser (usually at `http://localhost:8501`).

## 📊 Usage Example


![image](https://github.com/user-attachments/assets/1a5f5a49-4d3b-4c9a-9a81-c1725e930da7)

## 📈 Model Evaluation Metrics (Logistic Regression)

The Logistic Regression model utilized in this application was optimized and evaluated with the following key metrics:

* **ROC AUC:** 0.84
* **Precision-Recall AP:** 0.63
  
## ✉️ Contact

For any questions or suggestions, feel free to reach out:
* Suphawit MeeSak
* Suphawit11@icloud.com
* https://www.linkedin.com/in/suphawit-meesak/

---

import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load and Prepare the Data
def choose_file_dialog(initial_dir):
    Tk().withdraw()
    return askopenfilename(
        initialdir=initial_dir,
        title="Select the CSV file",
        filetypes=[("CSV files", "*.csv")]
    ) 
#IMPORTANT NOTE: The code will show an error/exit the program if the user cancels the file selection in the file dialouge box, 
# The code uses the file pattern of "Task 3 and 4_Loan_Data.csv"
# Please ensure to select the correct CSV file to avoid any issues.
csv_filename = choose_file_dialog(".")

if not os.path.exists(csv_filename):
    print(f"Error: The file '{csv_filename}' was not found.")
    print("Please ensure it is in the same directory as this script.")
    exit()

print("Loading data...")
df = pd.read_csv(csv_filename)

# Features and target variable
# 'customer_id' is excluded from training as it is just an identifier
features = [
    "credit_lines_outstanding", 
    "loan_amt_outstanding", 
    "total_debt_outstanding", 
    "income", 
    "years_employed", 
    "fico_score"
]
target = "default"

X = df[features]
y = df[target]

# Split data for a quick evaluation check
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Model
print("Training the Gradient Boosting model...")
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Model training complete. (Test AUC Score: {auc_score:.4f})\n")

# Helper function to handle user inputs cleanly
def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

# Continuous Loop for Predictions
while True:
    print("\n" + "="*50)
    print("  NEW PREDICTION (Type 'q' as Customer ID to exit)")
    print("="*50)
    
    cust_id = input("Enter Customer ID: ").strip()
    if cust_id.lower() == 'q':
        print("\nExiting the script. Goodbye!")
        break

    # Gather initial data
    credit_lines = get_float_input("Enter Credit Lines Outstanding: ")
    loan_amt = get_float_input("Enter Loan Amount Outstanding ($): ")
    total_debt = get_float_input("Enter Total Debt Outstanding ($): ")
    income = get_float_input("Enter Income ($): ")
    years_employed = get_float_input("Enter Years Employed: ")
    fico = get_float_input("Enter FICO Score: ")

    # --- Review and Edit Data Before Prediction ---
    while True:
        print("\n" + "-"*30)
        print("REVIEW CUSTOMER DETAILS:")
        print(f"[1] Customer ID:             {cust_id}")
        print(f"[2] Credit Lines Outstanding: {credit_lines}")
        print(f"[3] Loan Amount Outstanding:  ${loan_amt:,.2f}")
        print(f"[4] Total Debt Outstanding:   ${total_debt:,.2f}")
        print(f"[5] Income:                   ${income:,.2f}")
        print(f"[6] Years Employed:           {years_employed}")
        print(f"[7] FICO Score:               {fico}")
        print("-"*30)
        
        confirm = input("Are these details correct? (y/n): ").strip().lower()
        
        if confirm == 'y':
            break  # Break this review loop and proceed to prediction
        elif confirm == 'n':
            choice = input("Enter the number [1-7] of the detail you want to change: ").strip()
            
            if choice == '1':
                cust_id = input("Enter new Customer ID: ").strip()
            elif choice == '2':
                credit_lines = get_float_input("Enter new Credit Lines Outstanding: ")
            elif choice == '3':
                loan_amt = get_float_input("Enter new Loan Amount Outstanding ($): ")
            elif choice == '4':
                total_debt = get_float_input("Enter new Total Debt Outstanding ($): ")
            elif choice == '5':
                income = get_float_input("Enter new Income ($): ")
            elif choice == '6':
                years_employed = get_float_input("Enter new Years Employed: ")
            elif choice == '7':
                fico = get_float_input("Enter new FICO Score: ")
            else:
                print("Invalid selection. Please enter a number from 1 to 7.")
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

    # Predict Probability of Default (PD)
    new_customer_df = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines,
        "loan_amt_outstanding": loan_amt,
        "total_debt_outstanding": total_debt,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico
    }])

    pd_value = model.predict_proba(new_customer_df)[0][1]

    # Calculate Expected Loss (EL)
    lgd = 0.9 #The recovery rate was 10%
    expected_loss = (pd_value) * lgd * loan_amt

    # Display Results
    print("\n" + "-"*40)
    print(f"FINAL RESULTS FOR CUSTOMER: {cust_id}")
    print("-"*40)
    print(f"Probability of Default (PD): {pd_value:.2%}")
    print(f"Loss Given Default (LGD):    {lgd:.0%}")
    print(f"Loan Amount Outstanding:     ${loan_amt:,.2f}")
    print(f"Expected Loss (EL):          ${expected_loss:,.2f}")
    print("-"*40)
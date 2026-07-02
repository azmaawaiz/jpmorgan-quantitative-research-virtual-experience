import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
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
# IMPORTANT NOTE: The code will show an error/exit the program if the user cancels the file selection in the file dialouge box, 
# The code uses the file pattern of "Task 3 and 4_Loan_Data.csv"
# Please ensure to select the correct CSV file to avoid any issues.
csv_filename = choose_file_dialog(".")

if not csv_filename or not os.path.exists(csv_filename):
    print(f"Error: The file '{csv_filename}' was not found or no file was selected.")
    exit()

try:
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded '{csv_filename}'.")
    print(f"Initial Shape: {df.shape}")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Extract and clean the relevant columns 

target_cols = ["fico_score", "default"] 
for col in target_cols:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the CSV file.")
        exit()

data_clean = df[target_cols].dropna().reset_index(drop=True)
print("\nData Preview (First 5 rows):")
print(data_clean.head())

# Use a Decision Tree to find the optimal bucket boundaries

X = data_clean[["fico_score"]]
y = data_clean["default"] # Predicting default probability

max_buckets = 5 
tree_model = DecisionTreeRegressor(max_leaf_nodes=max_buckets, random_state=42)
tree_model.fit(X, y)

# Extract and clean thresholds
thresholds = sorted(tree_model.tree_.threshold[tree_model.tree_.threshold != -2])

# Construct the actual rating boundaries from 300 to 850
boundaries = [300] + [int(round(t)) for t in thresholds] + [850]
boundaries = sorted(list(set(boundaries))) # Ensure uniqueness and order

print("\n--- Optimal FICO Bucket Boundaries Found ---")
print(f"Boundaries: {boundaries}")

# Generate the Rating Map 
print("\n--- Generated Rating Map ---")
# To ensure lower rating = better credit, we sort the buckets from highest FICO to lowest
fico_ranges = []
for i in range(len(boundaries) - 1):
    fico_ranges.append((boundaries[i], boundaries[i+1]))

# Calculate default rate per bucket to sort them by credit quality
bucket_stats = []
for low, high in fico_ranges:
    subset = data_clean[(data_clean['fico_score'] >= low) & (data_clean['fico_score'] <= high)]
    def_rate = subset['default'].mean() if len(subset) > 0 else 0
    bucket_stats.append({'range': (low, high), 'def_rate': def_rate})

# Sort by default rate ascending (lowest default rate gets Rating 1)
bucket_stats.sort(key=lambda x: x['def_rate'])

for rating, stats in enumerate(bucket_stats, start=1):
    print(f"Rating {rating}: FICO {stats['range'][0]} - {stats['range'][1]} (Default Rate: {stats['def_rate']:.2%})")

# Interactive prompt for the Visualization
user_input = input("\nWould you like to visualize the rating map? (y/n): ").strip().lower()

if user_input == 'y':
    print("\nGenerating default rate by FICO bucket visualization (Ordered Low to High)...")
    
    # Create a copy of bucket_stats and sort it by the starting FICO score (low to high)
    visual_buckets = list(bucket_stats)
    visual_buckets.sort(key=lambda x: x['range'][0])
    
    # Format data for plotting
    plot_data = []
    for stats in visual_buckets:
        low, high = stats['range']
        
        # Find the original rating number assigned in Section 4
        # (Since we re-sorted, we look up its original position)
        orig_rating = next(i for i, b in enumerate(bucket_stats, start=1) if b['range'] == (low, high))
        
        # Create a clean label for the X-axis showing FICO range first
        bucket_label = f"{low}-{high}\n(Rating {orig_rating})"
        
        # Calculate rates
        def_rate = stats['def_rate']
        non_def_rate = 1.0 - def_rate
        
        # Append rows for both Defaulter and Non-Defaulter portions
        plot_data.append({'FICO Range': bucket_label, 'Rate': non_def_rate, 'Status': 'Non-Default'})
        plot_data.append({'FICO Range': bucket_label, 'Rate': def_rate, 'Status': 'Default'})
        
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    
    # Plotting FICO buckets vs Default Rate
    ax = sns.barplot(
        data=df_plot, 
        x="FICO Range", 
        y="Rate", 
        hue="Status",
        palette={"Non-Default": "blue", "Default": "orange"} # Maps Blue and Orange
    )
    
    # Format Y-axis to show percentages (0% to 100%)
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.title("Default Rate vs. FICO Score Buckets (Ascending FICO Order)", fontsize=14)
    plt.xlabel("FICO Score Ranges (Lowest to Highest)", fontsize=12)
    plt.ylabel("Percentage Rate", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Legend display
    plt.legend(title="Status", loc="upper right")
    plt.tight_layout()
    plt.show()
    print("Execution complete.")
else:
    print("\nEnding execution. Goodbye!")
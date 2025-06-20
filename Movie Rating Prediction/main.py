# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from io import StringIO

# Redirect print statements to a StringIO object for capturing output
class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.string_io = StringIO()

    def write(self, message):
        self.string_io.write(message)
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# Step 1: Load the dataset
try:
    data = pd.read_csv('IMDb India Movies.csv', encoding='latin1')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDb India Movies.csv' not found in the current directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Data Preprocessing
# Drop rows where 'Rating' is missing (our target)
data = data.dropna(subset=['Rating'])
print("\nShape after dropping rows with missing Rating:", data.shape)

# Fill missing values with defaults
data['Genre'] = data['Genre'].fillna('Unknown')
data['Director'] = data['Director'].fillna('Unknown')
data['Actor 1'] = data['Actor 1'].fillna('Unknown')
data['Actor 2'] = data['Actor 2'].fillna('Unknown')
data['Actor 3'] = data['Actor 3'].fillna('Unknown')
data['Year'] = data['Year'].fillna('(0)')  # Temporary placeholder
data['Duration'] = data['Duration'].fillna('0 min')
data['Votes'] = data['Votes'].fillna('0')

# Convert 'Genre' to list of strings
data['Genre'] = data['Genre'].astype(str).str.split(', ')
print("\nSample of Genre after splitting:")
print(data['Genre'].head())

# Check for missing values
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Step 3: Feature Engineering
# One-hot encode 'Genre'
try:
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(data['Genre']), 
                                 columns=mlb.classes_, 
                                 index=data.index)
    print("\nGenre Classes Encoded:", mlb.classes_)
    print("Genre Encoded Data Types:")
    print(genre_encoded.dtypes)
    data = pd.concat([data, genre_encoded], axis=1)
except Exception as e:
    print(f"Error encoding genres: {e}")
    exit()

# Frequency encoding for categorical features
data['Director_Freq'] = data['Director'].map(data['Director'].value_counts())
data['Actor1_Freq'] = data['Actor 1'].map(data['Actor 1'].value_counts())
data['Actor2_Freq'] = data['Actor 2'].map(data['Actor 2'].value_counts())
data['Actor3_Freq'] = data['Actor 3'].map(data['Actor 3'].value_counts())

# Clean numeric features
# Year: Extract 4-digit year and convert to numeric
data['Year'] = data['Year'].astype(str).str.extract(r'(\d{4})')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
year_median = data['Year'].median()
data['Year'] = data['Year'].fillna(year_median)
print("Year median:", year_median)

# Duration: Remove ' min' and convert to numeric
data['Duration'] = data['Duration'].astype(str).str.replace(' min', '')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
duration_median = data['Duration'].median()
data['Duration'] = data['Duration'].fillna(duration_median)
print("Duration median:", duration_median)

# Votes: Remove commas and convert to numeric
data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce').fillna(0)

# Keep a copy of the data with categorical columns for prediction
data_with_categoricals = data.copy()

# Drop original categorical columns from the training data
data = data.drop(['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)

print("\nData After Feature Engineering:")
print(data.head())
print("\nData Types After Feature Engineering:")
print(data.dtypes)

# Step 4: Prepare Data for Modeling
try:
    X = data.drop(['Rating', 'Name'], axis=1)  # Features
    y = data['Rating']  # Target
    print("\nFeatures (X) Head:")
    print(X.head())
    print("\nFeature Data Types:")
    print(X.dtypes)
    # Check for non-numeric columns
    non_numeric = X.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"Non-numeric columns found in X: {non_numeric}")
        exit()
except Exception as e:
    print(f"Error preparing data for modeling: {e}")
    exit()

# Split into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

# Step 5: Linear Regression Model
try:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    print("\nLinear Regression Results:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
    print("R² Score:", r2_score(y_test, y_pred_lr))
except Exception as e:
    print(f"Error training Linear Regression: {e}")
    exit()

# Step 6: Random Forest Model
try:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    print("\nRandom Forest Results:")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    print("R² Score:", r2_score(y_test, y_pred_rf))

    # Cross-Validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print("\nCross-Validation R² Scores:", cv_scores)
    print("Mean R²:", cv_scores.mean())
    print("Standard Deviation:", cv_scores.std())

    # Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    print("\nFeature Importances (Top 10):")
    print(importances.sort_values(ascending=False).head(10))

except Exception as e:
    print(f"Error training Random Forest: {e}")
    exit()

# Step 7: Save the Model
model_path = r"C:\Users\Hello !\Documents\Movie Rating Prediction\movie_rating_model.pkl"
try:
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved as '{model_path}'")
except Exception as e:
    print(f"Error saving model: {e}")
    try:
        fallback_path = "movie_rating_model_fallback.pkl"
        joblib.dump(rf_model, fallback_path)
        print(f"\nFallback: Model saved as '{fallback_path}' in current directory")
    except Exception as e2:
        print(f"Error saving model in fallback location: {e2}")
        exit()

# Step 8: Example Prediction
try:
    sample = X_test.iloc[0:1]
    predicted_rating = rf_model.predict(sample)[0]
    actual_rating = y_test.iloc[0]
    print(f"\nExample Prediction: Predicted Rating = {predicted_rating:.2f}, Actual Rating = {actual_rating:.2f}")
except Exception as e:
    print(f"Error making prediction: {e}")
    exit()

# Step 9: Prediction Function for New Movies
def predict_movie_rating(genres, director, actor1, actor2, actor3, year, duration, votes, mlb, data, model, X_columns):
    try:
        # Ensure genres is a list
        if isinstance(genres, str):
            genres = [genres]
        print(f"\nPredicting for genres: {genres}, director: {director}, actors: {actor1}, {actor2}, {actor3}")

        # Create a new data point
        new_data = pd.DataFrame({
            'Year': [year],
            'Duration': [duration],
            'Votes': [votes],
            'Director': [director],
            'Actor 1': [actor1],
            'Actor 2': [actor2],
            'Actor 3': [actor3],
            'Genre': [genres]
        })
        print("New data created:")
        print(new_data)

        # Frequency encoding
        director_counts = data['Director'].value_counts()
        actor1_counts = data['Actor 1'].value_counts()
        actor2_counts = data['Actor 2'].value_counts()
        actor3_counts = data['Actor 3'].value_counts()
        
        new_data['Director_Freq'] = new_data['Director'].map(director_counts).fillna(1)
        new_data['Actor1_Freq'] = new_data['Actor 1'].map(actor1_counts).fillna(1)
        new_data['Actor2_Freq'] = new_data['Actor 2'].map(actor2_counts).fillna(1)
        new_data['Actor3_Freq'] = new_data['Actor 3'].map(actor3_counts).fillna(1)
        print("After frequency encoding:")
        print(new_data)

        # One-hot encode genres
        genre_encoded = pd.DataFrame(mlb.transform(new_data['Genre']), columns=mlb.classes_)
        new_data = pd.concat([new_data, genre_encoded], axis=1)
        print("After genre encoding:")
        print(new_data)

        # Drop original categorical columns
        new_data = new_data.drop(['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)
        print("After dropping categorical columns:")
        print(new_data)

        # Ensure all columns match X
        for col in X_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        new_data = new_data[X_columns]
        print("After aligning columns with X:")
        print(new_data)

        # Predict
        prediction = model.predict(new_data)[0]
        print(f"Prediction successful: {prediction}")
        return prediction
    except Exception as e:
        print(f"Error in predict_movie_rating: {e}")
        return None

# Step 10: GUI for Movie Rating Prediction
class MovieRatingGUI:
    def __init__(self, root, mlb, data, model, X_columns, r2, mse, y_test, y_pred_rf, importances):
        self.root = root
        self.mlb = mlb
        self.data = data
        self.model = model
        self.X_columns = X_columns
        self.r2 = r2
        self.mse = mse
        self.y_test = y_test
        self.y_pred_rf = y_pred_rf
        self.importances = importances
        self.root.title("Movie Rating Prediction")
        self.root.geometry("800x600")

        # Create a canvas and scrollbar for the entire window
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Title
        tk.Label(self.scrollable_frame, text="Movie Rating Prediction", font=("Arial", 16, "bold"), fg="#333").pack(pady=10)

        # Model performance
        tk.Label(self.scrollable_frame, text=f"Model R² Score: {self.r2:.2f}", font=("Arial", 12), fg="#555").pack(pady=5)
        tk.Label(self.scrollable_frame, text=f"Model MSE: {self.mse:.2f}", font=("Arial", 12), fg="#555").pack(pady=5)

        # Input fields
        self.entries = {}
        self.genre_listbox = None

        # Genre (multi-select listbox)
        frame = tk.Frame(self.scrollable_frame)
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(frame, text="Genre (select multiple)", width=20, anchor="w").pack(side="left")
        self.genre_listbox = tk.Listbox(frame, selectmode="multiple", height=4, exportselection=False)
        for genre in mlb.classes_:
            self.genre_listbox.insert(tk.END, genre)
        self.genre_listbox.pack(side="right", expand=True, fill="x")
        # Pre-select "Drama"
        for idx, genre in enumerate(mlb.classes_):
            if genre == "Drama":
                self.genre_listbox.selection_set(idx)

        # Other fields
        fields = [
            ("Director", "Satyajit Ray"),
            ("Actor 1", "Soumitra Chatterjee"),
            ("Actor 2", "Sharmila Tagore"),
            ("Actor 3", "Unknown"),
            ("Year (e.g., 2023)", "2023"),
            ("Duration (minutes, e.g., 120)", "120"),
            ("Votes (e.g., 500)", "500")
        ]

        for field, default in fields:
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=field, width=20, anchor="w").pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side="right", expand=True, fill="x")
            self.entries[field] = entry

        # Predict button
        tk.Button(self.scrollable_frame, text="Predict Rating", command=self.predict, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=10)

        # Result label
        self.result_label = tk.Label(self.scrollable_frame, text="Predicted Rating: N/A", font=("Arial", 14), fg="#333")
        self.result_label.pack(pady=10)

        # Clear button
        tk.Button(self.scrollable_frame, text="Clear Fields", command=self.clear_fields, bg="#f44336", fg="white", font=("Arial", 12)).pack(pady=10)

        # Output area with scrollbar
        tk.Label(self.scrollable_frame, text="Console Output", font=("Arial", 14, "bold")).pack(pady=10)
        output_frame = tk.Frame(self.scrollable_frame)
        output_frame.pack(fill="both", padx=10, pady=5)
        self.output_text = tk.Text(output_frame, height=10, width=80, wrap="word")
        output_scrollbar = tk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        output_scrollbar.pack(side="right", fill="y")

        # Redirect print statements to the output_text widget
        sys.stdout = RedirectText(self.output_text)

        # Create the combined plot
        self.fig, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        self.fig.suptitle('Movie Rating Prediction Analysis', fontsize=16)

        # Plot 1: Distribution of Movie Ratings
        sns.histplot(data['Rating'], bins=20, kde=True, ax=self.axes[0])
        self.axes[0].set_title('Distribution of Movie Ratings')
        self.axes[0].set_xlabel('Rating')
        self.axes[0].set_ylabel('Frequency')

        # Plot 2: Top 10 Feature Importances
        self.importances.sort_values(ascending=False).head(10).plot(kind='bar', ax=self.axes[1])
        self.axes[1].set_title('Top 10 Feature Importances (Random Forest)')
        self.axes[1].set_xlabel('Features')
        self.axes[1].set_ylabel('Importance')
        self.axes[1].tick_params(axis='x', rotation=45)

        # Plot 3: Predicted vs Actual Ratings
        self.axes[2].scatter(self.y_test, self.y_pred_rf, alpha=0.5)
        self.axes[2].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        self.axes[2].set_xlabel('Actual Rating')
        self.axes[2].set_ylabel('Predicted Rating')
        self.axes[2].set_title('Predicted vs Actual Ratings (Random Forest)')

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Embed the plot in the Tkinter window
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.scrollable_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(pady=10)

    def predict(self):
        try:
            # Get selected genres
            selected_indices = self.genre_listbox.curselection()
            genres = [self.genre_listbox.get(idx) for idx in selected_indices]
            if not genres:
                messagebox.showerror("Input Error", "At least one genre must be selected!")
                return

            # Get other input values
            director = self.entries["Director"].get().strip()
            actor1 = self.entries["Actor 1"].get().strip()
            actor2 = self.entries["Actor 2"].get().strip()
            actor3 = self.entries["Actor 3"].get().strip()
            year = int(self.entries["Year (e.g., 2023)"].get().strip())
            duration = int(self.entries["Duration (minutes, e.g., 120)"].get().strip())
            votes = int(self.entries["Votes (e.g., 500)"].get().strip())

            # Validate inputs
            if not director or not actor1 or not actor2 or not actor3:
                messagebox.showerror("Input Error", "All fields must be filled!")
                return
            if year < 1900 or year > 2030:
                messagebox.showerror("Input Error", "Year must be between 1900 and 2030!")
                return
            if duration <= 0:
                messagebox.showerror("Input Error", "Duration must be positive!")
                return
            if votes < 0:
                messagebox.showerror("Input Error", "Votes cannot be negative!")
                return

            # Predict rating
            predicted_rating = predict_movie_rating(
                genres=genres,
                director=director,
                actor1=actor1,
                actor2=actor2,
                actor3=actor3,
                year=year,
                duration=duration,
                votes=votes,
                mlb=self.mlb,
                data=self.data,
                model=self.model,
                X_columns=self.X_columns
            )

            if predicted_rating is not None:
                self.result_label.config(text=f"Predicted Rating: {predicted_rating:.2f}")
            else:
                messagebox.showerror("Prediction Error", "Failed to predict rating. Check console output for details.")
        except ValueError:
            messagebox.showerror("Input Error", "Year, Duration, and Votes must be numeric!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def clear_fields(self):
        self.genre_listbox.selection_clear(0, tk.END)
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="Predicted Rating: N/A")

# Step 11: Launch the GUI
root = tk.Tk()
app = MovieRatingGUI(root, mlb, data_with_categoricals, rf_model, X.columns, r2_score(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_rf), y_test, y_pred_rf, importances)
root.mainloop()
from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ✅ Fix matplotlib thread issue
import matplotlib.pyplot as plt
import joblib
import os

app = Flask(__name__)

# =========================
# PATHS
# =========================
DATA_PATH = "cleaned_waste_data.csv"
MODEL_PATH = "model.pkl"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Convert to numeric (safe)
cols = ['waste_total', 'recycling_rate', 'sustainability_index']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(0)

# =========================
# KPIs
# =========================
total_waste = int(df['waste_total'].sum())
avg_recycling = round(df['recycling_rate'].mean(), 2)
avg_sustainability = round(df['sustainability_index'].mean(), 2)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# CREATE CHART
# =========================
def create_chart():
    if not os.path.exists("static"):
        os.makedirs("static")

    region_waste = df.groupby('region')['waste_total'].sum().sort_values(ascending=False)

    plt.figure()
    region_waste.plot(kind='bar')
    plt.title("Waste by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Waste")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("static/chart.png")
    plt.close()

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    create_chart()
    return render_template(
        'index.html',
        total_waste=total_waste,
        avg_recycling=avg_recycling,
        avg_sustainability=avg_sustainability,
        prediction=None
    )

# =========================
# PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        recycling = float(request.form['recycling_rate'])
        dump = float(request.form['dump_rate'])
        plastic = float(request.form['plastic_pct'])

        # ✅ Correct feature mapping
        input_df = pd.DataFrame([{
            'recycling_rate': recycling,
            'plastic_pct': plastic,
            'dump_rate': dump
        }])

        # 🔥 Ensure correct order
        input_df = input_df.reindex(columns=model.feature_names_in_)

        prediction = model.predict(input_df)[0]

        return render_template(
            'index.html',
            total_waste=total_waste,
            avg_recycling=avg_recycling,
            avg_sustainability=avg_sustainability,
            prediction=prediction
        )

    except Exception as e:
        print("ERROR:", e)
        return "Prediction Error. Check terminal."

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    print("App is running...")
    app.run(debug=True)
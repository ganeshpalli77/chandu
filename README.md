# HR Employee Attrition Analysis - Complete Project

This project contains a comprehensive implementation of HR employee attrition analysis using machine learning and interactive dashboards.

## 📁 Project Files

### Main Analysis Files

1. **`hr_attrition_complete.py`** - Complete analysis pipeline
   - Data loading and exploration
   - Data cleaning and preprocessing
   - Database storage (SQLite)
   - Machine Learning model training (Random Forest)
   - Model evaluation with feature importance
   - Model persistence for deployment

2. **`streamlit_dashboard.py`** - Interactive web dashboard
   - Real-time data filtering
   - Key Performance Indicators (KPIs)
   - Multiple visualization tabs
   - Attrition prediction tool
   - Beautiful, user-friendly interface

3. **`run hr_attrition_analysis.py`** - Quick analysis script (simplified version)

4. **`main.py`** - Original extracted code

### Data Files (Generated)

- `cleaned_hr_data.csv` - Cleaned dataset
- `workforce.db` - SQLite database
- `attrition_model.pkl` - Trained ML model
- `model_features.pkl` - Feature names for predictions
- `feature_importance.png` - Feature importance visualization

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. All required packages are:

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
sqlalchemy
streamlit
```

### Installation

Install all dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy streamlit
```

Or if you're using a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy streamlit
```

### Required Dataset

Ensure you have the dataset file in the same directory:
- **`WA_Fn-UseC_-HR-Employee-Attrition.csv`**

## 📊 Usage

### 1. Run Complete Analysis

Execute the comprehensive analysis pipeline:

```bash
python hr_attrition_complete.py
```

This will:
- ✅ Load and explore the dataset
- ✅ Clean and preprocess the data
- ✅ Store data in SQLite database
- ✅ Train Random Forest model
- ✅ Evaluate model performance
- ✅ Save model and generate visualizations

**Expected Output:**
- Cleaned dataset (`cleaned_hr_data.csv`)
- SQLite database (`workforce.db`)
- Trained model (`attrition_model.pkl`)
- Feature names (`model_features.pkl`)
- Feature importance chart (`feature_importance.png`)

### 2. Launch Interactive Dashboard

Run the Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

This will open an interactive web application in your browser with:
- 📊 **Key Performance Indicators**: Total employees, attrition rate, average salary, job satisfaction
- 🔍 **Dynamic Filters**: Filter by department, overtime, gender, age range
- 📈 **Multiple Visualization Tabs**:
  - Attrition Analysis
  - Income Analysis
  - Demographics
  - Work Patterns
- 🔮 **Attrition Prediction Tool**: Predict attrition risk for new employees

## 📈 Model Performance

The Random Forest Classifier achieves:
- **Accuracy**: ~83%
- **Balanced training** with class weights to handle imbalanced data
- **Feature importance analysis** to identify key attrition drivers

### Top Features Influencing Attrition:
1. OverTime
2. Monthly Income
3. Age
4. Years at Company
5. Distance From Home
6. Job Satisfaction
7. Work-Life Balance
8. Environment Satisfaction

## 🎯 Key Insights

Based on the analysis:
- 📈 **OverTime** is the strongest predictor of employee attrition
- 💰 **Lower income** correlates with higher attrition rates
- 👨‍💼 **Younger employees** tend to have higher attrition
- 🏠 **Distance from home** impacts retention
- 😊 **Job satisfaction** and **work-life balance** are critical factors

## 🛠️ Troubleshooting

### Common Issues

1. **"Dataset not found" error**
   - Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the same directory

2. **"Model not found" error in dashboard**
   - Run `hr_attrition_complete.py` first to train and save the model

3. **Import errors**
   - Install all required packages: `pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy streamlit`

4. **Streamlit not opening**
   - Check if another instance is running
   - Try a different port: `streamlit run streamlit_dashboard.py --server.port 8502`

## 📚 Project Structure

```
Untitled11/
│
├── hr_attrition_complete.py      # Main analysis script
├── streamlit_dashboard.py        # Interactive dashboard
├── run hr_attrition_analysis.py # Quick analysis
├── main.py                       # Original code
├── README.md                     # This file
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Input dataset
│
└── Generated Files:
    ├── cleaned_hr_data.csv       # Cleaned dataset
    ├── workforce.db              # SQLite database
    ├── attrition_model.pkl       # Trained model
    ├── model_features.pkl        # Feature names
    └── feature_importance.png    # Visualization
```

## 🔄 Workflow

1. **Data Collection** → Load HR dataset
2. **Data Cleaning** → Remove duplicates, handle missing values, encode categories
3. **Feature Engineering** → Select relevant features, create dummy variables
4. **Model Training** → Train Random Forest classifier with balanced classes
5. **Model Evaluation** → Assess performance metrics and feature importance
6. **Deployment** → Save model and create interactive dashboard
7. **Prediction** → Use dashboard to predict attrition risk for new employees

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all files are in the correct directory
3. Ensure all dependencies are installed
4. Check Python version compatibility (3.8+)

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ `hr_attrition_complete.py` runs without errors
- ✅ All output files are generated
- ✅ Streamlit dashboard opens in your browser
- ✅ You can interact with filters and visualizations
- ✅ Prediction tool returns results

## 📝 Notes

- The model uses **Random Forest** for robust performance
- **Class balancing** is applied to handle imbalanced attrition data
- **Feature importance** helps identify key retention factors
- The dashboard provides **real-time filtering** for detailed analysis
- All visualizations are **automatically updated** based on filters

---

**Built with** ❤️ **using Python, Scikit-learn, Pandas, and Streamlit**

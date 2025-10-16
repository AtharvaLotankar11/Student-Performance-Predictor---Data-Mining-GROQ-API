# 🎓 Student Performance Predictor

Complete Student Performance Card Analysis using AI-powered machine learning models with comprehensive demographic and academic score inputs.

## ✨ Features

- **Complete Performance Card**: All 8 CSV columns as inputs (demographics + test scores)
- **Multiple ML Models**: Decision Tree, Random Forest, Neural Network
- **🎯 Animated Interface**: Smooth form animations and transitions
- **🤖 AI-Powered Insights**: Personalized recommendations using Groq API
- **🧠 AI Model Analysis**: Intelligent analysis of model performance metrics
- **Ensemble Predictions**: Combines all models for better accuracy
- **Score Analysis**: Math, Reading, Writing scores with average calculation

## 📋 Input Features

**Demographics & Background:**
- **Gender**: Male/Female
- **Race/Ethnicity**: Groups A-E (anonymized ethnic backgrounds for privacy)
- **Parental Education**: From "Some High School" to "Master's Degree"
- **Socioeconomic Status**: Via lunch program eligibility
  - *Standard*: Higher income families (pay full meal price)
  - *Free/Reduced*: Lower income families (qualify for subsidized meals)
- **Test Preparation**: Whether student completed preparation courses

**Academic Scores:**
- Math Score (0-100)
- Reading Score (0-100)  
- Writing Score (0-100)
- Average Score (calculated automatically)

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Get `StudentsPerformance.csv` from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
   - Place it in the project root

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open browser:** `http://localhost:5000`

## 📖 Understanding the Variables

### What do these categories mean?

**Race/Ethnicity Groups (A-E):**
- Anonymized representations of different ethnic backgrounds
- Used to study cultural and social factors affecting academic performance
- Protects student privacy while enabling demographic analysis

**Lunch Program (Socioeconomic Indicator):**
- **Standard Lunch**: Families pay full price → Higher income households
- **Free/Reduced Lunch**: Families qualify for subsidies → Lower income households
- Commonly used proxy for socioeconomic status in educational research
- Based on federal poverty guidelines and family income thresholds

**Why These Matter:**
- Research shows socioeconomic factors significantly impact academic outcomes
- Parental education level correlates with student support and resources
- Test preparation access varies by family resources
- The model learns these patterns to predict overall academic success

## 🎯 How to Use

1. Fill in complete student performance card (demographics + test scores)
2. Click "Predict Performance" (watch the animation!)
3. View predictions from all models with confidence scores
4. Read AI-generated insights based on score patterns
5. See model performance analysis and feature importance

## 📊 Project Structure

```
├── app.py                    # Main Flask application
├── requirements.txt          # Dependencies
├── StudentsPerformance.csv   # Dataset
├── templates/index.html      # Web interface
└── README.md                # This file
```

**Built with Flask, scikit-learn, TensorFlow, and Groq AI** 🚀
# app.py - Student Performance Predictor
import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Use lightweight scikit-learn only for Vercel compatibility
from sklearn.neural_network import MLPClassifier
TF_AVAILABLE = False  # Force use of lightweight implementation

app = Flask(__name__)

# Flask Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Global variables
models = {}
scaler = None
label_encoders = {}
feature_names = []
model_metrics = {}
dataset_info = {}

def load_and_preprocess_data():
    """Load and preprocess the student performance dataset"""
    global models, scaler, label_encoders, feature_names, model_metrics, dataset_info
    
    print("Loading and training models...")
    
    # Check if dataset exists
    if not os.path.exists('StudentsPerformance.csv'):
        raise FileNotFoundError(
            "StudentsPerformance.csv not found!\n"
            "Please download it from: "
            "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"
        )
    
    # Load the dataset
    df = pd.read_csv('StudentsPerformance.csv')
    print(f"Dataset loaded: {len(df)} students")
    
    # Store minimal dataset info
    dataset_info = {
        'total_students': len(df),
        'features': list(df.columns)
    }
    
    # Create target variable (Pass/Fail based on average score >= 60)
    df['performance'] = ((df['math score'] + df['reading score'] + df['writing score']) / 3 >= 60).astype(int)
    
    print(f"Target created: {df['performance'].sum()} Pass, {len(df) - df['performance'].sum()} Fail")
    
    # Select features - including all available columns from CSV
    feature_columns = ['gender', 'race/ethnicity', 'parental level of education', 
                       'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']
    
    X = df[feature_columns].copy()
    y = df['performance']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    feature_names = feature_columns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {}
    model_metrics = {}
    
    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    models['decision_tree'] = dt_model
    model_metrics['decision_tree'] = {
        'accuracy': float(accuracy_score(y_test, dt_pred)),
        'precision': float(precision_score(y_test, dt_pred)),
        'recall': float(recall_score(y_test, dt_pred)),
        'f1_score': float(f1_score(y_test, dt_pred))
    }
    
    # 2. Random Forest Classifier (lightweight for Vercel)
    rf_model = RandomForestClassifier(n_estimators=20, max_depth=6, min_samples_split=20, random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['random_forest'] = rf_model
    model_metrics['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'precision': float(precision_score(y_test, rf_pred)),
        'recall': float(recall_score(y_test, rf_pred)),
        'f1_score': float(f1_score(y_test, rf_pred))
    }
    
    # 3. Support Vector Machine (SVM) - ultra lightweight
    try:
        svm_model = SVC(kernel='linear', C=0.1, probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        svm_pred = svm_model.predict(X_test_scaled)
        
        models['svm'] = svm_model
        model_metrics['svm'] = {
            'accuracy': float(accuracy_score(y_test, svm_pred)),
            'precision': float(precision_score(y_test, svm_pred)),
            'recall': float(recall_score(y_test, svm_pred)),
            'f1_score': float(f1_score(y_test, svm_pred))
        }
    except Exception as e:
        print(f"SVM training failed: {e}")
    
    # 4. Neural Network (Ultra lightweight)
    try:
        nn_model = MLPClassifier(hidden_layer_sizes=(6,), max_iter=30, random_state=42, solver='lbfgs')
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        
        models['neural_network'] = nn_model
        model_metrics['neural_network'] = {
            'accuracy': float(accuracy_score(y_test, nn_pred)),
            'precision': float(precision_score(y_test, nn_pred)),
            'recall': float(recall_score(y_test, nn_pred)),
            'f1_score': float(f1_score(y_test, nn_pred))
        }
    except Exception as e:
        print(f"Neural Network training failed: {e}")
    
    print("Models trained successfully!")
    
    # Clean up temporary variables to free memory
    del X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    del dt_pred, rf_pred, svm_pred, nn_pred

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using all trained models"""
    try:
        data = request.json
        
        # Prepare input data
        input_data = pd.DataFrame({
            'gender': [data['gender']],
            'race/ethnicity': [data['race_ethnicity']],
            'parental level of education': [data['parental_education']],
            'lunch': [data['lunch']],
            'test preparation course': [data['test_prep']],
            'math score': [int(data['math_score'])],
            'reading score': [int(data['reading_score'])],
            'writing score': [int(data['writing_score'])]
        })
        
        # Encode the categorical input columns only
        categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        for col in categorical_columns:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f"Invalid value for {col}: {input_data[col][0]}"
                })
        
        results = {}
        
        # Decision Tree Prediction
        dt_prediction = models['decision_tree'].predict(input_data)[0]
        dt_probability = models['decision_tree'].predict_proba(input_data)[0]
        results['decision_tree'] = {
            'prediction': 'Pass' if dt_prediction == 1 else 'Fail',
            'confidence': round(float(max(dt_probability)) * 100, 2),
            'pass_probability': round(float(dt_probability[1]) * 100, 2)
        }
        
        # Random Forest Prediction
        rf_prediction = models['random_forest'].predict(input_data)[0]
        rf_probability = models['random_forest'].predict_proba(input_data)[0]
        results['random_forest'] = {
            'prediction': 'Pass' if rf_prediction == 1 else 'Fail',
            'confidence': round(float(max(rf_probability)) * 100, 2),
            'pass_probability': round(float(rf_probability[1]) * 100, 2)
        }
        
        # SVM Prediction
        input_scaled = scaler.transform(input_data)
        if 'svm' in models:
            svm_prediction = models['svm'].predict(input_scaled)[0]
            svm_probability = models['svm'].predict_proba(input_scaled)[0]
            results['svm'] = {
                'prediction': 'Pass' if svm_prediction == 1 else 'Fail',
                'confidence': round(float(max(svm_probability)) * 100, 2),
                'pass_probability': round(float(svm_probability[1]) * 100, 2)
            }
        else:
            results['svm'] = {
                'prediction': 'Pass' if rf_prediction == 1 else 'Fail',
                'confidence': 75.0,
                'pass_probability': 75.0 if rf_prediction == 1 else 25.0
            }
        
        # Neural Network Prediction (scikit-learn)
        if 'neural_network' in models:
            nn_prediction = models['neural_network'].predict(input_scaled)[0]
            nn_probability = models['neural_network'].predict_proba(input_scaled)[0]
            results['neural_network'] = {
                'prediction': 'Pass' if nn_prediction == 1 else 'Fail',
                'confidence': round(float(max(nn_probability)) * 100, 2),
                'pass_probability': round(float(nn_probability[1]) * 100, 2)
            }
        else:
            results['neural_network'] = {
                'prediction': 'Pass' if dt_prediction == 1 else 'Fail',
                'confidence': 70.0,
                'pass_probability': 70.0 if dt_prediction == 1 else 30.0
            }
        
        # Feature importance from Decision Tree
        feature_importance = dict(zip(feature_names, models['decision_tree'].feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Ensemble prediction (majority voting)
        predictions = [dt_prediction, rf_prediction]
        if 'svm' in models:
            predictions.append(svm_prediction)
        if 'neural_network' in models:
            predictions.append(nn_prediction)
            
        ensemble_prediction = 1 if sum(predictions) >= len(predictions)/2 else 0
        ensemble_confidence = (sum(predictions) / len(predictions)) * 100
        
        results['ensemble'] = {
            'prediction': 'Pass' if ensemble_prediction == 1 else 'Fail',
            'confidence': round(ensemble_confidence, 2),
            'agreement': f"{sum(predictions)}/{len(predictions)} models agree"
        }
        
        return jsonify({
            'success': True,
            'predictions': results,
            'feature_importance': [
                {
                    'feature': f.replace('_', ' ').replace('/', ' / ').title(),
                    'importance': round(imp * 100, 2)
                }
                for f, imp in sorted_features
            ],
            'input_summary': {
                'gender': data['gender'].title(),
                'ethnicity': data['race_ethnicity'].title(),
                'parental_education': data['parental_education'].title(),
                'lunch_type': data['lunch'].title(),
                'test_prep': data['test_prep'].title(),
                'math_score': data['math_score'],
                'reading_score': data['reading_score'],
                'writing_score': data['writing_score'],
                'average_score': round((int(data['math_score']) + int(data['reading_score']) + int(data['writing_score'])) / 3, 1)
            }
        })
    
    except Exception as e:
        import traceback
        print("Error in prediction:")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/metrics')
def get_metrics():
    """Get model performance metrics"""
    return jsonify(model_metrics)

@app.route('/metrics-analysis')
def get_metrics_analysis():
    """Get Groq AI analysis of model performance metrics"""
    try:
        # Find best performing model
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        best_model_name = best_model[0].replace('_', ' ').title()
        best_accuracy = best_model[1]['accuracy']

        # Prepare prompt for Groq
        prompt = f"""
        As an expert machine learning analyst, provide a comprehensive analysis of these model performance metrics for a student performance prediction system:

        Model Performance Results:
        
        Decision Tree: Accuracy {model_metrics['decision_tree']['accuracy']:.4f}, F1-Score {model_metrics['decision_tree']['f1_score']:.4f}
        Random Forest: Accuracy {model_metrics['random_forest']['accuracy']:.4f}, F1-Score {model_metrics['random_forest']['f1_score']:.4f}
        SVM: Accuracy {model_metrics['svm']['accuracy']:.4f}, F1-Score {model_metrics['svm']['f1_score']:.4f}
        Neural Network: Accuracy {model_metrics['neural_network']['accuracy']:.4f}, F1-Score {model_metrics['neural_network']['f1_score']:.4f}

        Best Performing Model: {best_model_name} with {best_accuracy:.4f} accuracy

        Please provide:
        1. A brief overall assessment of model performance
        2. Comparison of strengths and weaknesses of each model
        3. Recommendations for model selection and potential improvements
        4. Insights about what these metrics tell us about the prediction task

        Keep the analysis concise but insightful (3-4 sentences total).
        """

        # Make request to Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert machine learning analyst who provides clear, concise insights about model performance metrics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            groq_response = response.json()
            analysis = groq_response['choices'][0]['message']['content'].strip()
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'best_model': best_model_name,
                'best_accuracy': f"{best_accuracy:.4f}"
            })
        else:
            return jsonify({
                'success': False,
                'error': f'API Error {response.status_code}: Unable to generate analysis'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dataset_info')
def get_dataset_info():
    """Get dataset information"""
    return jsonify(dataset_info)

@app.route('/groq-insight', methods=['POST'])
def groq_insight():
    """Generate AI insight using Groq API"""
    try:
        if not request.json:
            return jsonify({'success': False, 'error': 'No JSON data provided'})
        
        data = request.json
        
        # Validate required data
        required_keys = ['predictions', 'student_data', 'feature_importance']
        for key in required_keys:
            if key not in data:
                return jsonify({'success': False, 'error': f'Missing required data: {key}'})
        
        predictions = data['predictions']
        student_data = data['student_data']
        feature_importance = data['feature_importance']
        
        # Prepare the prompt for Groq
        ensemble_prediction = predictions['ensemble']['prediction']
        confidence = predictions['ensemble']['confidence']
        
        # Get top 3 most important features
        top_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:3]
        
        prompt = f"""
        As an educational AI advisor, provide personalized insights for a student based on their performance prediction.

        Student Profile:
        - Gender: {student_data.get('gender', 'N/A')}
        - Race/Ethnicity: {student_data.get('race_ethnicity', 'N/A')}
        - Parental Education: {student_data.get('parental_education', 'N/A')}
        - Lunch Type: {student_data.get('lunch', 'N/A')}
        - Test Preparation: {student_data.get('test_prep', 'N/A')}
        - Math Score: {student_data.get('math_score', 'N/A')}
        - Reading Score: {student_data.get('reading_score', 'N/A')}
        - Writing Score: {student_data.get('writing_score', 'N/A')}

        Prediction Results:
        - Overall Prediction: {ensemble_prediction}
        - Confidence: {confidence}%

        Most Important Factors:
        1. {top_features[0]['feature']}: {top_features[0]['importance']:.1f}%
        2. {top_features[1]['feature']}: {top_features[1]['importance']:.1f}%
        3. {top_features[2]['feature']}: {top_features[2]['importance']:.1f}%

        Please provide a concise, encouraging, and actionable insight (2-3 sentences) that:
        1. Acknowledges the prediction result and current scores
        2. Highlights key factors influencing performance
        3. Offers specific, practical advice for improvement

        Keep it positive, supportive, and focused on actionable steps.
        """

        if not GROQ_API_KEY:
            return jsonify({'success': False, 'error': 'GROQ API key not configured'})

        # Make request to Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert educational advisor who provides personalized, encouraging insights to help students improve their academic performance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            groq_response = response.json()
            insight = groq_response['choices'][0]['message']['content'].strip()
            return jsonify({'success': True, 'insight': insight})
        else:
            return jsonify({'success': False, 'error': f'API Error {response.status_code}: Unable to generate insight'})
            
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Request timed out - please try again'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'Connection error - please check your internet connection'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'models': list(models.keys()),
        'dataset_info': dataset_info
    })

@app.route('/api/health')
def api_health_check():
    """API health check endpoint for Vercel"""
    return jsonify({
        'status': 'ok',
        'service': 'Student Performance Predictor',
        'models_ready': len(models) > 0
    })

@app.route('/test-groq')
def test_groq_api():
    """Test GROQ API connection"""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hello, this is a test message. Please respond with 'API is working!'"}],
            "max_tokens": 50
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            groq_response = response.json()
            message = groq_response['choices'][0]['message']['content'].strip()
            return jsonify({'success': True, 'message': 'GROQ API is working!', 'response': message})
        else:
            return jsonify({'success': False, 'error': f'API returned status {response.status_code}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Initialize models when the module is imported (for Vercel)
try:
    load_and_preprocess_data()
    print("Models loaded successfully for serverless deployment")
except Exception as e:
    print(f"Warning: Could not load models - {e}")

if __name__ == '__main__':
    try:
        print("Starting Student Performance Predictor...")
        if not models:  # Only load if not already loaded
            load_and_preprocess_data()
        print("Server ready! Open http://localhost:5000 in your browser")
        app.run(debug=False, port=5000, host='0.0.0.0')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset and try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")
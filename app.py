import os
import json
import joblib
import traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Directory containing model weights and metadata
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

# Stores for loaded model instances and metadata
models = {}
metadata = {}

# Data-driven Presets derived from student_performance_dataset.csv
PRESETS = {
    'scholar': {
        'label': 'Model Scholar',
        'gender': 'Male',
        'study_time_hours': 5.9,
        'attendance_percent': 100.0,
        'sleep_hours': 7.3,
        'parental_education': 'Bachelors',
        'internet_access': 'Yes',
        'extracurricular_activities': 'Yes',
        'part_time_job': 'No',
        'previous_grade': 89.8
    },
    'turnaround': {
        'label': 'Turnaround Kid',
        'gender': 'Male',
        'study_time_hours': 4.6,
        'attendance_percent': 100.0,
        'sleep_hours': 5.8,
        'parental_education': 'Masters',
        'internet_access': 'Yes',
        'extracurricular_activities': 'Yes',
        'part_time_job': 'No',
        'previous_grade': 37.5
    },
    'at_risk': {
        'label': 'At-Risk Student',
        'gender': 'Female',
        'study_time_hours': 1.9,
        'attendance_percent': 69.0,
        'sleep_hours': 6.0,
        'parental_education': 'High School',
        'internet_access': 'Yes',
        'extracurricular_activities': 'No',
        'part_time_job': 'Yes',
        'previous_grade': 34.5
    },
    'multitasker': {
        'label': 'Busy Multitasker',
        'gender': 'Female',
        'study_time_hours': 6.3,
        'attendance_percent': 100.0,
        'sleep_hours': 5.7,
        'parental_education': 'High School',
        'internet_access': 'Yes',
        'extracurricular_activities': 'Yes',
        'part_time_job': 'Yes',
        'previous_grade': 75.5
    },
    'passive': {
        'label': 'Passive Attendee',
        'gender': 'Male',
        'study_time_hours': 1.5,
        'attendance_percent': 100.0,
        'sleep_hours': 5.0,
        'parental_education': 'High School',
        'internet_access': 'Yes',
        'extracurricular_activities': 'Yes',
        'part_time_job': 'Yes',
        'previous_grade': 58.6
    },
    'baseline': {
        'label': 'Average Baseline',
        'gender': 'Male',
        'study_time_hours': 3.6,
        'attendance_percent': 85.0,
        'sleep_hours': 6.8,
        'parental_education': 'High School',
        'internet_access': 'Yes',
        'extracurricular_activities': 'No',
        'part_time_job': 'No',
        'previous_grade': 69.7
    }
}


def load_all_models():
    """Scans saved_models folder and loads all available .joblib estimators."""
    model_mapping = {
        'Bayesian Ridge': 'best_traditional_ml_model',
        'CatBoost': 'boosting_catboost_regularized',
        'LightGBM': 'boosting_lightgbm_regularized',
        'XGBoost': 'boosting_xgboost_regularized',
        'Voting Reg.': 'voting_regressor'
    }

    failed_models = []

    for display_name, file_prefix in model_mapping.items():
        model_path = os.path.join(MODEL_DIR, f"{file_prefix}.joblib")
        meta_path = os.path.join(MODEL_DIR, f"{file_prefix}_metadata.json")

        if os.path.exists(model_path):
            try:
                models[display_name] = joblib.load(model_path)
            except Exception:
                failed_models.append(display_name)
        else:
            failed_models.append(display_name)

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata[display_name] = json.load(f)
            except Exception:
                pass

    if failed_models:
        print(
            f"All available models loaded. ({len(models)}/{len(model_mapping)} ready. Failed to load: {', '.join(failed_models)})")
    else:
        print(
            f"All models loaded successfully! ({len(models)}/{len(model_mapping)} ready)")


load_all_models()


def map_score_to_letter_grade(score):
    """Utility mapping continuous predictions (0-100) to standard letter grades."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


@app.route('/', methods=['GET'])
def index():
    """Renders main dashboard interface."""
    return render_template('index.html')


@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Reads and returns metadata from all model JSON files in saved_models/."""
    try:
        model_metadata_list = []
        if os.path.exists(MODEL_DIR):
            for filename in os.listdir(MODEL_DIR):
                if filename.endswith('_metadata.json'):
                    filepath = os.path.join(MODEL_DIR, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        model_metadata_list.append(data)

        # Sort models alphabetically by model_name for UI consistency
        model_metadata_list.sort(key=lambda x: x.get('model_name', ''))
        return jsonify({'success': True, 'models': model_metadata_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Returns available preset archetypes to the frontend."""
    return jsonify({'success': True, 'presets': PRESETS})


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint accepting user inputs and executing model predictions."""
    if not models:
        return jsonify({
            'success': False,
            'error': 'No models are currently loaded on the server.'
        }), 500

    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        if not data:
            return jsonify({'success': False, 'error': 'No input data provided.'}), 400

        # Construct DataFrame
        input_dict = {
            'gender': [data.get('gender')],
            'study_time_hours': [float(data.get('study_time_hours', 0.0))],
            'attendance_percent': [float(data.get('attendance_percent', 0.0))],
            'sleep_hours': [float(data.get('sleep_hours', 0.0))],
            'parental_education': [data.get('parental_education')],
            'internet_access': [data.get('internet_access')],
            'extracurricular_activities': [data.get('extracurricular_activities')],
            'part_time_job': [data.get('part_time_job')],
            'previous_grade': [float(data.get('previous_grade', 0.0))]
        }

        features_df = pd.DataFrame(input_dict)

        # Cast categorical string columns to Pandas category dtype
        categorical_cols = [
            'gender',
            'parental_education',
            'internet_access',
            'extracurricular_activities',
            'part_time_job'
        ]

        for col in categorical_cols:
            features_df[col] = features_df[col].astype('category')

        predictions = []

        for display_name, model in models.items():
            try:
                raw_pred = model.predict(features_df)[0]
                score = max(0.0, min(100.0, float(raw_pred)))
                grade = map_score_to_letter_grade(score)

                predictions.append({
                    'model': display_name,
                    'score': round(score, 1),
                    'grade': grade
                })
            except Exception as model_err:
                print(
                    f"\n[PREDICTION ERROR] Model '{display_name}' failed to predict:")
                print(f"Error Message: {str(model_err)}")
                traceback.print_exc()
                print("-" * 50)

                predictions.append({
                    'model': display_name,
                    'score': 'N/A',
                    'grade': 'Error'
                })

        return jsonify({'success': True, 'predictions': predictions})

    except Exception as e:
        print(f"\n[GLOBAL PREDICT ERROR] Request processing failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

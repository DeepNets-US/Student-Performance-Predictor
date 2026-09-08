import os
import io
import json
import joblib
import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Directory containing model weights and metadata
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

# Stores for loaded model instances and metadata
models = {}
metadata = {}

# Expected dataset schema definition and valid bounds/categories
REQUIRED_SCHEMA = {
    'gender': {'type': 'categorical', 'values': ['Male', 'Female']},
    'parental_education': {'type': 'categorical', 'values': ['High School', 'Bachelors', 'Masters', 'PhD', 'Not Disclosed']},
    'internet_access': {'type': 'categorical', 'values': ['Yes', 'No']},
    'extracurricular_activities': {'type': 'categorical', 'values': ['Yes', 'No']},
    'part_time_job': {'type': 'categorical', 'values': ['Yes', 'No']},
    'study_time_hours': {'type': 'numeric', 'min': 0.0, 'max': 24.0},
    'attendance_percent': {'type': 'numeric', 'min': 0.0, 'max': 100.0},
    'sleep_hours': {'type': 'numeric', 'min': 0.0, 'max': 24.0},
    'previous_grade': {'type': 'numeric', 'min': 0.0, 'max': 100.0}
}

# Categorical column definitions
CATEGORICAL_COLS = [
    'gender',
    'parental_education',
    'internet_access',
    'extracurricular_activities',
    'part_time_job'
]

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


def prepare_dataframe(df):
    """Converts categorical columns to Pandas 'category' dtypes for models."""
    df_formatted = df.copy()

    unwanted_features = [
        col for col in df.columns if col not in REQUIRED_SCHEMA.keys()]
    if unwanted_features:
        print('Dropping Columns:', unwanted_features)
        df_formatted = df_formatted.drop(columns=unwanted_features)

    for col in CATEGORICAL_COLS:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].astype('category')

    print(df_formatted.info())
    return df_formatted


def validate_dataframe(df):
    """
    Validates dataframe columns, data types, nulls, and value constraints.
    Returns exact row indices and offending values in error messages.
    """
    errors = []

    # 1. Check for missing required columns
    missing_cols = [col for col in REQUIRED_SCHEMA if col not in df.columns]
    if missing_cols:
        return False, [f"Missing required columns in dataset: {', '.join(missing_cols)}"]

    # 2. Check for missing/null values with exact row locations
    for col in REQUIRED_SCHEMA:
        null_indices = df[df[col].isnull()].index.tolist()
        if null_indices:
            errors.append(
                f"Column '{col}' contains {len(null_indices)} null value(s) at row index(es): {null_indices}"
            )

    # 3. Validate individual column schema rules
    for col, rules in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            continue

        if rules['type'] == 'categorical':
            # Extract non-null values that fall outside allowed categories
            non_null_series = df[col].dropna().astype(str)
            invalid_mask = ~non_null_series.isin(rules['values'])
            invalid_rows = non_null_series[invalid_mask]

            if not invalid_rows.empty:
                invalid_summary = [
                    f"Row {idx}: '{val}'" for idx, val in invalid_rows.items()
                ]
                errors.append(
                    f"Column '{col}' contains invalid categorical values. "
                    f"Expected allowed values: {rules['values']}. "
                    f"Found {len(invalid_rows)} violation(s) -> {', '.join(invalid_summary)}"
                )

        elif rules['type'] == 'numeric':
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Column '{col}' must contain numeric values, but found dtype '{df[col].dtype}'."
                )
            else:
                min_bound, max_bound = rules['min'], rules['max']

                # Check for out-of-bound lower values
                too_low = df[df[col] < min_bound][col]
                if not too_low.empty:
                    low_summary = [
                        f"Row {idx}: {val}" for idx, val in too_low.items()]
                    errors.append(
                        f"Column '{col}' has {len(too_low)} value(s) below min bound ({min_bound}). "
                        f"Violations -> {', '.join(low_summary)}"
                    )

                # Check for out-of-bound upper values
                too_high = df[df[col] > max_bound][col]
                if not too_high.empty:
                    high_summary = [
                        f"Row {idx}: {val}" for idx, val in too_high.items()]
                    errors.append(
                        f"Column '{col}' has {len(too_high)} value(s) above max bound ({max_bound}). "
                        f"Violations -> {', '.join(high_summary)}"
                    )

    if errors:
        return False, errors
    return True, []


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

        features_df = prepare_dataframe(pd.DataFrame(input_dict))
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


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Endpoint for uploading CSV datasets, schema validation, and running loaded models."""
    if not models:
        return jsonify({
            'success': False,
            'error': 'No models are currently loaded on the server.'
        }), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file attachment found in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file.'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload a CSV file.'}), 400

    try:
        # Load uploaded file into pandas
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF-8")))

        # Schema Validation
        is_valid, validation_errors = validate_dataframe(df)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'CSV Schema Validation Failed.',
                'details': validation_errors
            }), 400

        # Cast categories for model input
        formatted_df = prepare_dataframe(df)

        aggregated_predictions = []
        batch_row_results = []

        # Predict across the whole batch for each model
        for display_name, model in models.items():
            try:
                raw_preds = model.predict(formatted_df)
                clipped_preds = np.clip(raw_preds, 0.0, 100.0)
                avg_score = round(float(np.mean(clipped_preds)), 1)

                aggregated_predictions.append({
                    'model': display_name,
                    'score': avg_score,
                    'grade': map_score_to_letter_grade(avg_score)
                })
            except Exception as model_err:
                print(
                    f"[BATCH PREDICT ERROR] '{display_name}' failed: {str(model_err)}")
                aggregated_predictions.append({
                    'model': display_name,
                    'score': 'N/A',
                    'grade': 'Error'
                })

        # Calculate row-by-row outputs for response details
        for idx, row in formatted_df.iterrows():
            row_input = row.to_dict()
            row_preds = {}
            single_row_df = formatted_df.iloc[[idx]]

            for display_name, model in models.items():
                try:
                    pred_val = float(model.predict(single_row_df)[0])
                    score = round(max(0.0, min(100.0, pred_val)), 1)
                    row_preds[display_name] = {
                        'score': score,
                        'grade': map_score_to_letter_grade(score)
                    }
                except Exception:
                    row_preds[display_name] = {
                        'score': 'N/A', 'grade': 'Error'}

            batch_row_results.append({
                'row_index': idx + 1,
                'inputs': row_input,
                'predictions': row_preds
            })

        return jsonify({
            'success': True,
            'total_records': len(df),
            'aggregated_predictions': aggregated_predictions,
            'predictions': aggregated_predictions,
            'sample_row_predictions': batch_row_results[:20],
            'full_row_predictions': batch_row_results  # Included for complete CSV export
        })

    except Exception as e:
        print(f"[CSV PROCESSING ERROR]: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Failed to process file: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

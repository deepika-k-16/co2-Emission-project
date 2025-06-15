
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid
app = Flask(__name__)

model_data = joblib.load('clean_rf_model.pkl')
#print(type(model_data))
#print(model_data)

model = model_data['model']

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='ISO-8859-1')

            # Preprocess
            encoder = model_data['encoder']
            cat_features = model_data['categorical_features']
            num_features = model_data['numerical_features']

            X_num = df[num_features]
            X_cat = df[cat_features]
            X_cat_encoded = encoder.transform(X_cat)
            encoded_feature_names = encoder.get_feature_names_out(cat_features)
            X_cat_df = pd.DataFrame(X_cat_encoded.toarray(), columns=encoded_feature_names, index=df.index)
            X_final = pd.concat([X_num, X_cat_df], axis=1)

            predictions = model.predict(X_final)
            df['Predicted_CO2_Emissions'] = predictions

            # Generate unique filenames
            session_id = str(uuid.uuid4())
            csv_path = f'temp/{session_id}_predicted.csv'
            pdf_path = f'temp/{session_id}_report.pdf'
            os.makedirs('temp', exist_ok=True)

            # Save CSV
            df.to_csv(csv_path, index=False)

            # Save PDF
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.drawString(100, 750, "CO₂ Emissions Prediction Report")
            c.drawString(100, 730, f"Number of records: {len(df)}")
            c.drawString(100, 710, f"Average Predicted CO₂ Emission: {np.mean(predictions):.2f}")
            c.save()

            return render_template('prediction.html', csv_file=csv_path, report_file=pdf_path)
    return render_template('prediction.html')

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)



@app.route('/download_csv', methods=['POST'])
def download_csv():
    csv_data = request.form['csv_data']
    output = BytesIO()
    output.write(csv_data.encode())
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predicted_output.csv')

@app.route('/download_report', methods=['POST'])
def download_report():
    report_data = request.files['report']
    return send_file(report_data, mimetype='application/pdf', as_attachment=True, download_name='CO2_Report.pdf')

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    if request.method == 'POST':
        try:
            # Separate numeric and categorical inputs
            num_features = [
                float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4']),
                float(request.form['feature5']),
                float(request.form['feature6']),
            ]
            cat_features = [
                request.form['feature7'],  # Make
                request.form['feature8'],  # Model
                request.form['feature9'],
                request.form['vehicle_class'],  # Fuel Type
            ]
            


            # Convert to DataFrame to match training format
            df_num = pd.DataFrame([num_features], columns=model_data['numerical_features'])
            df_cat = pd.DataFrame([cat_features], columns=model_data['categorical_features'])

            # Encode categorical features
            encoder = model_data['encoder']
            cat_encoded = encoder.transform(df_cat)
            cat_encoded_df = pd.DataFrame(cat_encoded.toarray(), columns=encoder.get_feature_names_out())

            # Combine all features
            final_input = pd.concat([df_num, cat_encoded_df], axis=1)

            # Predict
            prediction = model.predict(final_input)[0]
            return render_template('calculate.html', prediction=round(prediction, 2))
        except Exception as e:
            return render_template('calculate.html', error=f"Invalid input. {str(e)}")
    return render_template('calculate.html')

if __name__ == '__main__':
    app.run(debug=True)
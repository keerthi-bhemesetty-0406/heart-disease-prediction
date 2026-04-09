❤️ Heart Guard: ML Heart Disease Prediction App
A full-stack web application that uses Machine Learning to predict the likelihood of heart disease based on health metrics.
This project is deployed on Render using a Flask backend and a PostgreSQL database.

🚀 Live Demo
Link:https://heart-disease-prediction-jiyf.onrender.com

🛠️ Tech Stack
Frontend: HTML5, CSS3, JavaScript (Responsive UI)

Backend: Flask (Python)

Database: PostgreSQL (via Render)

Machine Learning: * Model: Stacking Hybrid Model (Random Forest + XGBoost)

Preprocessing: StandardScaler (Scikit-Learn)

Deployment: Render, GitHub

✨ Key Features
User Authentication: Secure Signup and Login system using Flask-SQLAlchemy and Werkzeug password hashing.

Persistent Data: User accounts and prediction history are stored permanently in a PostgreSQL database.

Real-time Prediction: Uses a trained .pkl model to provide instant health risk assessments.

Responsive Design: Works on both mobile and desktop browsers.

📁 Project Structure
Plaintext
heart-disease-prediction/
├── app.py                     
├── heart_disease_stack_model.pkl 
├── scaler.pkl                  
├── requirements.txt            
├── templates/                  


⚙️ How to Run Locally
Clone the repository:
git clone https://github.com/keerthi-bhemesetty-0406/heart-disease-prediction.git

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open http://127.0.0.1:5000 in your browser.

🧠 Model Information
The model was trained on the UCI Heart Disease dataset. 
It utilizes a Stacking Classifier to combine the strengths of Random Forest and XGBoost, achieving high accuracy in detecting cardiovascular risks.

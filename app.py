from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib 
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Security key for session handling and flash messages
app.secret_key = 'super_secret_heartguard_key' 

# Database Configuration (SQLite)
import os

# Get the Database URL from Render's Environment Variables
db_url = os.environ.get('DATABASE_URL')

if db_url:
    # Fix for SQLAlchemy: Render gives 'postgres://', but we need 'postgresql://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    # If running locally on your laptop, use SQLite
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'heartguard.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Database Model ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='owner', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # The Foreign Key links this prediction to a specific User ID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(20))
    result = db.Column(db.String(50))
    prob = db.Column(db.Float)
    date = db.Column(db.String(50))

# Initialize database file automatically
with app.app_context():
    db.create_all()

# --- Routes ---

@app.route('/')
def index():
    """Landing Page"""
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Basic Validation
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered!', 'danger')
            return redirect(url_for('signup'))

        # Securely hash the password
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Save to Database
        new_user = User(fullname=fullname, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        # MODERN FLOW: Automatic Login after Signup
        session['user_id'] = new_user.id
        session['user_name'] = new_user.fullname
        
        flash('Account created! Welcome to HeartGuard AI.', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        # Check credentials
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.fullname
            print(f"SUCCESS: Redirecting {user.fullname} to dashboard")
            return redirect(url_for('dashboard'))
        else:
            print("FAILURE: User not found or wrong password")
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login')) # Adjust to your login route name
    
    current_user = User.query.get(user_id)
    # Fetch ONLY this user's records
    user_preds = Prediction.query.filter_by(user_id=user_id).all()

    # Calculate metrics for THIS user only
    total = len(user_preds)
    high_count = len([p for p in user_preds if p.result == "High Risk"])
    low_count = len([p for p in user_preds if p.result == "Low Risk"])
    last_res = user_preds[-1].result if total > 0 else "No predictions yet"

    return render_template('dashboard.html',
                           user=current_user, 
                           total1=total, 
                           high=high_count, 
                           low=low_count, 
                           last=last_res)


@app.route('/logout')
def logout():
    session.clear() # Clear all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

# 3. Route for the History Page
@app.route('/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    # Get the list of predictions, newest first
    user_preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.id.desc()).all()

    return render_template('history.html', predictions=user_preds)

# 4. Route for the Results Page (called after clicking "Predict")
@app.route('/results')
def results():
    return render_template('results.html')


model = joblib.load('heart_disease_stack_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    # 1. Ensure user is logged in
    user_id = session.get('user_id')
    if not user_id:
        # Redirect to your login route if not authenticated
        return redirect(url_for('login')) 

    if request.method == 'POST':
        try:
            current_date = datetime.now().strftime("%-m/%-d/%Y")
            # 2. Get raw inputs from form
            age = float(request.form.get('age'))
            sex = float(request.form.get('sex'))
            cp = float(request.form.get('chest_pain_type'))
            rbps = float(request.form.get('resting_bp_s'))
            chol = float(request.form.get('cholesterol'))
            fbs = float(request.form.get('fasting_blood_sugar'))
            ecg = float(request.form.get('resting_ecg'))
            mhr = float(request.form.get('max_heart_rate'))
            exang = float(request.form.get('exercise_angina'))
            oldpeak = float(request.form.get('oldpeak'))
            slope = float(request.form.get('st_slope'))

            # 3. Recreate the 16 features for the Scaler
            features_16 = [age, rbps, chol, mhr, oldpeak]
            features_16.append(1.0 if sex == 1 else 0.0)
            features_16.append(1.0 if cp == 2 else 0.0)
            features_16.append(1.0 if cp == 3 else 0.0)
            features_16.append(1.0 if cp == 4 else 0.0)
            features_16.append(1.0 if fbs == 1 else 0.0)
            features_16.append(1.0 if ecg == 1 else 0.0)
            features_16.append(1.0 if ecg == 2 else 0.0)
            features_16.append(1.0 if exang == 1 else 0.0)
            features_16.append(1.0 if slope == 1 else 0.0)
            features_16.append(1.0 if slope == 2 else 0.0)
            features_16.append(1.0 if slope == 3 else 0.0)

            # 4. Scale and Predict
            final_features = np.array([features_16])
            scaled_data = scaler.transform(final_features)
            prediction = model.predict(scaled_data)[0]
            
            if hasattr(model, "predict_proba"):
                prob = round(model.predict_proba(scaled_data)[0][1] * 100, 1)
            else:
                prob = 100.0 if prediction == 1 else 0.0
            
            result_label = "High Risk" if prediction == 1 else "Low Risk"
            sex_label = "Male" if sex == 1 else "Female"

            # 5. SAVE TO DATABASE (Linked to the unique User ID)
            new_prediction = Prediction(
                user_id=user_id,   # The foreign key link
                age=int(age),
                sex=sex_label,
                result=result_label,
                prob=prob,
                date=current_date
            )
            
            db.session.add(new_prediction)
            db.session.commit()

            # 6. RENDER RESULTS
            return render_template('results.html', result=result_label, prob=prob)

        except Exception as e:
            db.session.rollback() # Undo any partial DB changes on error
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    # Run the app in debug mode for easier development
    app.run(debug=True)

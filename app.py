import matplotlib
matplotlib.use('Agg')  # Backend pour les environnements sans GUI
import matplotlib.pyplot as plt
import base64
import datetime
from io import BytesIO, StringIO
import json
import os

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor




# === Config Flask ===
app = Flask(__name__)
app.secret_key = 'azertyuiop'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

from flask_migrate import Migrate

migrate = Migrate(app, db)

import json
def from_json_filter(s):
    return json.loads(s)

app.jinja_env.filters['from_json'] = from_json_filter

# === Modèles SQLAlchemy ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    target_column = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    mse = db.Column(db.Float)
    r2 = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    confusion_matrix = db.Column(db.Text)
    roc_curve = db.Column(db.Text)
    regression_plot = db.Column(db.Text)
    feature_importance = db.Column(db.Text)
    preview_data = db.Column(db.Text)


with app.app_context():
    db.create_all()


# === Options de Modèles ===
MODEL_OPTIONS = {
    'classification': {
        'random_forest': ('Random Forest', RandomForestClassifier()),
        'xgboost': ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        'logistic_regression': ('Logistic Regression', LogisticRegression(max_iter=1000))
    },
    'regression': {
        'random_forest': ('Random Forest', RandomForestRegressor()),
        'xgboost': ('XGBoost', XGBRegressor()),
        'linear_regression': ('Linear Regression', LinearRegression())
    }
}


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')



@app.route('/')
def index():
    return render_template('index.html', active_page='index')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Les mots de passe ne correspondent pas.")
            return redirect(url_for('signup'))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Nom d'utilisateur ou email déjà utilisé.")
            return redirect(url_for('signup'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Compte créé avec succès.")
        return redirect(url_for('login'))

    return render_template('auth/signup.html', active_page='signup')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username']
        password = request.form['password']

        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Connexion réussie.")
            return redirect(url_for('index'))
        else:
            flash("Identifiants incorrects.")
            return redirect(url_for('login'))

    return render_template('auth/login.html', active_page='login')


@app.route('/logout')
def logout():
    session.clear()
    flash("Déconnexion réussie.")
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_task = request.args.get('task', 'classification')

    if request.method == 'POST':
        try:
            if 'analyze' in request.form and 'dataset' not in request.files:
                df = pd.read_json(session['dataset'], orient='split')
                filename = session.get('filename', '')
            else:
                file = request.files['dataset']
                if file.filename == '':
                    flash("Aucun fichier sélectionné.")
                    return redirect(request.url)

                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)

                session['dataset'] = df.to_json(orient='split')
                session['filename'] = filename

            preview = df.head(10).to_dict('records')
            stats = {
                'rows': len(df),
                'columns': list(df.columns),
                'missing': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }

            if 'analyze' not in request.form:
                return render_template('upload.html',
                                       preview=preview,
                                       stats=stats,
                                       columns=list(df.columns),
                                       filename=filename,
                                       model_options=MODEL_OPTIONS[selected_task],
                                       all_model_options=MODEL_OPTIONS,
                                       selected_task=selected_task,
                                       active_page='upload')

            target = request.form['target']
            model_type = request.form['model']
            task = request.form['task']

            df.dropna(inplace=True)

            if task == 'classification' and df[target].dtype == 'object':
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target])

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_name, model = MODEL_OPTIONS[task][model_type]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            visualizations = {}

            if task == 'classification':
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                visualizations['confusion_matrix'] = fig_to_base64(fig)

                if len(set(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
                    ax.plot([0, 1], [0, 1], 'k--')
                    visualizations['roc_curve'] = fig_to_base64(fig)

            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                visualizations['regression_plot'] = fig_to_base64(fig)

            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(X.columns, model.feature_importances_)
                visualizations['feature_importance'] = fig_to_base64(fig)

            results = {}
            if task == 'classification':
                results['accuracy'] = accuracy_score(y_test, y_pred)
                results['report'] = classification_report(y_test, y_pred, output_dict=True)
            else:
                results['mse'] = mean_squared_error(y_test, y_pred)
                results['r2'] = r2_score(y_test, y_pred)

            analysis = Analysis(
                user_id=session['user_id'],
                filename=filename,
                model_type=model_name,
                task_type=task,
                target_column=target,
                accuracy=results.get('accuracy'),
                mse=results.get('mse'),
                r2=results.get('r2'),
                confusion_matrix=visualizations.get('confusion_matrix'),
                roc_curve=visualizations.get('roc_curve'),
                regression_plot=visualizations.get('regression_plot'),
                feature_importance=visualizations.get('feature_importance'),
                preview_data=json.dumps(preview)
            )
            db.session.add(analysis)
            db.session.commit()

            return render_template('results.html',
                                   task=task,
                                   model_name=model_name,
                                   results=results,
                                   visualizations=visualizations,
                                   preview=preview)

        except Exception as e:
            flash(f"Erreur: {str(e)}", 'error')
            return redirect(request.url)

    return render_template('upload.html',
                           model_options=MODEL_OPTIONS[selected_task],
                           all_model_options=MODEL_OPTIONS,
                           selected_task=selected_task,
                           active_page='upload')


# Les autres routes statiques
@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/services')
def services():
    return render_template('services.html', active_page='services')

@app.route('/projects')
def projects():
    return render_template('projects.html', active_page='projects')

@app.route('/features')
def features():
    return render_template('features.html', active_page='features')

@app.route('/team')
def team():
    return render_template('team.html', active_page='team')

@app.route('/faq')
def faq():
    return render_template('faq.html', active_page='faq')

@app.route('/testimonials')
def testimonials():
    return render_template('testimonials.html', active_page='testimonials')

@app.route('/404')
def not_found():
    return render_template('404.html', active_page='not_found')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    analyses = Analysis.query.filter_by(user_id=session['user_id']).order_by(Analysis.created_at.desc()).all()
    return render_template('service.html', analyses=analyses, active_page='history')


@app.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != session['user_id']:
        flash("Vous n'avez pas accès à cette analyse.")
        return redirect(url_for('history'))

    return render_template("view_analysis.html", analysis=analysis)




@app.route('/users')
def list_users():
    users = User.query.all()  # Récupère tous les utilisateurs
    return render_template('users.html', users=users)


if __name__ == '__main__':
    app.run(debug=True)
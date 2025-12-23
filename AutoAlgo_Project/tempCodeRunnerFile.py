import os
import pandas as pd
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = "super_secret_key"

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ================= DATA ANALYSIS + TRAINING =================
def train_and_visualize(filepath):
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    graphs = []

    # -------- Dataset Overview --------
    plt.figure(figsize=(6, 4))
    df.hist()
    plt.tight_layout()
    dataset_img = "dataset_overview.png"
    plt.savefig(os.path.join(STATIC_FOLDER, dataset_img))
    plt.close()
    graphs.append(dataset_img)

    # -------- Task Detection --------
    task = "Classification" if y.nunique() <= 10 else "Regression"

    # -------- CLASSIFICATION --------
    if task == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000) if y.nunique() == 2 else DecisionTreeClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_img = "confusion_matrix.png"
        plt.savefig(os.path.join(STATIC_FOLDER, cm_img))
        plt.close()
        graphs.append(cm_img)

        return task, model.__class__.__name__, score, graphs, len(df)

    # -------- REGRESSION --------
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Regression Output")

        reg_img = "regression_plot.png"
        plt.savefig(os.path.join(STATIC_FOLDER, reg_img))
        plt.close()
        graphs.append(reg_img)

        return task, "Linear Regression", score, graphs, len(df)


# ================= ROUTES =================
@app.route('/')
@app.route('/home.html')
def home():
    return render_template('home.html')


@app.route('/main.html', methods=['GET', 'POST'])
def main_dashboard():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            task, algo, score, graphs, rows = train_and_visualize(filepath)

            return render_template(
                'main.html',
                result_ready=True,
                task=task,
                algo=algo,
                score=round(score, 4),
                graphs=graphs,
                rows=rows
            )

    return render_template('main.html', result_ready=False)


if __name__ == '__main__':
    app.run(debug=True)

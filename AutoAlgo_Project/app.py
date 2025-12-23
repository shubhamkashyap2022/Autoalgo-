import os, uuid
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score

# ===================== FLASK APP =====================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.static_folder, exist_ok=True)

# ===================== HELPERS =====================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===================== CORE ML + VISUAL =====================
def train_and_visualize(filepath):
    df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    graphs = []

    # ===================== REGRESSION =====================
    if y.nunique() > 10:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        # Simple Linear Regression
        if X.shape[1] == 1:
            plt.figure(figsize=(7, 5))
            plt.scatter(X, y, label="Actual Data")
            plt.plot(X, model.predict(X), color="red", label="Regression Line")
            plt.xlabel(X.columns[0])
            plt.ylabel("Target")
            plt.title("Simple Linear Regression")
            plt.legend()
            algo = "Linear Regression"

        # Multiple Linear Regression
        else:
            plt.figure(figsize=(7, 5))
            plt.scatter(y_test, preds)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Multiple Linear Regression (Actual vs Predicted)")
            algo = "Multiple Linear Regression"

        plot_name = f"reg_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join(app.static_folder, plot_name))
        plt.close()

        graphs.append(plot_name)
        return "Regression", algo, score, graphs, len(df)

    # ===================== LOGISTIC REGRESSION =====================
    if y.nunique() == 2:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))

        # Binary Logistic (Sigmoid)
        if X.shape[1] == 1:
            feature = X.columns[0]
            x_range = np.linspace(X[feature].min(), X[feature].max(), 300).reshape(-1, 1)
            probs = model.predict_proba(x_range)[:, 1]

            plt.figure(figsize=(7, 5))
            plt.scatter(X[feature], y)
            plt.plot(x_range, probs, color="red")
            plt.axhline(0.5, linestyle="--", color="green")
            plt.xlabel(feature)
            plt.ylabel("Probability")
            plt.title("Logistic Regression (Binary)")
            algo = "Logistic Regression (Binary)"

        # Multiple Logistic Regression
        else:
            plt.figure(figsize=(7, 5))
            plt.scatter(X.iloc[:, 0], y, c=y, cmap="viridis")
            plt.xlabel(X.columns[0])
            plt.ylabel("Class")
            plt.title("Multiple Logistic Regression")
            algo = "Multiple Logistic Regression"

        plot_name = f"logistic_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join(app.static_folder, plot_name))
        plt.close()

        graphs.append(plot_name)
        return "Classification", algo, score, graphs, len(df)

    # ===================== DECISION TREE =====================
    if y.nunique() <= 5:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))

        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=X.columns,
            class_names=[str(c) for c in sorted(y.unique())],
            filled=True
        )

        plot_name = f"tree_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join(app.static_folder, plot_name))
        plt.close()

        graphs.append(plot_name)
        return "Classification", "Decision Tree", score, graphs, len(df)

    # ===================== RANDOM FOREST =====================
    if X.shape[1] > 5:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))

        plt.figure(figsize=(20, 10))
        plot_tree(
            model.estimators_[0],
            feature_names=X.columns,
            class_names=[str(c) for c in sorted(y.unique())],
            filled=True
        )

        plot_name = f"forest_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join(app.static_folder, plot_name))
        plt.close()

        graphs.append(plot_name)
        return "Classification", "Random Forest", score, graphs, len(df)

    # ===================== K-MEANS CLUSTERING =====================
    if X.shape[1] >= 2:

        model = KMeans(n_clusters=3, random_state=42)
        clusters = model.fit_predict(X)

        plt.figure(figsize=(7, 5))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="viridis")
        plt.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            marker="*", s=300, color="red", label="Centroids"
        )
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title("K-Means Cluster Analysis")
        plt.legend()

        plot_name = f"kmeans_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join(app.static_folder, plot_name))
        plt.close()

        graphs.append(plot_name)
        return "Clustering", "K-Means Clustering", 0, graphs, len(df)


# ===================== ROUTES =====================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/main.html", methods=["GET", "POST"])
def main_dashboard():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or not allowed_file(file.filename):
            return redirect(request.url)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(filepath)

        task, algo, score, graphs, rows = train_and_visualize(filepath)

        return render_template(
            "main.html",
            result_ready=True,
            task=task,
            algo=algo,
            score=round(score, 4),
            graphs=graphs,
            rows=rows
        )

    return render_template("main.html", result_ready=False)

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(debug=True)
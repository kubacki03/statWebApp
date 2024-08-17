import joblib
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import io
import base64
from scipy.stats import shapiro, stats
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Konieczne do używania sesji
# Folder do przechowywania przesłanych plików
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


import numpy as np
import pandas as pd


def transform_column(col, max_value):
    return (col - max_value) ** 2


def hellwig(df):

    df_numeric = df.select_dtypes(include=[np.number])


    if df_numeric.empty:
        print("No numeric columns found in the DataFrame.")
        return


    max_values = df_numeric.max()


    for column in df_numeric.columns:
        df_numeric[column] = transform_column(df_numeric[column], max_values[column])

    print("Obiekt wzorcowy:")
    print(df_numeric)

    row_sums = df_numeric.sum(axis=1)
    di0 = np.sqrt(row_sums)

    dsr = di0.mean()
    differences = di0 - dsr
    squared_differences = differences ** 2
    Sd0 = np.sqrt(squared_differences.mean())
    d0 = dsr + 2 * Sd0
    si = 1 - (di0 / d0)
    si_series = pd.Series(si, index=df_numeric.index)
    return si_series


def clean_data(df, listDestimulants):

    pd.set_option('display.max_columns', None)


    df.set_index(df.columns[0], inplace=True)


    df[listDestimulants] = 1 / df[listDestimulants]


    print("Dane po zamianie destymulant na stymulanty:")
    print(df)


    df_numeric = df.select_dtypes(include=[np.number])

    mean_values = df_numeric.mean()


    std_values = df_numeric.std()


    df_normalized = (df_numeric - mean_values) / std_values

    print(df_normalized)

    corr = df_normalized.corr()

    print(corr)


    inverse_matrix = np.linalg.inv(corr)


    inverse_matrix_df = pd.DataFrame(inverse_matrix, index=corr.columns, columns=corr.columns)


    diagonal_values = np.diag(inverse_matrix_df)


    columns_to_remove = inverse_matrix_df.columns[diagonal_values > 10]


    df_cleaned = df_numeric.drop(columns=columns_to_remove)

    cv_values = std_values / mean_values

    columns_to_remove = cv_values[cv_values < 0.1].index
    print(columns_to_remove)


    df_cleaned = df_normalized.drop(columns=columns_to_remove)


    print("\nOczyszczony DataFrame:")
    print(df_cleaned)
    return df_cleaned


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)


            session['uploaded_file'] = filename


            df = pd.read_excel(filepath)



            columns = df.columns[1:]

            return render_template('select_columns.html', columns=columns, filename=filename)

    return render_template('index.html')



@app.route('/process', methods=['POST'])
def process():
    selected_columns = request.form.getlist('columns')
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)


    table_html2 = df.to_html(classes='data', header="true", index=True)


    cleaned_df = clean_data(df, selected_columns)


    cleaned_filename = f"cleaned_{filename}"
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
    cleaned_df.to_excel(cleaned_filepath)


    table_html = cleaned_df.to_html(classes='data', header="true", index=True)



    return render_template('result.html', tables=table_html, cleaned_filename=cleaned_filename, tables2=table_html2)


@app.route("/histograms",methods=['POST'])
def histograms():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)
    df = df.iloc[:, 1:]


    num_columns = len(df.columns)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(8, num_columns * 4))

    if num_columns == 1:
        axes = [axes]

    for i, column in enumerate(df.columns):
        axes[i].hist(df[column].dropna(), bins=30, edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()


    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()


    img_base64 = base64.b64encode(img.getvalue()).decode()


    return render_template("histograms.html", img_data=img_base64)

@app.route('/ranking', methods=['POST'])
def ranking():
    cleaned_filename = request.form['cleaned_filename']
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)

    cleaned_df = pd.read_excel(cleaned_filepath, index_col=0)

    si_series = hellwig(cleaned_df)

    ssr = si_series.mean()
    sstd = si_series.std()
    smax = si_series.max()
    smin = si_series.min()

    g1 = []
    g2 = []
    g3 = []
    g4 = []

    for index, s in si_series.items():
        if ssr + sstd <= s <= smax:
            g1.append(index)
        elif ssr <= s <= ssr + sstd:
            g2.append(index)
        elif ssr - sstd <= s < ssr:
            g3.append(index)
        elif smin <= s < ssr - sstd:
            g4.append(index)

    return render_template('result_ranking.html', g1=g1, g2=g2, g3=g3, g4=g4)


@app.route("/clustering", methods=['POST'])
def clustering():
    cleaned_filename = request.form['cleaned_filename']
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)


    cleaned_df = pd.read_excel(cleaned_filepath, index_col=0)


    clustered_df = createClusters(cleaned_df)



    clusters = {}
    for cluster_label in clustered_df['Cluster'].unique():


        clusters[cluster_label] = clustered_df[clustered_df['Cluster'] == cluster_label].index.tolist()


    sorted_clusters = dict(sorted(clusters.items()))


    return render_template('clusters.html', clusters=sorted_clusters)


def createClusters(df):

    n = len(df)
    k = (n * 0.25).__floor__()


    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)


    df['Cluster'] = kmeans.labels_


    return df


@app.route("/correlation", methods=['POST'])
def correlation():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)
    dd = df.iloc[:, 1:]


    corr = dd.corr()


    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)


    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()


    img_base64 = base64.b64encode(img.getvalue()).decode()


    return render_template("correlation.html", matrix=corr.to_html(classes="table table-bordered", border=0),
                           img_data=img_base64)



@app.route("/shapiro", methods=['POST'])
def shapiro_route():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)

    dd = df.iloc[:, 1:]


    results = shapiro_wilk(dd)


    return render_template("shapiro.html", results=results)
def shapiro_wilk(df):
    results = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            stat, p_value = shapiro(df[column].dropna())  # Remove NaN values if present
            result = {
                'column': column,
                'stat': stat,
                'p_value': p_value,
                'normality': 'Nie ma podstaw do odrzucenia hipotezy o normalności rozkładu' if p_value > 0.05 else 'Odrzucamy hipotezę o normalności rozkładu'
            }
            results.append(result)
    return results


@app.route("/setY", methods=['POST'])
def setY():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)
    columns = df.columns[1:]

    return render_template("setY.html",columns=columns)

@app.route("/setVal", methods=['POST'])
def setVal():
    selected_columns = request.form.getlist('columns')
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)
    df.drop(columns=selected_columns, inplace=True)

    dd = df.iloc[:, 1:]

    variables = dd.columns
    return render_template("setVal.html", variables=variables)



@app.route("/prognosis", methods=['POST'])
def prognosis():

    input_values = {col: request.form[col] for col in request.form}

    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)



    df = pd.read_excel(filepath)
    df = df.iloc[:, 1:]  # Usunięcie pierwszej kolumny (indeksu)


    cols = df.columns


    Y = list(cols.difference(input_values.keys()))

    if not Y:
        return "Nie można określić zmiennej zależnej, brak kolumn do wybrania."

    Y = Y[0]


    if Y not in df.columns:
        raise ValueError(f"Kolumna {Y} nie została znaleziona w danych")


    X = df.drop(columns=[Y])  # Wszystkie kolumny oprócz zmiennej zależnej
    y = df[Y]  # Zmienna zależna


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)



    y_pred = model.predict(X_test)


    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')


    joblib.dump(model, 'model.pkl')


    input_df = pd.DataFrame([input_values])


    model = joblib.load('model.pkl')


    prediction = model.predict(input_df)
    r2 = r2_score(y_test, y_pred)


    return render_template('result_prediction.html', input_values=input_values, prediction=prediction[0], zalezna=Y,error=r2)

@app.route("/factor", methods=['POST'])
def factor_analysis():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    df = pd.read_excel(filepath)
    df = df.iloc[:, 1:]


    corr_matrix = np.corrcoef(df, rowvar=False)


    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)


    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]


    num_factors = np.sum(sorted_eigenvalues > 1)


    factor_loadings = sorted_eigenvectors[:, :num_factors] * np.sqrt(sorted_eigenvalues[:num_factors])


    loadings_df = pd.DataFrame(factor_loadings, columns=[f'Factor {i+1}' for i in range(num_factors)],
                               index=df.columns)


    def highlight_values(val):
        color = 'background-color: green;' if val > 0.5 else ''
        return color


    styled_loadings_df = loadings_df.style.applymap(highlight_values).to_html()


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, 'o-', color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Number of Factors')
    plt.ylabel('Eigenvalue')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')


    return render_template('factor.html', plot_url=plot_url, loadings_df=styled_loadings_df)


@app.route("/stats",methods=['POST'])
def stats():
    filename = session.get('uploaded_file')
    if not filename:
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(filepath)
    df = df.iloc[:, 1:]

    statystyki = df.describe()


    statystyki_html = statystyki.to_html(classes='table table-striped')

    return render_template('describe.html', tables=[statystyki_html])


if __name__ == '__main__':
    app.run(debug=True)
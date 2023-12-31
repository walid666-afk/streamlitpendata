import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu

st.set_page_config(page_title='TUGAS UAS PENDATA')
st.write("---")
st.markdown("<h1 style='text-align: center;'>UAS PENAMBANGAN DATA</h1>", unsafe_allow_html=True)
st.write("---")




selected = option_menu(
    menu_title=None,
    options=["Description", "Preprocessing", "Principal Component Analysis", "Classification", "Implementation"],
    icons=["house","calculator","clipboard", "table", "send"],
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
)




if selected == "Description" :
    st.markdown("<h1 style='text-align: center;'>DESKRIPSI DATASET</h1>", unsafe_allow_html=True)
    st.write("###### Dataset yang digunakan adalah dataset penyakit Anemia, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('dataanemia.csv')
    st.dataframe(df)
    st.write("###### Sumber Dataset : https://www.kaggle.com/datasets/ragishehab/anemia-data-set")
    st.write(" Dataset ini berisi informasi tentang individu yang didiagnosis dengan anemia, kondisi yang ditandai dengan penurunan jumlah sel darah merah atau hemoglobin dalam darah. Data termasuk informasi demografis seperti gender (di kode sebagai 0 untuk laki-laki dan 1 untuk perempuan) dan pengukuran klinis seperti tingkat hemoglobin, rata-rata hemoglobin korpuskular (MCH), konsentrasi Hemoglobin Korpuscular Rata-rata ( MCHC), dan volume korpuscular rata-ratanya. (MCV). Variabel target adalah adanya atau tidak adanya anemia. (coded as 0 for not anemic and 1 for anemic). Dataset ini dapat digunakan untuk mengeksplorasi hubungan antara variabel ini dan inciden anemia, serta untuk mengembangkan model prediktif untuk mengidentifikasi individu yang berisiko mengembangkan kondisi ini. ")
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")
if selected == "Preprocessing":
    st.markdown("<h1 style='text-align: center;'>NORMALISASI DATA</h1>", unsafe_allow_html=True)
    st.subheader(" Rumus Normalisasi Data :")
    st.image('rumus_normalisasi.png', use_column_width=False, width=250)

    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    # Mendefinisikan Varible X dan Y
    df = pd.read_csv('dataanemia.csv')
    X = df.drop(columns=['Result'])
    y = df['Result'].values
    df
    st.subheader("Pemisahan Kolom Result Sebagai Atribut Target")
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Result).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

if selected == "Principal Component Analysis":
    st.markdown("<h1 style='text-align: center;'>HASIL TEST PCA</h1>", unsafe_allow_html=True)
    df = pd.read_csv('dataanemia.csv')
    X = df.drop(columns=['Result'])
    y = df['Result'].values
    
    pca_components = st.selectbox("Select Number of PCA components", [2, 3, 4, 5])

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)

    pca = PCA(n_components=pca_components)
    pca_features = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(pca_features, columns=[
                          f"PC{i+1}" for i in range(pca_components)])
    pca_df["Result"] = df["Result"]

    st.write(pca_df.head())

    X_pca = pca_df.drop("Result", axis=1)
    y_pca = pca_df["Result"]

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y_pca, test_size=0.2, random_state=42)

    # PCA Naive Bayes
    st.subheader("PCA Naive Bayes")
    nb_pca = GaussianNB()
    nb_pca.fit(X_train_pca, y_train_pca)
    y_pred_nb_pca = nb_pca.predict(X_test_pca)
    accuracy_nb_pca = accuracy_score(y_test_pca, y_pred_nb_pca)
    st.write("Accuracy:",100 * accuracy_nb_pca)

    # PCA KNN
    st.subheader("PCA KNN")
    knn_pca = KNeighborsClassifier()
    knn_pca.fit(X_train_pca, y_train_pca)
    y_pred_knn_pca = knn_pca.predict(X_test_pca)
    accuracy_knn_pca = accuracy_score(y_test_pca, y_pred_knn_pca)
    st.write("Accuracy:",100 * accuracy_knn_pca)

    # PCA Decision Tree
    st.subheader("PCA Decision Tree")
    dt_pca = DecisionTreeClassifier()
    dt_pca.fit(X_train_pca, y_train_pca)
    y_pred_dt_pca = dt_pca.predict(X_test_pca)
    accuracy_dt_pca = accuracy_score(y_test_pca, y_pred_dt_pca)
    st.write("Accuracy:",100 * accuracy_dt_pca)

    # PCA ANNBP
    st.subheader("PCA ANNBP")
    annbp_pca = MLPClassifier()
    annbp_pca.fit(X_train_pca, y_train_pca)
    y_pred_annbp_pca = annbp_pca.predict(X_test_pca)
    accuracy_annbp_pca = accuracy_score(y_test_pca, y_pred_annbp_pca)
    st.write("Accuracy:",100 * accuracy_annbp_pca)
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

if selected == "Classification":
    # Nilai X training dan Nilai X testing
    df = pd.read_csv('dataanemia.csv')
    X = df.drop(columns=['Result'])
    y = df['Result'].values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("Modeling"):
        st.markdown("<h1 style='text-align: center;'>CALSSIFICATION MODELS</h1>", unsafe_allow_html=True)
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        mlp_model = st.checkbox('ANNBackpropagation')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        # Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(
                    gaussian_akurasi))
            if k_nn:
                st.write(
                    "Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write(
                    "Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if mlp_model:
                st.write(
                    'Model ANN (MLP) accuracy score: {0:0.2f}'.format(mlp_accuracy))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [gaussian_akurasi, knn_akurasi, dt_akurasi, mlp_accuracy],
                'Model': ['Naive Bayes', 'K-NN', 'Decission Tree', 'ANNBP'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")


if selected == "Implementation":
    with st.form("my_form"):
        st.markdown("<h1 style='text-align: center;'>IMPLEMENTASI</h1>", unsafe_allow_html=True)
        gender = st.selectbox('Masukkan Gender:', ('0', '1'))
        hemoglobin = st.number_input('Masukkan Hemoglobin : ')
        mch = st.number_input('Masukkan MCH : ')
        mchc = st.number_input('Masukkan MCHC : ')
        mcv = st.number_input(
            'Masukkan MCV : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                             ('Naive Bayes', 'K-NN', 'Decision Tree', 'ANNBackpropaganation'))

        apply_pca = st.checkbox("Include PCA")

        prediksi = st.form_submit_button("Submit")
        df = pd.read_csv('dataanemia.csv')
        X = df.drop(columns=['Result'])
        y = df['Result'].values

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(X)
        training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
        training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)
 # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        # Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))
        if prediksi:
            inputs = np.array([
                hemoglobin,
                mch,
                mchc,
                mcv,
                int(gender) 
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            # if apply_pca:
            #     pca = PCA(n_components=2)
            #     X_pca = pca.fit_transform(X)
            #     input_norm = pca.fit_transform(input_norm)

            if apply_pca and X.shape[1] > 1 and X.shape[0] > 1:
                pca = PCA(n_components=min(X.shape[1], X.shape[0]))
                X_pca = pca.fit_transform(X)
                input_norm = pca.transform(input_norm)

            if model == 'Naive Bayes':
                mod = gaussian
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'K-NN':
                mod = knn
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'Decision Tree':
                mod = dt
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'ANNBackpropaganation':
                mod = mlp
                if apply_pca:
                    input_norm = pca.transform(input_norm)

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
            ada = 1
            tidak_ada = 0
            if input_pred == ada:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'Pasien di Diagnosis penyakit ANEMIA')
            else:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'Pasien Tidak di Diagnosis penyakit ANEMIA')
    st.write("---")
    st.write("By Walid Rijal Awali")
    st.write("© Copyright 2023.")

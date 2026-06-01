import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split, KFold


st.set_page_config(
    page_title="AI Customer Segmentation Dashboard",
    layout="wide"
)

st.title("AI Destekli Müşteri Segmentasyonu Dashboard")
st.write("K-Means ve DBSCAN algoritmaları ile müşteri verilerini analiz eder.")

uploaded_file = st.file_uploader("CSV veri dosyanızı yükleyiniz", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Ayarlar")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    selected_features = st.sidebar.multiselect(
        "Modelde kullanılacak sayısal değişkenleri seçiniz",
        numeric_cols,
        default=[
            col for col in [
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ] if col in numeric_cols
        ]
    )

    if len(selected_features) < 2:
        st.warning("En az 2 sayısal değişken seçmelisiniz.")
    else:
        st.subheader("1. Veri Önizleme")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Satır Sayısı", df.shape[0])
        col2.metric("Sütun Sayısı", df.shape[1])
        col3.metric("Seçilen Özellik Sayısı", len(selected_features))

        st.subheader("2. Veri Anlama")

        c1, c2 = st.columns(2)

        with c1:
            st.write("Eksik Değerler")
            st.dataframe(df.isnull().sum())

        with c2:
            st.write("Tanımsal İstatistikler")
            st.dataframe(df[selected_features].describe())

        st.subheader("3. Keşifçi Veri Analizi")

        feature_x = st.selectbox("X ekseni", selected_features, index=0)
        feature_y = st.selectbox("Y ekseni", selected_features, index=1)

        st.subheader("Scatter Plot Analizi")

        fig_scatter = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color="Gender" if "Gender" in df.columns else None,
            size="Annual Income (k$)" if "Annual Income (k$)" in df.columns else None,
            hover_data=selected_features,
            title="Müşteri Verilerinin Scatter Plot Dağılımı"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
        corr = df[selected_features].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Korelasyon Matrisi"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Aykırı Değer Analizi")

        fig_box = px.box(
            df,
            y=selected_features,
            title="Boxplot ile Aykırı Değer Analizi"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        X = df[selected_features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.subheader("4. K-Means Küme Sayısı Belirleme")

        k_values = range(2, 11)

        wcss = []
        silhouette_scores_kmeans = []
        davies_scores = []
        calinski_scores = []

        for k in k_values:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(X_scaled)

            wcss.append(model.inertia_)
            silhouette_scores_kmeans.append(
                silhouette_score(X_scaled, labels)
            )
            davies_scores.append(
                davies_bouldin_score(X_scaled, labels)
            )
            calinski_scores.append(
                calinski_harabasz_score(X_scaled, labels)
            )

        col_elbow, col_sil = st.columns(2)

        with col_elbow:
            fig_elbow = px.line(
                x=list(k_values),
                y=wcss,
                markers=True,
                title="Elbow Method / WCSS",
                labels={"x": "Küme Sayısı", "y": "WCSS"}
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        with col_sil:
            fig_sil = px.line(
                x=list(k_values),
                y=silhouette_scores_kmeans,
                markers=True,
                title="K-Means Silhouette Score",
                labels={"x": "Küme Sayısı", "y": "Silhouette Score"}
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        st.subheader("Ek Kümeleme Performans Metrikleri")

        col_db, col_ch = st.columns(2)

        with col_db:
            fig_db = px.line(
                x=list(k_values),
                y=davies_scores,
                markers=True,
                title="Davies-Bouldin Score",
                labels={
                    "x": "Küme Sayısı",
                    "y": "Davies-Bouldin Score"
                }
            )
            st.plotly_chart(fig_db, use_container_width=True)

        with col_ch:
            fig_ch = px.line(
                x=list(k_values),
                y=calinski_scores,
                markers=True,
                title="Calinski-Harabasz Score",
                labels={
                    "x": "Küme Sayısı",
                    "y": "Calinski-Harabasz Score"
                }
            )
            st.plotly_chart(fig_ch, use_container_width=True)

        best_k = list(k_values)[silhouette_scores_kmeans.index(max(silhouette_scores_kmeans))]

        selected_k = st.sidebar.slider(
            "K-Means küme sayısı",
            min_value=2,
            max_value=10,
            value=int(best_k)
        )

        st.success(f"Silhouette Score'a göre önerilen K değeri: {best_k}")

        st.subheader("5. Senaryo Bazlı Model Değerlendirme")

        # SENARYO 1: Holdout %80 - %20
        X_train, X_test = train_test_split(
            X_scaled,
            test_size=0.20,
            random_state=42
        )

        holdout_model = KMeans(
            n_clusters=selected_k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        holdout_model.fit(X_train)

        train_labels = holdout_model.labels_
        test_labels = holdout_model.predict(X_test)

        train_silhouette = silhouette_score(X_train, train_labels)
        test_silhouette = silhouette_score(X_test, test_labels)

        # SENARYO 2: 5-Fold Cross Validation
        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        fold_scores = []

        for train_index, test_index in kf.split(X_scaled):
            X_train_fold = X_scaled[train_index]
            X_test_fold = X_scaled[test_index]

            fold_model = KMeans(
                n_clusters=selected_k,
                init="k-means++",
                random_state=42,
                n_init=10
            )

            fold_model.fit(X_train_fold)
            fold_labels = fold_model.predict(X_test_fold)

            fold_score = silhouette_score(
                X_test_fold,
                fold_labels
            )

            fold_scores.append(fold_score)

        kfold_mean_score = sum(fold_scores) / len(fold_scores)

        # SENARYO 3: Holdout %70 - %30
        X_train_70, X_test_30 = train_test_split(
            X_scaled,
            test_size=0.30,
            random_state=42
        )

        holdout_model_70 = KMeans(
            n_clusters=selected_k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        holdout_model_70.fit(X_train_70)

        labels_70 = holdout_model_70.predict(X_test_30)

        scenario3_score = silhouette_score(
            X_test_30,
            labels_70
        )

        scenario_results = pd.DataFrame({
            "Senaryo": [
                "Senaryo 1: Holdout (%80 - %20) Eğitim",
                "Senaryo 1: Holdout (%80 - %20) Test",
                "Senaryo 2: 5-Fold Cross Validation",
                "Senaryo 3: Holdout (%70 - %30)"
            ],
            "Yöntem": [
                "Holdout",
                "Holdout",
                "K-Fold",
                "Holdout"
            ],
            "Silhouette Score": [
                round(train_silhouette, 4),
                round(test_silhouette, 4),
                round(kfold_mean_score, 4),
                round(scenario3_score, 4)
            ]
        })

        st.dataframe(scenario_results)

        fig_scenario = px.bar(
            scenario_results,
            x="Senaryo",
            y="Silhouette Score",
            color="Yöntem",
            title="Senaryo Bazlı K-Means Performans Karşılaştırması"
        )

        st.plotly_chart(fig_scenario, use_container_width=True)

        st.subheader("Veri Bölme Oranları")

        split_df = pd.DataFrame({
            "Senaryo": [
                "Holdout (%80 Eğitim)",
                "Holdout (%20 Test)",
                "Holdout (%70 Eğitim)",
                "Holdout (%30 Test)",
                "5-Fold Cross Validation"
            ],

            "Oran": [
                80,
                20,
                70,
                30,
                100
            ]
        })

        fig_split = px.pie(
            split_df,
            names="Senaryo",
            values="Oran",
            title="Senaryo Bazlı Veri Bölme Oranları"
        )

        st.plotly_chart(fig_split, use_container_width=True)

        st.subheader("6. Final K-Means Modeli")

        kmeans = KMeans(
            n_clusters=selected_k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

        kmeans_silhouette = silhouette_score(
            X_scaled,
            df["KMeans_Cluster"]
        )

        kmeans_davies = davies_bouldin_score(
            X_scaled,
            df["KMeans_Cluster"]
        )

        kmeans_calinski = calinski_harabasz_score(
            X_scaled,
            df["KMeans_Cluster"]
        )

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)

        col_k1.metric("Silhouette Score", round(kmeans_silhouette, 4))
        col_k2.metric("WCSS / Inertia", round(kmeans.inertia_, 2))
        col_k3.metric("Davies-Bouldin", round(kmeans_davies, 4))
        col_k4.metric("Calinski-Harabasz", round(kmeans_calinski, 2))

        st.subheader("7. DBSCAN Modeli")

        st.write(
            "DBSCAN için farklı eps ve min_samples değerleri denenerek "
            "en uygun parametre kombinasyonu otomatik seçilmiştir."
        )

        eps_values = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
        min_samples_values = [3, 4, 5, 6, 8, 10]

        best_dbscan_score = -1
        best_dbscan_params = None
        best_dbscan_labels = None
        best_cluster_count = 0
        best_noise_count = 0

        dbscan_results = []

        for eps_val in eps_values:
            for min_samp in min_samples_values:
                temp_dbscan = DBSCAN(
                    eps=eps_val,
                    min_samples=min_samp
                )

                temp_labels = temp_dbscan.fit_predict(X_scaled)

                cluster_count = len(set(temp_labels)) - (
                    1 if -1 in temp_labels else 0
                )

                noise_count = list(temp_labels).count(-1)

                if cluster_count >= 2:
                    score = silhouette_score(X_scaled, temp_labels)

                    dbscan_results.append({
                        "eps": eps_val,
                        "min_samples": min_samp,
                        "Küme Sayısı": cluster_count,
                        "Noise Sayısı": noise_count,
                        "Silhouette Score": round(score, 4)
                    })

                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_params = (eps_val, min_samp)
                        best_dbscan_labels = temp_labels
                        best_cluster_count = cluster_count
                        best_noise_count = noise_count

        if best_dbscan_labels is not None:
            df["DBSCAN_Cluster"] = best_dbscan_labels
            dbscan_silhouette = best_dbscan_score
            dbscan_cluster_count = best_cluster_count
            noise_count = best_noise_count

            dbscan_davies = davies_bouldin_score(
                X_scaled,
                df["DBSCAN_Cluster"]
            )

            dbscan_calinski = calinski_harabasz_score(
                X_scaled,
                df["DBSCAN_Cluster"]
            )

            st.success(
                f"En iyi DBSCAN parametreleri: "
                f"eps={best_dbscan_params[0]}, "
                f"min_samples={best_dbscan_params[1]}"
            )

            col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

            col_d1.metric("DBSCAN Küme Sayısı", dbscan_cluster_count)
            col_d2.metric("Aykırı / Noise Sayısı", noise_count)
            col_d3.metric("Silhouette Score", round(dbscan_silhouette, 4))
            col_d4.metric("Davies-Bouldin", round(dbscan_davies, 4))
            col_d5.metric("Calinski-Harabasz", round(dbscan_calinski, 2))

            st.write("Denenen DBSCAN parametre sonuçları:")
            st.dataframe(pd.DataFrame(dbscan_results))

        else:
            df["DBSCAN_Cluster"] = -1
            dbscan_silhouette = None
            dbscan_davies = None
            dbscan_calinski = None
            dbscan_cluster_count = 0
            noise_count = len(df)

            st.warning(
                "DBSCAN verilen parametre aralıklarında anlamlı en az 2 küme oluşturamadı. "
                "Bu nedenle performans metrikleri hesaplanamadı."
            )

            col_d1, col_d2, col_d3 = st.columns(3)

            col_d1.metric("DBSCAN Küme Sayısı", dbscan_cluster_count)
            col_d2.metric("Aykırı / Noise Sayısı", noise_count)
            col_d3.metric("DBSCAN Silhouette Score", "Hesaplanamadı")

        st.subheader("8. Algoritma Karşılaştırması")

        comparison_df = pd.DataFrame({
            "Algoritma": ["K-Means", "DBSCAN"],
            "Silhouette Score": [
                round(kmeans_silhouette, 4),
                round(dbscan_silhouette, 4) if dbscan_silhouette is not None else 0
            ],
            "Davies-Bouldin": [
                round(kmeans_davies, 4),
                round(dbscan_davies, 4) if dbscan_davies is not None else 0
            ],
            "Calinski-Harabasz": [
                round(kmeans_calinski, 2),
                round(dbscan_calinski, 2) if dbscan_calinski is not None else 0
            ],
            "Küme Sayısı": [
                selected_k,
                dbscan_cluster_count
            ]
        })

        st.dataframe(comparison_df)

        fig_compare_sil = px.bar(
            comparison_df,
            x="Algoritma",
            y="Silhouette Score",
            color="Algoritma",
            title="K-Means ve Silhouette Score Karşılaştırması",
            text="Silhouette Score"
        )
        st.plotly_chart(fig_compare_sil, use_container_width=True)

        fig_compare_db = px.bar(
            comparison_df,
            x="Algoritma",
            y="Davies-Bouldin",
            color="Algoritma",
            title="K-Means ve Davies-Bouldin Karşılaştırması",
            text="Davies-Bouldin"
        )
        st.plotly_chart(fig_compare_db, use_container_width=True)

        fig_compare_ch = px.bar(
            comparison_df,
            x="Algoritma",
            y="Calinski-Harabasz",
            color="Algoritma",
            title="K-Means ve Calinski-Harabasz Karşılaştırması",
            text="Calinski-Harabasz"
        )
        st.plotly_chart(fig_compare_ch, use_container_width=True)

        st.subheader("9. K-Means Görselleştirme")

        fig_kmeans = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color=df["KMeans_Cluster"].astype(str),
            title="K-Means Kümeleme Sonucu",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig_kmeans, use_container_width=True)

        if len(selected_features) >= 3:
            z_feature = st.selectbox(
                "3D grafik için Z ekseni",
                selected_features,
                index=2
            )

            fig_3d = px.scatter_3d(
                df,
                x=feature_x,
                y=feature_y,
                z=z_feature,
                color=df["KMeans_Cluster"].astype(str),
                title="3D K-Means Müşteri Segmentasyonu",
                labels={"color": "Cluster"}
            )

            st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("10. Otomatik Segment Yorumları")

        summary = df.groupby("KMeans_Cluster")[selected_features].mean().round(2)
        st.dataframe(summary)

        for cluster_id, row in summary.iterrows():
            yorum = f"Cluster {cluster_id}: "

            if (
                "Annual Income (k$)" in selected_features
                and "Spending Score (1-100)" in selected_features
            ):
                income = row["Annual Income (k$)"]
                spending = row["Spending Score (1-100)"]

                avg_income = df["Annual Income (k$)"].mean()
                avg_spending = df["Spending Score (1-100)"].mean()

                if income >= avg_income and spending >= avg_spending:
                    segment_name = "Premium / yüksek değerli müşteri grubu"
                    advice = (
                        "Bu grup için özel kampanyalar, sadakat programları "
                        "ve premium teklifler önerilebilir."
                    )

                elif income >= avg_income and spending < avg_spending:
                    segment_name = "Potansiyel müşteri grubu"
                    advice = (
                        "Gelir seviyesi yüksek ancak harcama skoru düşük olduğu için "
                        "kişiselleştirilmiş kampanyalar uygulanabilir."
                    )

                elif income < avg_income and spending >= avg_spending:
                    segment_name = "Aktif fakat fiyat duyarlı müşteri grubu"
                    advice = (
                        "İndirim, fırsat kampanyaları ve dönemsel promosyonlar "
                        "etkili olabilir."
                    )

                else:
                    segment_name = "Düşük etkileşimli müşteri grubu"
                    advice = (
                        "Bu grup için düşük maliyetli bilgilendirme ve temel kampanya "
                        "stratejileri tercih edilebilir."
                    )

                yorum += (
                    f"{segment_name}. Ortalama gelir: {income}, "
                    f"ortalama harcama skoru: {spending}. {advice}"
                )

            else:
                yorum += (
                    "Seçilen değişkenlere göre benzer özellik gösteren "
                    "müşteri grubudur."
                )

            st.info(yorum)

        st.subheader("11. Sonuçlu Veri Seti")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Sonuçları CSV Olarak İndir",
            data=csv,
            file_name="segmentasyon_sonuclari.csv",
            mime="text/csv"
        )

else:
    st.info("Analizi başlatmak için CSV dosyanızı yükleyiniz.")




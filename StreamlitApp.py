import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

data = pd.read_csv("data/Churn_Modelling.csv")
model = joblib.load("data/rf_model.joblib")

st.set_page_config(
    page_title="Bank Customer Exploration Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    .appview-container {
        max-width: 1400px;
        margin: auto;
    }
    """,
    unsafe_allow_html=True
)

st.markdown(
            """
            <h2 style="text-align: center;">Bunk Churn Prediction</h3>
            """, unsafe_allow_html=True)

##################################
# INTRODUCTION CANTAINER 
##################################

with st.container():
    with st.container(border=True):
            st.markdown(
            """
            <h10 style="text-align: justify;">In diesem Projekt untersuchen wir die Faktoren, die die Abwanderung von Bankkunden beeinflussen, und erstellen Vorhersagen mithilfe von Methoden der Explorativen Datenanalyse (EDA) und des maschinellen Lernens.

            Unsere Analyse beginnt mit der Visualisierung der wichtigsten Merkmale der Kundendaten durch Boxplots und Countplots. Diese Diagramme bieten Einblicke in verschiedene Aspekte der Kundenbasis und helfen uns, die entscheidenden Faktoren zu verstehen, die das Verhalten der Kunden beeinflussen können. Die Ergebnisse dieser Analyse bilden die Grundlage für die Entwicklung von Vorhersagemodellen und Strategien zur Kundenbindung, um datenbasierte Entscheidungen zu treffen.</h10>
            """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

# COLOR
viridis_colors = px.colors.sequential.Viridis

##############################################
# BOXPLOT AND COUNTPLOT
##############################################
with col3:
    with st.container(border=True, height=700):

        selection = st.selectbox("Wählen Sie den Diagrammtyp aus, um den Datensatz auszuwerten:", ["Boxplot", "Countplot"])

        if selection == "Boxplot":
            box_feature = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','IsActiveMember', 'EstimatedSalary']
            
            selected_variable = st.selectbox("Wählen Sie den Feature aus:", box_feature)
            
            custom_colors = ['#e6c200', '#482173']
            
            fig = px.box(data, y=selected_variable, x='Exited', title=selected_variable, width=400, height=400,
                         color='Exited', color_discrete_sequence=custom_colors)
            
            fig.update_layout(
                yaxis=dict(autorange=True),
                xaxis=dict(autorange=True)
            )
            
            st.plotly_chart(fig)
            st.write(f'Median für {selected_variable}: {data[selected_variable].median()}')

        elif selection == 'Countplot':
            count_features = ['Age','Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']
            selected_variable = st.selectbox("Wählen Sie den Feature aus:", count_features)
            
            custom_colors = ['#e6c200', '#482173']
            
            fig = px.histogram(data, x=selected_variable, color='Exited', barmode='group', 
                               title=f'Countplot of {selected_variable}', width=400, height=400,
                               color_discrete_sequence=custom_colors)
            fig.update_xaxes(categoryorder='total descending')
            
            fig.update_layout(
                yaxis=dict(autorange=True),
                xaxis=dict(autorange=True)
            )
            
            st.plotly_chart(fig)

##################################
# MAP
##################################
with col4:
    with st.container(border=True, height=700):
        st.write("Kundengeography")

        fig_map = px.choropleth(data['Geography'].value_counts(), 
                            locations=data['Geography'].unique(), 
                            locationmode="country names",
                            color=data['Geography'].value_counts().values,
                            scope="europe",
                            range_color=(data['Geography'].value_counts().min(), data['Geography'].value_counts().max()),
                            color_continuous_scale=viridis_colors
        )
        
        fig_map.update_geos(
            visible=True,
            showcountries=True,
            countrycolor="lightgrey",
            showland=True,
            landcolor="white",
            fitbounds="locations"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.table(data['Geography'].value_counts())

with st.expander("Business insights aus der Visualisierung:"):
    with st.container(border=True): 
        
        st.markdown(
        """
        <p style="text-align: justify;">Wir stellen folgende Beobachtungen fest:</p>
        <ul>
            <li>Es gibt keinen wesentlichen Unterschied in der Verteilung der Kreditwürdigkeit zwischen den gebliebenen und abgewanderten Kunden.</li>
            <li>Ältere Kunden neigen eher dazu, abzuspringen als jüngere, was auf unterschiedliche Servicepräferenzen in den Altersgruppen hinweist. Die Bank sollte möglicherweise ihre Zielgruppenstrategie überprüfen oder Maßnahmen zur Kundenbindung für verschiedene Altersgruppen anpassen.
            <li>In Bezug auf die Dauer der Kundenbeziehung zeigen sich extreme Enden (Kunden mit kurzer oder sehr langer Bankbeziehung) als eher abwanderungsgefährdet im Vergleich zu denen mit durchschnittlicher Aufenthaltsdauer.</li>
            <li>Besorgniserregend ist, dass die Bank Kunden mit signifikanten Kontoständen verliert, was wahrscheinlich ihre verfügbaren Mittel für Kredite beeinträchtigt.</li>
            <li>Weder die Anzahl der Produkte noch das Gehalt haben einen signifikanten Einfluss auf die Wahrscheinlichkeit der Abwanderung.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

col5, col6, col7  = st.columns(3)

##################################
# FEATURES IMPORTANCE FOR PREDICTIONS 
##################################
with col5:
    with st.container(border=True, height=800):
        st.markdown(
            """
            <h3 style="text-align: center;">Feature Importance</h3>
            """, unsafe_allow_html=True
        )

        feature_importance_df = pd.read_csv("data/feature_importance.csv", usecols=["Feature", "Importance"])
        
        fig = px.bar(
            feature_importance_df.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Importance Score", "Feature": "Features"},
            width=350,  
            height=450,  
            color_discrete_sequence=["#e6c200"] 
        )

        fig.update_layout(
            coloraxis_showscale=False, 
            xaxis_title="Importance Score",
            yaxis_title="Features",
        )

        st.plotly_chart(fig)


##################################
# USER INPUTS
##################################

with col6:
    with st.container(border=True, height=800):

        CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=400)
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=1)
        Balance = st.number_input("Balance", min_value=0, max_value=300000, value=5000)
        NumOfProducts = st.number_input("Number of Products", min_value=0, max_value=4)
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0, max_value=200000, value=50000)
        Geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        IsActiveMember = st.selectbox("Active Member?", ["Yes", "No"])
        HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])

        user_input = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "EstimatedSalary": [EstimatedSalary],
        "Geography_Germany": [1 if Geography == "Germany" else 0],
        "Geography_Spain": [1 if Geography == "Spain" else 0],
        "Gender_Male": [1 if Gender == "Male" else 0],
        "IsActiveMember": [1 if IsActiveMember == "Yes" else 0],
        "HasCrCard": [1 if HasCrCard == "Yes" else 0],
        })

##################################
# COLOUMN 4 PREDICTION BUTTON
##################################
with col7:
    with st.container(border=True, height=800):

        if st.button("Vorhersagen"):
            prediction = model.predict(user_input)[0]
            probabilities = model.predict_proba(user_input)[0]
            
            churn_prob = round(probabilities[1] * 100, 2)  
            retain_prob = round(probabilities[0] * 100, 2) 
            
            fig = px.pie(values=[retain_prob, churn_prob], names=["Retain", "Churn"], 
                        color=["Retain", "Churn"], 
                        color_discrete_map={"Retain": "#482173", "Churn": "#e6c200"},  
                        hole=0.3, width=300)
            
            fig.update_traces(textinfo='percent+label')  
            st.plotly_chart(fig)

            if prediction == 1:
                st.error("🚨 Der Kunde wird voraussichtlich abwandern!")
            else:
                st.success("✅ Der Kunde wird voraussichtlich bleiben!")

            
##########################
# FAZIT 
##########################
with st.expander("Fazit zur Kundenabwanderung:"):
    with st.container(border=True):

        st.markdown(
            """
            <p style="text-align: justify;">
            - <b>Beste Modellwahl:</b> RandomForestClassifier zeigte die beste Leistung.<br><br>

            - <b>Wichtigste Ergebnisse:</b><br>
              - Klasse 0 (bleibende Kunden): Precision: 92%, Recall: 88%, F1-Score: 90%<br>
              - Klasse 1 (abgewanderte Kunden): Precision: 60%, Recall: 70%, F1-Score: 65%<br>
              - ROC-AUC-Score: 88,07%<br><br>

            - <b>Interpretation:</b> 
              Für die Prognose der Kundenabwanderung ist es entscheidend, alle potenziellen Kunden zu identifizieren, die das Unternehmen verlassen könnten. 
              Daher war die wichtigste Metrik der <b>Recall für Klasse "1" (Abwanderung)</b>.<br><br>

            - Da der Recall für abgewanderte Kunden <b>70%</b> beträgt, bedeutet dies, dass das Modell <b>70%</b> der Kunden, die die Bank verlassen könnten, korrekt identifiziert. 
              Dies ist entscheidend, da jeder verlorene Kunde zu einem Umsatzverlust für das Unternehmen führt.<br><br>

            - <b>Herausforderungen & Optimierungsmöglichkeiten:</b><br>
              • Ungleichgewicht der Klassen: Das ursprüngliche Daten-Set hatte eine ungleiche Verteilung zwischen bleibenden (0) und abwandernden Kunden (1). 
                Dies beeinflusste die Ergebnisse, da das Modell besser bleibende Kunden vorhersagte als abwandernde.<br>
              • Optimierung: Die Verwendung einer <b>benutzerdefinierten Verlustfunktion (custom loss function)</b> könnte helfen, wenn Geschäftsdaten zu Kundenverlusten verfügbar sind.<br>
            </p>
            """,
            unsafe_allow_html=True
        )
  
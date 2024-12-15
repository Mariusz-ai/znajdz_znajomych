import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ustawienie szerokości strony (musi być na początku)
st.set_page_config(layout="wide")

# Zmienia kontekst rozmiaru i stylu czcionek, aby były czytelne podczas prezentacji lub rozmowy
sns.set_theme(style="ticks", context="talk")
plt.style.use("dark_background")

# Tytuł
st.markdown("<h1 style='text-align: center;'>Znajdź znajomych</h1>", unsafe_allow_html=True)

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# Funkcja filtrowania danych (ogólnych) bez usuwania pustych wartości
def filter_data_for_overall(df, age, edu_level, fav_animals, fav_place, gender):
    if age:
        df = df[df['age'].isin(age)]
    if edu_level:
        df = df[df['edu_level'].isin(edu_level)]
    if fav_animals:
        fav_animals = [x for x in fav_animals if x != 'Brak ulubionych']
        df = df[df['fav_animals'].isin(fav_animals)]
    if fav_place:
        fav_place = [x for x in fav_place if x != 'Brak ulubionych']
        df = df[df['fav_place'].isin(fav_place)]
    if gender != "Wszyscy":
        df = df[df['gender'] == gender]
    return df

# Funkcja filtrowania danych (dla klastrów)
def filter_data_for_cluster(df, age, edu_level, fav_animals, fav_place, gender, cluster_id=None):
    if age:
        df = df[df['age'].isin(age)]
    if edu_level:
        df = df[df['edu_level'].isin(edu_level)]
    if fav_animals:
        fav_animals = [x for x in fav_animals if x != 'Brak ulubionych']
        df = df[df['fav_animals'].isin(fav_animals)]
    if fav_place:
        fav_place = [x for x in fav_place if x != 'Brak ulubionych']
        df = df[df['fav_place'].isin(fav_place)]
    if gender != "Wszyscy":
        df = df[df['gender'] == gender]
    if cluster_id is not None:
        df = df[df["Cluster"] == cluster_id]  # Filtracja po klastrze
    return df

# Użycie pełnych danych przed filtracją
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

# Funkcja zliczająca kategorie
def count_categories(df, column, categories_order):
    category_counts = df[column].value_counts()
    counts = []
    for cat in categories_order:
        count = category_counts.get(cat, 0)
        counts.append(count)
    return counts

# Funkcja rysująca wykres radarowy
def plot_radar(ax, data, categories, color, title):
    N = len(categories)  # Liczba kategorii
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Upewnij się, że dane mają taką samą długość jak kategorie
    if len(data) < N:
        data = data + [0] * (N - len(data))  # Dodaj brakujące dane jako 0
    elif len(data) > N:
        data = data[:N]  # Przytnij dane, jeśli ich jest za dużo

    # Dodaj pierwszy punkt na koniec, aby zamknąć wykres radarowy
    data = data + [data[0]]
    angles += [angles[0]]

    # Rysowanie wykresu
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.plot(angles, data, color=color, linewidth=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, rotation=45, ha='center')
    ax.set_title(title, fontsize=14, pad=30)
    ax.yaxis.grid(True, color="gray", linestyle="dotted")

#=====SIDEBAR=====
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    
    age = st.multiselect("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.multiselect("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.multiselect("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    if not fav_animals:
        fav_animals = ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy']
    fav_place = st.multiselect("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne', 'Brak ulubionych'])
    if not fav_place:
        fav_place = ['Nad wodą', 'W lesie', 'W górach', 'Inne', 'Brak ulubionych']
    gender = st.radio("Płeć", ['Wszyscy', 'Mężczyzna', 'Kobieta'])

# Załaduj dane i modele
model = get_model()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
all_df = get_all_participants()  # Załaduj dane przed użyciem

# Filtrowanie danych
filtered_df_overall = all_df.copy()
filtered_df_cluster = filter_data_for_cluster(all_df, age, edu_level, fav_animals, fav_place, gender, cluster_id=None)
filtered_df_overall = filter_data_for_overall(all_df, age, edu_level, fav_animals, fav_place, gender)

#=====TWORZENIE ZAKŁADEK=====
tab1, tab2 = st.tabs(["maCHiNG", "Eksploracja danych, trochę inaczej!"])

with tab1:
    # Predykcja klastra użytkownika
    person_df = pd.DataFrame([{
        'age': ', '.join(age) if age else 'unknown',
        'edu_level': ', '.join(edu_level) if edu_level else 'unknown',
        'fav_animals': ', '.join(fav_animals) if fav_animals else 'Brak ulubionych',
        'fav_place': ', '.join(fav_place) if fav_place else 'Brak ulubionych',
        'gender': gender,
    }])
    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

    # Wyświetlanie danych klastrów
    st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = filter_data_for_cluster(filtered_df_cluster, age, edu_level, fav_animals, fav_place, gender, predicted_cluster_id)
    st.metric("Liczba twoich znajomych", len(same_cluster_df))

    # Wyznaczanie danych z klastrów
    age_counts_cluster = count_categories(same_cluster_df, 'age', ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    fav_animals_counts_cluster = count_categories(same_cluster_df, 'fav_animals', ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place_counts_cluster = count_categories(same_cluster_df, 'fav_place', ['Nad wodą', 'W lesie', 'W górach', 'Inne'])

    # Wykres radarowy - dane z klastra
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 4), subplot_kw=dict(polar=True))
    plot_radar(axs1[0], age_counts_cluster, ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'], '#1f77b4', 'Wiek')
    plot_radar(axs1[1], fav_animals_counts_cluster, ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'], '#ff7f0e', 'Ulubione zwierzęta')
    plot_radar(axs1[2], fav_place_counts_cluster, ['Nad wodą', 'W lesie', 'W górach', 'Inne'], '#2ca02c', 'Ulubione miejsca')

    st.pyplot(fig1)

    # Wykres radarowy - dane ogólne
    st.subheader("Lączna ilość znajomych")
    st.metric("Idealne dopasowanie", len(filtered_df_overall))

    # Wyliczanie i rysowanie wykresów ogólnych
    age_counts_overall = count_categories(filtered_df_overall, 'age', ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    fav_animals_counts_overall = count_categories(filtered_df_overall, 'fav_animals', ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place_counts_overall = count_categories(filtered_df_overall, 'fav_place', ['Nad wodą', 'W lesie', 'W górach', 'Inne', 'Brak ulubionych'])

    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4), subplot_kw=dict(polar=True))
    plot_radar(axs2[0], age_counts_overall, ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'], '#1f77b4', 'Wiek')
    plot_radar(axs2[1], fav_animals_counts_overall, ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'], '#ff7f0e', 'Ulubione zwierzęta')
    plot_radar(axs2[2], fav_place_counts_overall, ['Nad wodą', 'W lesie', 'W górach', 'Inne', 'Brak ulubionych'], '#2ca02c', 'Ulubione miejsca')

    st.pyplot(fig2)

with tab2:
    # Wykres słoneczny
    st.header("Zabawa z danymi, interaktywny wykres słoneczny który możesz zmieniać klikająć w 1-4 pierścień")

    # Przygotowanie danych
    df_gender_edu_age_animals_place_filtered = filter_data_for_overall(df, age, edu_level, fav_animals, fav_place, gender)[['gender', 'edu_level', 'age', 'fav_animals', 'fav_place']].dropna()

    if not df_gender_edu_age_animals_place_filtered.empty:
        # Generowanie wykresu słonecznego
        fig_sunburst_gender_edu_age_animals_place = px.sunburst(
            df_gender_edu_age_animals_place_filtered,
            path=['gender', 'edu_level', 'age', 'fav_animals', 'fav_place'],
            color='age',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        # Aktualizacja wykresu
        fig_sunburst_gender_edu_age_animals_place.update_layout(
            width=1200,
            height=1200,
            font=dict(size=16, family="Arial, sans-serif", color="black"),
            margin=dict(t=50, b=50, l=50, r=50)
            showlegend=True  # Dodanie legendy
        )

        # Dodanie procentów obok etykiet
        fig_sunburst_gender_edu_age_animals_place.update_traces(
            textinfo='label+percent parent',  # Wyświetla etykiety z procentami
            textfont=dict(size=14, family="Arial", color="black", weight='bold')
        )

        # Przypisanie do zmiennej danych kliknięcia
        click_data = st.plotly_chart(fig_sunburst_gender_edu_age_animals_place, use_container_width=True, return_data=True)

        # Funkcja obsługująca kliknięcia na segmentach wykresu
        def update_data_based_on_click(click_data):
            if click_data is not None and 'points' in click_data:
                clicked_label = click_data['points'][0]['label']
                # Filtrujemy dane na podstawie klikniętego segmentu
                filtered_data = df_gender_edu_age_animals_place_filtered[df_gender_edu_age_animals_place_filtered.apply(lambda x: clicked_label in x.values, axis=1)]
                return filtered_data
            return df_gender_edu_age_animals_place_filtered

        # Zaktualizowane dane po kliknięciu
        filtered_data = update_data_based_on_click(click_data)

        # Wyświetlanie wykresu słonecznego
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_sunburst_gender_edu_age_animals_place, use_container_width=True)

        # Opis legendy
        with col2:
            st.subheader("Opis legendy:")
            levels = ['gender', 'edu_level', 'age', 'fav_animals', 'fav_place']
            level_names = {
                'gender': '1 pierścień - Płeć',
                'edu_level': '2 pierścień - Wykształcenie',
                'age': '3 pierścień - Wiek',
                'fav_animals': '4 pierścień - Ulubione zwierzęta',
                'fav_place': '5 pierścień - Ulubione miejsce'
            }
            for level in levels:
                st.markdown(
                    f'<span style="font-weight: bold;">{level_names[level]}</span>',
                    unsafe_allow_html=True
                )
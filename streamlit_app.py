# Import library
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import folium
import requests
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Marines Pollution",
    layout="wide",
    initial_sidebar_state="expanded")

# Load data
df = pd.read_excel('Marine Pollution data.xlsx')

# --- Preprocessing ---
# Tampilkan jumlah missing values sebelum
selected_columns = [
    'obs_form_vers', 'aware_ans', 'pollutiontype_id', 'pollution_type',
    'material_id', 'material', 'pollution_desc', 'LAT_1', 'LONG', 'Country'
]

df = df[selected_columns]

# Plots

def bar(df):
    # Filter hanya yang bertipe string dan cukup banyak jumlahnya
    df_filter = df[df['pollution_type'].apply(lambda x: isinstance(x, str))]
    pollution_counts = df_filter['pollution_type'].value_counts()
    pollution_counts = pollution_counts[pollution_counts >= 100]

    types = pollution_counts.index.tolist()
    counts = pollution_counts.values.tolist()

    n_bars = len(counts)
    cmap = cm.get_cmap('Blues_r', n_bars)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_bars)]

    bar = go.Bar(
        x=counts,
        y=types,
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='Pollution Type: <b>%{y}</b><br>Count: %{x}<extra></extra>',
        hoverlabel=dict(
            bgcolor='blue',
            font=dict(color='white', size=14)
        )
    )

    layout = go.Layout(
        xaxis=dict(title='Number of Case'),
        yaxis=dict(title='Pollution Type', autorange='reversed'),
    )

    fig = go.Figure(data=[bar], layout=layout)
    fig.update_layout(width=800, height=280)
    st.plotly_chart(fig)

def bubble(df):
    # Hapus baris yang memiliki nilai NaN di kolom selain 'pollution_desc'
    df = df.dropna(subset=[col for col in df.columns if col != 'pollution_desc'])

    # Hitung jumlah material dan filter yang muncul setidaknya 10 kali
    category_counts = df['material'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    filtered_df = df[df['material'].isin(valid_categories)]
    filtered_counts = filtered_df['material'].value_counts()

    categories = filtered_counts.index.tolist()
    counts = filtered_counts.values.tolist()

    # Siapkan data untuk bubble chart
    plot_data = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Size': [count * 5 for count in counts],
        'X': [i * 0.8 for i in range(len(categories))],
        'Y': [1] * len(categories)
    })

    # Buat bubble chart menggunakan Plotly
    fig = px.scatter(
        plot_data,
        x='X',
        y='Y',
        size='Size',
        color='Count',
        hover_data={
            'X': False,
            'Y': False,
            'Category': False,
            'Count': False
        },
        custom_data=['Category', 'Count'],
        color_continuous_scale='blues',
        size_max=180
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=plot_data['X'],
            ticktext=plot_data['Category'],
            showgrid=False,
            tickfont=dict(size=15)
        ),
        yaxis=dict(showticklabels=False),
        xaxis_title='',
        yaxis_title='',
        showlegend=False,
        width=1400,
        coloraxis_colorbar=dict(title='Total Materials')
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color='grey')),
        hovertemplate="<b>Category:</b> %{customdata[0]}<br><b>Count:</b> %{customdata[1]}<extra></extra>",
        mode='markers+text',
    )

    # Tampilkan di Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Pastikan resource NLTK telah diunduh
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def wordcloud(df):

    if 'pollution_desc' not in df.columns:
        st.warning("Kolom 'pollution_desc' tidak ditemukan pada DataFrame.")
        return

    # Preprocessing kolom
    df['processed_pollution_desc'] = df['pollution_desc'].apply(preprocess_text)

    # Gabungkan semua teks hasil preprocessing
    all_descriptions = " ".join(df['processed_pollution_desc'].dropna())

    if not all_descriptions.strip():
        st.info("Tidak ada data deskripsi yang valid untuk ditampilkan dalam word cloud.")
        return

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

    # Tampilkan menggunakan matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    # Tampilkan ke Streamlit
    st.pyplot(fig)

def line(df):

    if 'obs_form_vers' not in df.columns:
        st.warning("Kolom 'obs_form_vers' tidak ditemukan pada DataFrame.")
        return

    # Hitung jumlah per versi form dan urutkan berdasarkan versi
    version_counts = df['obs_form_vers'].value_counts().sort_index()

    # Buat line chart dengan Plotly
    fig = go.Figure(data=go.Scatter(
        x=version_counts.index,
        y=version_counts.values,
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='Form Version: %{x}<br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title='Form Version',
        yaxis_title='Count',
        template='plotly_white',
        width=900,
        height=400
    )

    # Tampilkan ke Streamlit
    st.plotly_chart(fig, use_container_width=True)

def pie(df):
    if 'aware_ans' not in df.columns:
        st.warning("Kolom 'aware_ans' tidak ditemukan pada DataFrame.")
        return

    # Ganti label kategori
    df['aware_ans'] = df['aware_ans'].replace({'Y': 'Yes', 'N': 'No'})

    aware_counts = df['aware_ans'].value_counts().sort_index()
    if aware_counts.empty:
        st.info("Tidak ada data valid pada kolom 'aware_ans'.")
        return

    percentages = aware_counts / aware_counts.sum()

    # Hitung ranking untuk menentukan warna
    ranking = aware_counts.rank(method='min', ascending=False).to_dict()
    num_categories = len(aware_counts)
    step = (1.0 - 0.05) / (num_categories - 1) if num_categories > 1 else 0
    rank_to_color_value = {
        rank: 1.0 - ((rank - 1) * step)
        for rank in range(1, num_categories + 1)
    }

    colors = [mcolors.to_hex(cm.Blues(rank_to_color_value[ranking[k]])) for k in aware_counts.index]

    # Buat pie chart
    pie = go.Pie(
        labels=aware_counts.index.astype(str),
        values=aware_counts.values,
        marker=dict(colors=colors, line=dict(color='white', width=1)),
        hovertemplate='<b>Awareness: %{label}</b><br>Amount: %{value} (%{percent})<extra></extra>',
        hoverlabel=dict(
            bgcolor='azure',
            font=dict(color='black', size=16)
        ),
        sort=False,
        direction='clockwise',
        textinfo='percent',
        textfont=dict(size=16),
    )

    # Layout pie chart
    layout = go.Layout(
        width=750,
        height=450
    )

    fig = go.Figure(data=[pie], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

def geospasial(df):
    # Validasi kolom yang dibutuhkan
    if not {'Country', 'LAT_1', 'LONG'}.issubset(df.columns):
        st.warning("DataFrame harus memiliki kolom 'Country', 'LAT_1', dan 'LONG'.")
        return

    # Hitung jumlah kasus per negara
    country_counts = df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']

    # Ambil koordinat unik per negara
    country_coords = df[['Country', 'LAT_1', 'LONG']].drop_duplicates(subset=['Country'])

    # Gabungkan koordinat dan jumlah kasus
    df_country = pd.merge(country_counts, country_coords, on='Country', how='left')

    # Ambil nama negara untuk pewarnaan peta
    country_list = df_country['Country'].unique().tolist()
    count_dict = df_country.set_index('Country')['Count'].to_dict()

    # Ambil data geojson
    geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json"
    try:
        geojson_data = requests.get(geojson_url).json()
    except:
        st.error("Gagal mengambil data geojson. Periksa koneksi internet.")
        return

    # Tambahkan properti 'Count' ke geojson
    for feature in geojson_data['features']:
        country_name = feature['properties']['name']
        count = count_dict.get(country_name, None)
        feature['properties']['Count'] = int(count) if count is not None else "No data"

    # Buat peta Folium
    m = folium.Map(location=[0, 0], zoom_start=2)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': 'blue' if feature['properties']['name'] in country_list else 'gray',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'Count'],
            aliases=['Country:', 'Observations:'],
            localize=True,
            sticky=True
        )
    ).add_to(m)

    # Tampilkan peta di Streamlit
    st_folium(m, width=1100, height=400)


# ==== MAIN CONTENT ====

col1, maincol, col2 = st.columns([1, 4, 1])

with maincol:
    st.title("Marine Pollution")
    st.markdown("""
    The presence of fishing vessels plays a crucial role in supporting food security and the maritime economy in many island nations.
    These vessels serve as the backbone of fishing activities, providing a primary source of protein for coastal communities and creating employment opportunities for millions of people. 
    However, behind their vital contribution to human well-being, fishing vessels also pose a serious challenge in the form of marine pollution. 
    Pollution recorded in observation reports indicates that vessel activities, including those operating within the Exclusive Economic Zones (EEZ) of Pacific countries, contribute to the accumulation of waste and the degradation of the marine environment. 
    This phenomenon presents an irony, where the very tools used to harvest the sea become one of the main factors damaging its habitat.
    """)

    # Bar chart
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Types of Marine Pollution</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1.5])

        with col1:
            bar(df)
            st.markdown("Fig 1: Number of pollution incidents by type, dominated by waste dumped overboard and oil leakages.")

        with col2:
            st.markdown("""
            Data shows that the types of pollution are dominated by waste dumped overbroad, followed by oil spillages and leakages, as well as abandoned or lost fishing gear. 
            Unfortunately, the materials involved in this pollution consist of substances that are difficult to decompose, such as plastics, metals, old fishing gear, waste oils, and others.
            
            This pollutant is known for its long life in the marine environment, often taking hundreds of years to decompose. 
            Their continued presence threatens marine biodiversity, disrupts the food chain, and introduces microplastics into the ecosystem which can ultimately affect human health.
            """)

    # Bubble chart
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Most Common Polluting Materials</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)
        bubble(df)
        st.markdown("Fig 2: Most frequent materials found in marine pollution, including plastic, metal, and old fishing gear.")

    # Word Cloud
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Dominant Words in Pollution Descriptions</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)
        st.markdown("""
        The word cloud visualization of marine pollution descriptions provides a clear picture of the most dominant types of waste polluting the waters, particularly as a result of fishing vessel activities. 
        Words such as "plastic," "bag," "bottle," "net," and "food wrapper" appear with high frequency, indicating that single-use plastic waste and remnants of fishing activities are the main contributors to the pollution. 
        Additionally, terms like "empty," "contain," "oil," and "drum" suggest the presence of liquid waste and chemical substances from vessels, such as used oil or discarded fuel containers. 
        This waste not only causes visual pollution but also has serious impacts on the marine ecosystem, endangering sea creatures that may mistake plastic for food.
        """)
        wordcloud(df)
        st.markdown("Fig 3: Most frequent terms in pollution descriptions, highlighting specific polluting items like plastic bags and oil drums.")

    # Geo chart
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Geographic Spread of Marine Pollution</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)
        st.markdown("""
        The following are the results of observations on marine pollution caused by fishing vessel activities in several island countries such as Papua New Guinea, Micronesia, and others.
        """)
        geospasial(df)
        st.markdown("Fig 4: Distribution of marine pollution cases reported in island nations such as Papua New Guinea and Micronesia.")

    # Line chart
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Trend of Marine Pollution Reports Over Time</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2.5, 1.5])
        with col1:
            line(df)
            st.markdown("Fig 5: Yearly frequency of marine pollution reports, reflecting ongoing and possibly increasing vessel-related waste discharge.")
        
        with col2:
            st.markdown("""Observations on marine pollution caused by fishing vessel activities have been carried out for quite a long time. 
            There have been many reports each year, as illustrated in the chart.
            """)            
            st.markdown("""This consistent reporting pattern suggests that marine pollution is not a one-off or rare event, but rather a problem that has persisted for many years. 
            Such persistence suggests systemic problems—whether in law enforcement, awareness, or operational practices on board vessels—that have not been effectively addressed.
            """)
            st.markdown("""Furthermore, the data underscore the need for long-term monitoring and strategic intervention. 
            The presence of a stable or even increasing trend in pollution reporting suggests that current mitigation efforts may be inadequate.
            """)

    # Pie chart
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Awareness of Marine Pollution</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.8, 1.7])  # grafik agak kecil, teks bisa lebih luas

        with col1:
            st.markdown("""
                <div style="margin-top: 4rem;"> </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            After numerous reports, are fishermen actually aware of the importance of protecting the marine environment by not polluting it?
                        
            As many as 57.4% of fishermen are already aware of the importance of protecting the marine environment. 
            However, the rest still need to be educated about the significance of environmental conservation, especially considering that the ocean is where they carry out their daily activities. 
            This effort is necessary so that we can work together to protect the ocean from potential pollution.
            """)
            st.markdown("""
            While it is positive to see that more than half of the respondents understand the importance of preserving the marine ecosystem, the remaining 42.6% who are still unaware of it constitute a significant portion of the population that can unknowingly contribute to further environmental damage. 
            This gap highlights the urgent need for targeted education and outreach programs that emphasize the direct impacts of pollution on their livelihoods and the long-term sustainability of fisheries.
            """)

        with col2:
            pie(df)
            st.markdown("Fig 6: 57.4% of fishermen are aware of the need to protect the marine environment, while 42.6% remain unaware.")

    # Conclusion
    with st.container():
        st.markdown("""
            <div style="margin-top: 3rem; display: flex; justify-content: space-between; align-items: flex-start; font-size: 32px">
                <strong>Conclusion</strong>
            </div>

            <hr style="margin-top: 1rem; margin-bottom: 1rem; border-top: 1px solid #BBB;" />
        """, unsafe_allow_html=True)
        st.markdown("""
        Fishing vessels are critical to sustaining livelihoods and food security in island nations, but their activities are inadvertently contributing to a growing marine pollution crisis.
        """)
        st.markdown("""
        Through various forms of waste—from plastic and oil to abandoned fishing gear—these vessels leave a damaging footprint in the ocean. 
        Visualizations from this analysis illustrate not only the types and substances of pollution but also its geographic and temporal patterns.
        """)
        st.markdown("""
        While awareness among some fishermen is promising, much more needs to be done to foster a culture of environmental stewardship in maritime communities. 
        Addressing this issue requires everyone to ensure that the oceans remain a viable source of life for future generations.
        """)

    with st.container():
        st.markdown("""
            <hr style="margin-top: 3rem; margin-bottom: 1rem; border-top: 0px solid #BBB;" />
        """, unsafe_allow_html=True)

        st.markdown("""
            The data is sourced from the [Pacific Data Hub](https://pacificdata.org/) and uses data from the [Marine Pollution](https://pacificdata.org/data/dataset/marine-pollutionb7c085de-2247-4d9a-bb01-6aef1fbe25a2) report on marine pollution in the Pacific region.
            """)
        
        st.markdown("""
            This project was created by a group consisting of **Azra Feby Awfiyah**, **Diva Sanjaya Wardani**, **Farah Saraswati**, and **Nasywa Alif Widyasari** for the [Pacific Data Challenge](https://pacificdatavizchallenge.org/).
            """)

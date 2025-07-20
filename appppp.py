import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import requests
import random


# --- Password Protection ---
password = st.text_input("Enter password to access the dashboard:", type="password")
if password != "Exoplanet25":
    st.warning("Incorrect password. Try again.")
    st.stop()

# --- App Configuration ---
st.set_page_config(layout="wide")

# --- Initialise session state for subtitles and presentation mode ---
if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False # Default to off

# --- Function to check if the device is iOS ---
def is_ios():
    return st.session_state.get("is_ios_detected", False)

# --- Sidebar Settings ---
with st.sidebar:
    st.title("🔧 Settings")
    
    # Dark Mode toggle remains
    dark_mode = st.toggle("🌙 Enable Dark Mode", value=True) 
    font_size = st.slider("🔠 Base Font Size (px)", 12, 28, 16)
    speech_rate = st.slider("🗣️ Speech Rate", 0.5, 2.0, 1.0, 0.1)
    st.caption(f"🔊 Current speed: {speech_rate}x")

    st.markdown("---")
    st.subheader("📽️ Presentation Tools")
    
    # Presentation Mode toggle, directly updates session state
    st.session_state.presentation_mode = st.toggle(
        "✨ Presentation Mode", 
        value=st.session_state.presentation_mode, # Use current session state value
        help="Adjusts chart fonts, colors, and layouts for better visibility during presentations."
    )

    st.markdown("---")
    st.subheader("📱 Device Simulation")

    # iOS Simulation Toggle to sidebar's top level ---
    if st.button("Toggle iOS Simulation", key="sidebar_ios_toggle"):
        st.session_state["is_ios_detected"] = not st.session_state["is_ios_detected"]
        st.write(f"**iOS Simulated:** {st.session_state['is_ios_detected']}")


# --- Apply Global Streamlit UI Theme based on dark_mode ---
theme = "plotly_dark" if dark_mode else "plotly_white" 
bg_color = "#0e1117" if dark_mode else "white"
text_color = "white" if dark_mode else "black"

st.markdown(
    f"""
    <style>
    .main {{
        background-color: {bg_color};
        color: {text_color};
    }}
    html, body, [class*="css"] {{
        font-size: {font_size}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper functions for dynamic styling based on presentation_mode ---

def get_presentation_fonts():
    """Returns font sizes based on the global presentation_mode state."""
    is_on = st.session_state.presentation_mode
    if is_on:
        return {
            "base_font": 20,
            "title_font": 26,
            "axis_font": 18,
            "legend_font": 16,
            "marker_size": 7 
        }
    else:
        return {
            "base_font": 14,
            "title_font": 18,
            "axis_font": 12,
            "legend_font": 11,
            "marker_size": 4 
        }

def get_chart_styling(current_theme, is_dark_mode):
    """
    Returns Plotly layout updates based on presentation_mode and current theme,
    only forcing pure black background if dark_mode is also active in presentation mode.
    """
    fonts = get_presentation_fonts()

    layout_updates = {
        "font": dict(size=fonts["base_font"]),
        "title": dict(font=dict(size=fonts["title_font"])),
        "xaxis": dict(title=dict(font=dict(size=fonts["axis_font"]))),
        "yaxis": dict(title=dict(font=dict(size=fonts["axis_font"]))),
        "legend": dict(font=dict(size=fonts["legend_font"])),
        "template": current_theme 
    }

    if st.session_state.presentation_mode and is_dark_mode:
        layout_updates["plot_bgcolor"] = "#000000" 
        layout_updates["paper_bgcolor"] = "#000000" 
        layout_updates["xaxis"]["gridcolor"] = "#333333" 
        layout_updates["yaxis"]["gridcolor"] = "#333333" 
        layout_updates["xaxis"]["tickfont"] = dict(color="white")
        layout_updates["yaxis"]["tickfont"] = dict(color="white")
        layout_updates["xaxis"]["title"]["font"]["color"] = "white"
        layout_updates["yaxis"]["title"]["font"]["color"] = "white"
        layout_updates["title"]["font"]["color"] = "white"
        layout_updates["legend"]["font"]["color"] = "white"

    return layout_updates


# --- Browser-based TTS Function with Controls ---
def speak_text_via_browser(text, rate=1.0):
    escaped_text = text.replace("'", "\\'").replace("\n", " ").replace("`", "'")

    components.html(f"""
        <script>
            const is_ios = /iPad|iPhone|iPod/.test(navigator.userAgent);
            if (!is_ios) {{
                
                window.currentSpeechMsg = new SpeechSynthesisUtterance('{escaped_text}');
                window.currentSpeechMsg.rate = {rate};
                
                window.speechSynthesis.cancel();

                window.speechSynthesis.speak(window.currentSpeechMsg);

                window.currentSpeechMsg.onend = function(event) {{
                    console.log('Speech finished in browser.');
                }};
                window.currentSpeechMsg.onerror = function(event) {{
                    console.error('Speech error in browser:', event.error);
                }};
            }} else {{
                console.log("Not speaking via browser for iOS device.");
            }}
        </script>
    """, height=0)

def is_ios():
    return st.session_state.get("is_ios_detected", False)

# --- iOS-specific TTS Function ---
# This function is specifically for iOS devices, as they handle TTS differently.
def speak_text_for_ios(text, rate=1.0):
    escaped_text = text.replace("'", "\\'").replace("\n", " ").replace("`", "'")

    components.html(f"""
        <div id="ios-tts-container"></div>
        <script>
            const is_ios = /iPad|iPhone|iPod/.test(navigator.userAgent);

            if (is_ios) {{
                const container = document.getElementById("ios-tts-container");
                // Clear previous button to prevent duplicates on reruns
                while(container.firstChild) {{
                    container.removeChild(container.firstChild);
                }}

                const button = document.createElement("button");
                button.textContent = "🔊 Speak (iOS)";
                button.style.fontSize = "16px";
                button.style.marginTop = "10px";
                button.id = "ios-speak-button"; 

                // Attach the event listener to the button
                button.onclick = function() {{
                    // Make msg a global variable within this iframe's context
                    window.currentIOSSppechMsg = new SpeechSynthesisUtterance('{escaped_text}');
                    window.currentIOSSppechMsg.rate = {rate};

                    // Clear any previous speech queue
                    window.speechSynthesis.cancel();

                    // Speak the new message
                    window.speechSynthesis.speak(window.currentIOSSppechMsg);

                    // Optional: Add event listeners for debugging
                    window.currentIOSSppechMsg.onend = function(event) {{
                        console.log('Speech finished on iOS button.');
                    }};
                    window.currentIOSSppechMsg.onerror = function(event) {{
                        console.error('Speech error on iOS button:', event.error);
                    }};
                }};

                container.appendChild(button);
                console.log("iOS speak button rendered.");
            }}
        </script>
    """, height=100)


# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads and preprocesses the exoplanet dataset."""
    df = pd.read_csv("Cleaned Exoplanet Dataset.csv")
    df.rename(columns={"Name": "Planet name"}, inplace=True)

    df['Disc. Year'] = pd.to_numeric(df['Disc. Year'], errors='coerce')

    def classify_star_type(temp):
        if temp < 3500: return "M-type (Cool)"
        elif temp < 5000: return "K-type"
        elif temp < 6000: return "G-type (like our Sun)"
        elif temp < 7500: return "F-type"
        elif temp < 10000: return "A/B-type"
        else: return "O-type (Very Hot)"

    df["Stellar Type"] = df["star temp (clean)"].apply(classify_star_type)
    return df

@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost model."""
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")
    return model

data = load_data()
model = load_model()

# --- Charting Functions (Updated to use global state for styling and local subtitle) ---

# Helper to display subtitle locally for each chart
def display_chart_subtitle(chart_key, description):
    # Initialise a specific subtitle for this chart if it doesn't exist
    if f'subtitle_{chart_key}' not in st.session_state:
        st.session_state[f'subtitle_{chart_key}'] = ""

    # Button for all platforms: sets subtitle + speaks (on Android/Windows)
    if st.button("🔊 Describe Chart", key=f"tts_{chart_key}"):
        st.session_state[f'subtitle_{chart_key}'] = description
        speak_text_via_browser(description, rate=speech_rate)

    # Separate iOS button: speaks the same description
    if is_ios():
        if st.button("🔊 Speak (iOS only) & Show Subtitle", key=f"tts_ios_streamlit_{chart_key}"):
            st.session_state[f'subtitle_{chart_key}'] = description
            speak_text_for_ios(description, rate=speech_rate)

    # Show subtitle on screen (on *all* platforms)
    if st.session_state[f'subtitle_{chart_key}']:
        st.markdown(f"**🗒️ Subtitle:** {st.session_state[f'subtitle_{chart_key}']}")



# --- Function to fetch a random exoplanet from NASA Exoplanet Archive ---
def fetch_random_exoplanet():
    """
    Fetches a random exoplanet's details from the NASA Exoplanet Archive TAP service.
    Returns a dictionary of planet and host star properties, or None if unsuccessful.
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = """
    SELECT pl_name, pl_rade, pl_bmasse, pl_orbper, st_teff, st_mass, st_rad
    FROM ps
    WHERE pl_name IS NOT NULL
    AND pl_rade IS NOT NULL
    AND pl_bmasse IS NOT NULL
    AND pl_orbper IS NOT NULL
    AND st_teff IS NOT NULL
    AND st_mass IS NOT NULL
    AND st_rad IS NOT NULL
    """ 
    params = {"query": query, "format": "json"}
    
    try:
        response = requests.get(url, params=params, timeout=20) # Increased timeout slightly
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Performing the random sample using pandas after fetching
            return df.sample(1).iloc[0].to_dict()
        else:
            return None # No data returned
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Exoplanet Archive: {e}")
        return None
    except ValueError as e: # For JSON decoding errors
        print(f"Error decoding JSON response: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return None


# --- Chart Functions ---
# Function to plot average host star temperature by discovery method
# This function creates a bar chart showing the average temperature of host stars grouped by discovery method.
def plot_avg_temp_by_discovery(df):
    st.subheader("Average Host Star Temperature by Discovery Method")
    subset = df[['Discovery method', 'star temp (clean)']].dropna()
    avg_temps = subset.groupby('Discovery method')['star temp (clean)'].mean().sort_values(ascending=False)

    fig = px.bar(avg_temps, x=avg_temps.index, y=avg_temps.values,
                 labels={'x': 'Discovery Method', 'y': 'Average Star Temperature (K)'},
                 color=avg_temps.values, color_continuous_scale=px.colors.sequential.Mint)
    fig.update_layout(xaxis_tickangle=-45)
    
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_method = avg_temps.idxmax()
        st.metric("🔥 Hottest Method", top_method, f"{int(avg_temps.max())} K")

    description = "This bar chart shows the average host star temperature for each discovery method."
    display_chart_subtitle("avg_temp", description)


# --- Function to plot planet mass vs host star temperature scatter plot
# This function creates a scatter plot showing the relationship between planet mass and host star temperature.
def plot_mass_vs_temp_scatter(df):
    st.subheader("Planet Mass vs. Host Star Temperature")
    subset = df.dropna(subset=['Mass (MJ)', 'star temp (clean)'])
    
    fonts = get_presentation_fonts()
    
    fig = px.scatter(subset, x='Mass (MJ)', y='star temp (clean)', color='Stellar Type',
                     hover_data=['Planet name', 'Radius (RJ)', 'Discovery method'],
                     title='Planet Mass vs Host Star Temperature',
                     labels={'Mass (MJ)': 'Planet Mass (Jupiter Mass)', 'star temp (clean)': 'Star Temperature (K)'})
    
    fig.update_traces(marker=dict(size=fonts["marker_size"]))
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("🪐 Total Planets Shown", value=len(subset))
    
    description = "This scatter plot shows the relationship between planet mass and host star temperature."
    display_chart_subtitle("mass_vs_temp", description)

# --- Function to plot star temperature by star mass using a strip plot
# This function creates a strip plot showing the distribution of star temperatures across different mass bins.
def plot_temp_by_mass_strip(df):
    st.subheader("Star Temperature by Star Mass")
    df['star_mass_bin'] = pd.cut(df['star mass (clean)'], bins=4).astype(str)
    subset = df[['star_mass_bin', 'star temp (clean)']].dropna()

    fonts = get_presentation_fonts()

    fig = px.strip(subset, x='star_mass_bin', y='star temp (clean)', color='star_mass_bin',
                   title='Distribution of Star Temperature by Star Mass Bin',
                   labels={'star_mass_bin': 'Star Mass Bins (Solar Mass)', 'star temp (clean)': 'Star Temperature (K)'})

    fig.update_traces(marker=dict(size=fonts["marker_size"]))
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("🎼 Avg Temp", f"{int(subset['star temp (clean)'].mean())} K")

    description = "This strip plot shows star temperatures across different mass bins."
    display_chart_subtitle("temp_by_mass", description)

# --- Function to plot the distribution of planet radius using a histogram
# This function creates a histogram showing the distribution of exoplanet radii.
def plot_radius_distribution_hist(df):
    st.subheader("Distribution of Planet Radius")
    subset = df['radius (clean)'].dropna()
    fig = px.histogram(subset, nbins=30, title="Distribution of Planet Radius",
                       labels={'value': 'Radius (Jupiter Radii)'},
                       color_discrete_sequence=['skyblue'])
    
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("🌍 Mean Radius (RJ)", value=round(subset.mean(), 2))
    
    description = "This histogram shows the distribution of exoplanet radii."
    display_chart_subtitle("radius_distribution", description)

# --- Function to plot a correlation heatmap of numeric features
# This function creates a heatmap showing the correlation between numeric features in the dataset.
def plot_correlation_heatmap(df):
    st.subheader("Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)
    corr = numeric_cols.corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                     colorscale='Viridis', zmin=-1, zmax=1))
    fig.update_layout(title="Correlation Heatmap of Numeric Features")
    
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_corr = upper.abs().unstack().nlargest(1)
        st.metric(label="🔗 Strongest Correlation",
                  value=f"{max_corr.index[0][0].split(' ')[0]} & {max_corr.index[0][1].split(' ')[0]}",
                  delta=f"{max_corr.iloc[0]:.2f}")

    description = "This heatmap shows the correlation between numerical features."
    display_chart_subtitle("correlation_heatmap", description)
    
# --- Function to plot the count of planets by discovery method
# This function creates a bar chart showing the number of exoplanets discovered by each method.
def plot_discovery_method_counts(df):
    st.subheader("Count of Planets by Discovery Method")
    method_counts = df['Discovery method'].value_counts()
    fig = px.bar(method_counts, x=method_counts.index, y=method_counts.values,
                 labels={'x': 'Discovery Method', 'y': 'Number of Planets'},
                 color=method_counts.values, color_continuous_scale='Plasma')
    fig.update_layout(xaxis_tickangle=-45)
    
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    st.plotly_chart(fig, use_container_width=True)

    description = "This bar chart shows the number of exoplanets found by each discovery method."
    display_chart_subtitle("discovery_method", description)

# --- Function to plot discovery methods over time using a swarm plot
# This function creates a swarm plot showing how discovery methods have evolved over the years.
def plot_discovery_swarm_over_time(df): 
    st.subheader("Discovery Methods Over the Years")
    subset = df[['Discovery method', 'Disc. Year']].dropna()

    fonts = get_presentation_fonts()
    
    fig = px.strip(
        subset,
        x='Disc. Year',
        y='Discovery method',
        color='Discovery method',
        title='Exoplanet Discovery Methods Over the Years'
    )

    fig.update_traces(marker=dict(size=fonts["marker_size"]))
    fig.update_layout(get_chart_styling(theme, dark_mode)) 

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not subset.empty:
            st.metric("🗓️ Range of Years", f"{int(subset['Disc. Year'].min())}–{int(subset['Disc. Year'].max())}")
        else:
            st.metric("🗓️ Range of Years", "N/A")

    description = "This swarm plot shows how discovery methods have evolved over time."
    display_chart_subtitle("discovery_swarm", description)


# --- Main App ---
st.title("🚀 Exoplanet Explorer Dashboard")
st.write("An interactive dashboard to explore, visualise, and analyse exoplanet data.")


# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Explore", "🔮 Predict", "🔭 Discover", "ℹ️ About"])

CHART_FUNCTIONS = {
    "Average Host Star Temp by Discovery Method": plot_avg_temp_by_discovery,
    "Planet Mass vs Star Temp (Scatter)": plot_mass_vs_temp_scatter,
    "Star Temp by Star Mass (Strip Plot)": plot_temp_by_mass_strip,
    "Distribution of Planet Radius (Histogram)": plot_radius_distribution_hist,
    "Feature Correlation Heatmap": plot_correlation_heatmap,
    "Count of Planets by Discovery Method": plot_discovery_method_counts,
    "Discovery Methods Over the Years (Swarm Plot)": plot_discovery_swarm_over_time,
}

# --- Explore Tab ---
with tab1:
    st.header("Explore Exoplanet Trends")
    # Store the previous chart option to detect changes
    previous_chart_option = st.session_state.get('last_chart_option', None)
    
    chart_option = st.selectbox("Choose a chart to explore:", options=list(CHART_FUNCTIONS.keys()))
    st.info("📌 Pro tip: Change the chart type to explore new insights.")
    st.markdown("---")

    # If the chart option changed, clear all specific chart subtitles
    if chart_option != previous_chart_option:
        for key in st.session_state:
            if key.startswith('subtitle_'):
                st.session_state[key] = ""
    st.session_state['last_chart_option'] = chart_option # Update the last selected option

    selected_chart_func = CHART_FUNCTIONS[chart_option]
    selected_chart_func(data)


# --- Predict Tab ---
with tab2:
    st.header("🔮 Predict Host Star Temperature")
    st.markdown("Select an exoplanet to pre-fill its values, or adjust the sliders manually to predict the host star's temperature.")

    # Dropdown for planet selection
    planet_names = [""] + sorted(data["Planet name"].dropna().unique().tolist())
    selected_planet = st.selectbox("📌 Choose an exoplanet (optional):", planet_names)

    # Default values
    defaults = {
        'distance': 500.0, 'radius': 1.0, 'mass': 1.0, 
        'star_mass': 1.0, 'period': 365.0
    }

    if selected_planet:
        planet_data = data[data["Planet name"] == selected_planet].iloc[0]
        defaults['distance'] = planet_data.get('distance (clean)', defaults['distance'])
        defaults['radius'] = planet_data.get('radius (clean)', defaults['radius'])
        defaults['mass'] = planet_data.get('mass (clean)', defaults['mass'])
        defaults['star_mass'] = planet_data.get('star mass (clean)', defaults['star_mass'])
        defaults['period'] = planet_data.get('Period (days)', defaults['period'])

        # Show discovery details
        st.markdown("### 🪐 Planet Details")
        st.markdown(f"- **Discovery Method:** {planet_data['Discovery method']}")
        st.markdown(f"- **Discovery Year:** {int(planet_data['Disc. Year']) if pd.notna(planet_data['Disc. Year']) else 'Unknown'}")
        st.markdown(f"- **Host Star Temperature:** {int(planet_data['star temp (clean)'])} K")
        st.markdown(f"- **Stellar Type:** {planet_data['Stellar Type']}")
        st.markdown("---")
        st.info("🔄 Values below are pre-filled. Feel free to tweak them.")

    # Sliders for input
    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("Distance (ly)", 0.0, 12000.0, defaults['distance'])
        radius = st.slider("Planet Radius (RJ)", 0.0, 3.0, defaults['radius'])
        mass = st.slider("Planet Mass (MJ)", 0.0, 40.0, defaults['mass'])
    with col2:
        star_mass = st.slider("Host Star Mass (M☉)", 0.1, 5.0, defaults['star_mass'])
        period = st.slider("Orbital Period (days)", 0.1, 5000.0, defaults['period'])

    # Prediction
    if st.button("Predict Star Temperature"):
        input_data = {
            'distance (clean)': distance,
            'radius (clean)': radius,
            'Distance_Radius_Interaction': distance * radius,
            'mass (clean)': mass,
            'star mass (clean)': star_mass,
            'Period (days)': period,
            'Mass_x_Radius': mass * radius,
            'Log_Period': np.log1p(period)
        }
        features_ordered = [
            'distance (clean)', 'radius (clean)', 'Distance_Radius_Interaction', 'mass (clean)', 
            'star mass (clean)', 'Period (days)', 'Mass_x_Radius', 'Log_Period'
        ]
        input_df = pd.DataFrame([input_data], columns=features_ordered)
        prediction = model.predict(input_df)
        st.success(f"🌟 Predicted Host Star Temperature: **{prediction[0]:.0f} K**")


# --- Discover Tab (Content for tab3 from st.tabs) ---
with tab3: 
    # --- Discover Tab (Content for tab3 from st.tabs) ---
    st.header("🔭 Discover a Random Exoplanet")
    st.markdown("Click the button below to fetch details about a randomly selected exoplanet from the NASA Exoplanet Archive!")

    # Using session state to persist the last fetched planet
    if 'current_random_planet' not in st.session_state:
        st.session_state.current_random_planet = None

    if st.button("🎲 Surprise Me with a Planet"):
        with st.spinner("🌀 Searching the cosmos..."):
            st.session_state.current_random_planet = fetch_random_exoplanet()
            

    planet = st.session_state.current_random_planet

    if planet: 
        # Defining variables here, only if a planet was successfully fetched
        planet_name = planet.get('pl_name', 'Unknown Planet')
        planet_rade = planet.get('pl_rade', 'N/A')
        planet_bmasse = planet.get('pl_bmasse', 'N/A')
        planet_orbper = planet.get('pl_orbper', 'N/A')
        st_teff = planet.get('st_teff', 'N/A')
        st_mass = planet.get('st_mass', 'N/A')
        st_rad = planet.get('st_rad', 'N/A')

        # Layout: Left (planet name + gif), Right (info + speak)
        left_col, right_col = st.columns([3,2]) 

        with left_col:
            st.markdown("#### 🪐 Planetary Properties")
            st.markdown(f"- Radius: {planet_rade} Earth radii")
            st.markdown(f"- Mass: {planet_bmasse} Earth masses")
            st.markdown(f"- Orbital Period: {planet_orbper} days")

            st.markdown("#### ☀️ Host Star")
            st.markdown(f"- Temperature: {st_teff} K")
            st.markdown(f"- Mass: {st_mass} Solar masses")
            st.markdown(f"- Radius: {st_rad} Solar radii")

            # Define spoken_text 
            spoken_text = (
                f"Here are the details for Exoplanet {planet_name}. "
                f"Planetary Properties: Radius: {planet_rade} Earth radii. "
                f"Mass: {planet_bmasse} Earth masses. "
                f"Orbital Period: {planet_orbper} days. "
                f"Host Star properties: Temperature: {st_teff} Kelvin. "
                f"Mass: {st_mass} Solar masses. "
                f"Radius: {st_rad} Solar radii."
            )
            if st.button("🔊 Read Planet Info", key="read_planet_info_button_unique"):
                speak_text_via_browser(spoken_text, rate=speech_rate)

        with right_col:
            st.markdown(
                f"<h2 style='text-align:center;'>🌠 <span class='planet-name-span'>{planet_name}</span></h2>",
                unsafe_allow_html=True
            )
            
            
            st.image("assets/Ring_Planet.gif", width=200) 
            

    elif st.session_state.current_random_planet is not None:
        # This part runs if the button was clicked but fetch_random_exoplanet returned None
        st.error("Couldn’t find a planet this time. The galaxy must be busy—try again!")

    # The 'else' case for 'planet' (when current_random_planet is None at page load)
    # is implicitly handled, as nothing will be displayed if 'if planet:' is false.

# --- About Tab (Content for tab4 from st.tabs) ---
with tab4: 
    st.header("ℹ️ About This Dashboard")

    about_text = """
    Welcome to the Exoplanet Explorer! 🚀  
    This interactive dashboard lets you dive into real astronomical data, explore patterns in planetary systems, and even make predictions using a machine learning model.
    
    Feel free to explore the different charts, use the accessibility features in the sidebar, and try the predictive model! 
    """

    purpose_text_display = """
    - **Purpose:** To visualise trends in exoplanet discovery, physical properties, and stellar characteristics.  
    - **Built with:** `Streamlit`, `Pandas`, `Plotly`, and `XGBoost`.
    """

    purpose_text_spoken = (
        "This dashboard was created to explore trends in exoplanet discovery, "
        "planetary properties, and stellar characteristics. "
        "It was built using Streamlit, Pandas, Plotly, and XGBoost."
    )

    # 📝 Intro section
    st.markdown(about_text)
    if st.button("🔊 Speak Intro", key="tts_intro"):
        speak_text_via_browser(about_text, rate=speech_rate)

    # 📝 Purpose section
    st.markdown(purpose_text_display)
    if st.button("🔊 Speak Purpose", key="tts_purpose"):
        speak_text_via_browser(purpose_text_spoken, rate=speech_rate)
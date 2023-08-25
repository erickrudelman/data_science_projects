import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Display the image at the beginning of the app
image_path = 'VVD.jpeg'
st.image(image_path, caption='Virgil van Dijk', use_column_width=True)

st.title('EPL Player Stats 21/22 ')

st.markdown("""
Exploration of EPL players stats data!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Kaggle.com](https://www.kaggle.com/datasets/jkanthony/premier-league-stats-top-50-players).
""")

st.sidebar.header('User Input Features')

# Load the EPL dataset (replace with your file path)
csv_file_path = 'Premier_League_players.csv'
df = pd.read_csv(csv_file_path)

# Filter out teams with ',' in their name
df = df[~df['Team'].str.contains(',')].copy()

#Display data set
df

#Display all columns list
df.columns.tolist()


# Sidebar - Team selection (using EPL club names)
selected_team = st.sidebar.multiselect('Club', df['Team'].unique(), df['Team'].unique())


# Filtering data based on user input
df_selected = df[df['Team'].isin(selected_team)]

st.header('Display Player Stats of Selected Club(s)')
st.write('Data Dimension: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
st.dataframe(df_selected)

# Download EPL player stats data as CSV
def download_link(df, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="epl_player_stats.csv">{text}</a>'
    return href

st.markdown(download_link(df_selected, 'Download CSV File'), unsafe_allow_html=True)

#A scatter plot is created to show the correlation between the 'Min' column and both the 'G' (goals) and 'A' (assists) columns for the selected players within the selected team(s).
#The user can select specific players from the selected team(s) using the sidebar.

# Initialize session state to store user selections
if 'selected_players_a' not in st.session_state:
    st.session_state.selected_players_a = []

if 'selected_players_g' not in st.session_state:
    st.session_state.selected_players_g = []


#PLAYER COMPARISON
#Player Selection
selected_players_for_comparison = st.sidebar.multiselect('Select Players for Comparison', df_selected['Player'].unique(), [], key='comparison_players')
# Outside the if-statement, define df_selected_players
df_selected_players = df_selected[df_selected['Player'].isin(selected_players_for_comparison)]


#Stat Selection
selected_stats_for_comparison = st.sidebar.selectbox('Select Stats for Comparison', ['Min', 'A', 'G'], key='comparison_stats')

# Inside the if-statement, perform the comparative analysis
if st.button('Compare Player Stats'):
    st.header(f'Comparative Analysis: {selected_stats_for_comparison} vs {selected_players_for_comparison}')
    
    # Create a scatter plot for the selected stats
    fig, ax = plt.subplots(figsize=(10, 6))
    scatters = []
    for player in selected_players_for_comparison:
        player_data = df_selected_players[df_selected_players['Player'] == player]
        scatter = ax.scatter(player_data['Min'], player_data[selected_stats_for_comparison], alpha=0.7)
        scatters.append((scatter, player))
    
    ax.set_xlabel('Minutes Played (Min)')
    ax.set_ylabel(selected_stats_for_comparison)
    ax.set_title(f'Comparative Analysis: {selected_stats_for_comparison} vs Minutes Played')
    
    # Adjust label positions for better readability
    for scatter, player in scatters:
        for i, (x, y) in enumerate(scatter.get_offsets()):
            ax.annotate(player, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.legend([scatter for scatter, player in scatters], [player for scatter, player in scatters], 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', markerscale=0.6)
    
  
    plt.tight_layout()  # Adjust plot layout for better visualization
    st.pyplot(fig)

# Remove teams with ',' in their name
df_selected = df_selected[~df_selected['Team'].str.contains(',')]

# Compute statistics for the top 10 teams with most goals and assists
top_teams_goals = df_selected.groupby('Team')['G'].sum().nlargest(10)
top_teams_assists = df_selected.groupby('Team')['A'].sum().nlargest(10)

# Display the statistics for the top teams
st.header('Top 10 Teams with Most Goals')
st.write(top_teams_goals)

st.header('Top 10 Teams with Most Assists')
st.write(top_teams_assists)

# Display statistics for top 4 players in goals and assists for each top team with clickable tables
st.header('Top Players in Goals and Assists for Top 10 Teams')
for team in top_teams_goals.index.get_level_values(0):
    if st.button(f'Top Players - {team}'):
        st.subheader(f'Top Players for Team: {team}')
        top_players_goals = df_selected[df_selected['Team'] == team].nlargest(4, 'G')
        top_players_assists = df_selected[df_selected['Team'] == team].nlargest(4, 'A')
        
        st.write('Top Players in Goals:')
        st.table(top_players_goals[['Player', 'G', 'A']])
        
        st.write('Top Players in Assists:')
        st.table(top_players_assists[['Player', 'G', 'A']])

# Compute the correlation matrix between player statistics and team performance
correlation_matrix = df_selected.corr()

# Create a heatmap to visualize the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)

# Set heatmap title and labels
plt.title('Correlation Heatmap between Player Stats and Team Performance')
plt.xlabel('Player Statistics')
plt.ylabel('Player Statistics')

# Display the heatmap in Streamlit
st.pyplot(fig)


# Explanation about the heatmap
st.markdown("""
The values in each cell of the heatmap represent the correlation coefficient between two player statistics. 
The values range from -1 to 1. A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation. 
A value close to 0 indicates a weak or no correlation.

The color map (cmap) is set to 'coolwarm', where cooler colors represent negative correlations, warmer colors represent positive correlations, and white represents a correlation close to 0.

We set the title of the heatmap to 'Correlation Heatmap between Player Stats and Team Performance'.

We label the x-axis and y-axis with the names of player statistics to indicate which statistics are being compared.
""")


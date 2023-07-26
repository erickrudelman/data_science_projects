# Import required libraries
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

#Download dataset
spacex_df =  pd.read_csv(spacex_launch_dash.csv)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()



# Create a dash application
app = dash.Dash(__name__)
                               
app.layout = html.Div(children=[ html.H1('SpaceX Launches', 
                                style={'textAlign': 'center', 'color': '#503D36',
                                'font-size': 40}),
                                  dcc.Dropdown(id='id',
                                                options=[
                                                        {'label':'CCAFS LC-40', 'value':'CCAFS LC-40'},
                                                        {'label':'VAFB SLC-4E', 'value':'VAFB SLC-4E'},
                                                        {'label':'KSC LC-39A', 'value':'KSC LC-39A'},                                                    
                                                        {'label':'CCAFS SLC-40', 'value':'CCAFS SLC-40'}
                                                ],
                                                value='ALL',
                                                placeholder="Select a Launch Site here",
                                                searchable=True
                                                ),
                                html.Br(),

                                # Add a pie chart                                
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                                min=0,
                                                max=10000,
                                                step=1000,
                                                value=[min_payload, max_payload]
                                                ),
                                
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),

                                ])
#TASK 2: Add a callback function to render success-pie-chart bases on selected site dropdown
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))

def get_pie_chart(entered_site):    
    if entered_site == 'ALL':
        fig = px.pie(spacex_df, values='class', 
        names='Launch Sites', 
        title='Launch Success by Site')
        return fig
    else:
        specificlaunch_df = spacex_df[spacex_df['Launch Site']==entered_site]
        specificlaunch_df = specificlaunch_df.groupby(['Launch Site', 'class']).size().reset_index(name='class count')
        fig = px.pie(specificlaunch_df,
                    values='class count',
                    names='class',
                    title=f'The Success Launches for the Site {entered_site}')
        return fig
        
# TASK 4: Add a callback function to render the `success-payload-scatter-chart` scatter plot.
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'),
              Input(component_id='payload-slider', component_property='value')]
              )
def scatterplot(entered_site, payload):
    filtered_df = spacex_df[spacex_df['Payload Mass (kg)'].between(payload[0], payload[1])]
    if entered_site == 'ALL':
        fig = px.scatter(filtered_df,
                        x='Payload Mass (kg)',
                        y='class',
                        color='Booster Version Category',
                        title='Link of Payload and Success for all Sites'
                        )
        return fig
    else:
        fig = px.scatter(filtered_df[filtered_df['Launch Site']==entered_site],
                        x='Payload Mass (kg)',
                        y='class',
                        color='Booster Version Category',
                        title=f'Link of Payload and Success for site {entered_site}'
                        )
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server()

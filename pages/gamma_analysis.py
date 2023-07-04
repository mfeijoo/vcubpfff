import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import streamlit.components.v1 as components
#from io import StringIO
import boto3
import pymedphys

st.title("VCU & Blue Physics Gamma Analysis")

s3 = boto3.client('s3')

response = s3.list_objects_v2(Bucket='bluephysicsaws', Delimiter='/')

st.markdown('## Select file A:')

listofdirectoriesA = []

for common_prefix in response.get('CommonPrefixes', []):
        # Extract the folder name from the CommonPrefixes
        folder_name = common_prefix['Prefix'].rstrip('/')
        listofdirectoriesA.append(folder_name)

directory1A = st.selectbox('Select Directory A', listofdirectoriesA)

response2A = s3.list_objects_v2(Bucket='bluephysicsaws', Prefix=directory1A)

#list of files in directory except

listoffilesA = [file['Key'] for file in response2A.get('Contents', [])]

filenowA =  st.selectbox('Select File A', listoffilesA)

pathA = 's3://bluephysicsaws/%s' %(filenowA)

st.markdown('## Select file B:')

listofdirectoriesB = []

for common_prefix in response.get('CommonPrefixes', []):
        # Extract the folder name from the CommonPrefixes
        folder_name = common_prefix['Prefix'].rstrip('/')
        listofdirectoriesB.append(folder_name)

directory1B = st.selectbox('Select Directory B', listofdirectoriesB)

response2B = s3.list_objects_v2(Bucket='bluephysicsaws', Prefix=directory1B)

#list of files in directory except

listoffilesB = [file['Key'] for file in response2B.get('Contents', [])]

filenowB =  st.selectbox('Select File B', listoffilesB)

pathB = 's3://bluephysicsaws/%s' %(filenowB)

#Create dataframes and draw the plots in the same graph
if 'Crossline' in filenowB:
         newcolumns = ['aX', 'Y', 'Z', 'dose', 'dummy']
         PDD = False
         rows_to_skip = 89
else:
    if 'Inline' in filenowB:
        newcolumns = ['X', 'aX', 'Z', 'dose', 'dummy']
        PDD = False
        rows_to_skip = 89
    else:
        newcolumns = ['X', 'Y', 'aX', 'dose', 'dummy']
        PDD = True
        rows_to_skip = 88

dfA = pd.read_csv(pathA, skiprows=rows_to_skip, skipfooter=2, engine='python')
dfB = pd.read_csv(pathB, skiprows=rows_to_skip, skipfooter=2, engine='python')
dfA.columns = newcolumns
dfB.columns = newcolumns
dfA['file'] = 'A'
dfB['file'] = 'B'

df = pd.concat([dfA, dfB])

#move all dose to positive
df['dose'] = np.abs(df.dose)

fig0 = px.scatter(df, x='aX', y='dose', color='file', title='Gamma analysis', labels=True)

fig0.update_layout(xaxis_title='position (cm)',
                   yaxis_title = 'relative dose (%)')



#calculate gamma
reference = df.loc[df.file == 'A', ['aX', 'dose']].to_numpy()
evaluation = df.loc[df.file == 'B', ['aX', 'dose']].to_numpy()
axis_reference = reference[:, 0]
dose_reference = reference[:, 1]

axis_evaluation = evaluation[:, 0]
dose_evaluation = evaluation[:, 1]

dose_percent_threshold = st.slider('Dose Threshold (%)', 0, 3, 1)
distance_mm_threshold = st.slider('Distance Threshold (mm)', 0, 3, 1)
local_global = st.radio('Local or Global Gamma?', ('Local', 'Global'))

gamma_options = {
      'dose_percent_threshold': dose_percent_threshold,
      'distance_mm_threshold': distance_mm_threshold,
      'lower_percent_dose_cutoff': 10,
      'interp_fraction': 10, #should be 10 or more for more accuarate results
      'max_gamma': 2,
      'random_subset': None,
      'local_gamma': local_global == 'Local', #False indicates global gamma is calculated
      'ram_available': 2**29 #1/2 GB
}

gamma = pymedphys.gamma(
      axis_reference, dose_reference,
      axis_evaluation, dose_evaluation,
      **gamma_options
)

gammatoplot = gamma * 100

valid_gamma = gamma[~np.isnan(gamma)]

pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100

gammaline = go.Scatter(mode='markers', 
                    x=axis_reference, 
                    y=gammatoplot, 
                    marker=dict(color='red', size=2, opacity=0.5), 
                    showlegend=True,
                    name='gamma')

fig0.add_traces(gammaline)

st.plotly_chart(fig0)

st.write('Pass Ratio(\u03B3<=1): %.2f%%' %pass_ratio)


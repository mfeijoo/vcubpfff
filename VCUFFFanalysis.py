import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
import streamlit.components.v1 as components
from io import StringIO

def R2 (x, y):
    coeff, cov = np.polyfit(x,y,1, cov=True)
    stdcoeff = np.sqrt(np.diag(cov))
    #print ("std of coeff = ", stdcoeff)
    polinomio = np.poly1d(coeff)
    yhat = polinomio(x)
    ybar = np.sum(y)/len(y)
    sstot = np.sum((y - ybar)**2)
    ssres = np.sum((y - yhat)**2)
    R2 = 1 - ssres/sstot
    xline = np.linspace(x.min(), x.max())
    slopeline = go.Scatter(mode='lines',
                           x=xline,
                           y=polinomio(xline),
                           line_color='red',
                           line_dash='dash',
                           showlegend=False)
    #stotext = 'y = %.4f x + %.4f \n$R^2 = %.4f$' %(coeff[0], coeff[1], R2)
    return slopeline, coeff[0]

st.title("VCU & Blue Physics FFF profiles analysis")

listofdirectories = glob('*/')

directory1 = st.selectbox('Select Directory', listofdirectories)

#list of files in directory except

listoffiles = list(set(glob('%s/*.csv' %directory1)) - set(glob('%s/*PDD*.csv' %directory1)))

filenow =  st.selectbox('Select File', listoffiles)

uploaded_file = st.file_uploader('...Or upload a .csv file with 89 lines in header, Lap format', type=['csv'])

if uploaded_file is not None:
    if 'Crossline' in uploaded_file.name:
        newcolumns = ['aX', 'Y', 'Z', 'dose', 'dummy']
    else:
        newcolumns = ['X', 'aX', 'Z', 'dose', 'dummy']

    df = pd.read_csv(uploaded_file, skiprows=89, skipfooter=2, engine='python')
else:
    if 'Crossline' in filenow:
        newcolumns = ['aX', 'Y', 'Z', 'dose', 'dummy']
    else:
        newcolumns = ['X', 'aX', 'Z', 'dose', 'dummy']

    df = pd.read_csv(filenow, skiprows=89, skipfooter=2, engine='python')

df.columns = newcolumns



#increase resolution interpolating 100 new points between raw data

dfhd = pd.DataFrame({'aX':np.linspace(df.aX.min(), df.aX.max(), df.shape[0]*100)})
dfn = dfhd.merge(df, how='outer', sort=True)
dfn['ndose'] = dfn.dose.interpolate()

hdline = go.Scatter(mode='markers', 
                    x=dfn.aX, 
                    y=dfn.ndose, 
                    marker=dict(color='cyan', size=2, opacity=0.5), 
                    showlegend=False,
                    name='interpolation')

#calculate derivative to find inflection points from raw data

df['deltax'] = df.aX.diff()
df['deltadose'] = df.dose.diff()
df['derivative'] = df.deltadose / df.deltax

#calculated derivative after 10 points

df['deltax10'] = df.aX.diff(10)
df['deltadose10'] = df.dose.diff(10)
df['derivative10'] = df.deltadose10/df.deltax10

#find inflection points

inflection1criteria =  st.slider('inflection1 criteria', 5, 60, 15)

inflection1 = df.loc[df.derivative > inflection1criteria, 'aX'].max()
inflection2 = df.loc[df.derivative < -inflection1criteria, 'aX'].min()

#find second inflection points

inflection2criteria = st.slider('inflection2 criteria', 1, 5, 2)

inflection3 = df.loc[df.derivative10 > inflection2criteria, 'aX'].max()
inflection4 = df.loc[df.derivative10 < -inflection2criteria, 'aX'].min()

#Find Pick Value
peakvalue = df.loc[(df.aX > inflection3)&(df.aX < inflection4), 'dose'].sum()
totalvalue =  df.loc[:, 'dose'].sum()
percentagepeak = peakvalue / totalvalue * 100

#Find field size before normalization
edgen = dfn.loc[dfn.ndose>50, 'aX'].min()
edgep = dfn.loc[dfn.ndose>50, 'aX'].max()
fieldsize1 =  -edgen + edgep

xcenter = 0

#Draw thd 20% lines using the hd plot

l20n = dfn.loc[(dfn.ndose > 20), 'aX'].min()
l20p = dfn.loc[(dfn.ndose > 20), 'aX'].max()

#find areas positive and negative
areandf = dfn.loc[(dfn.aX > l20n) & (dfn.aX <= 0), ['aX', 'ndose']]
areapdf = dfn.loc[(dfn.aX > 0)&(dfn.aX < l20p),['aX', 'ndose']]
area1 = areandf.ndose.sum()
area2 = areapdf.ndose.sum()
totalarea = area1 + area2
percentarea1 = area1 / totalarea * 100
percentarea2 = area2 / totalarea * 100


#Calculate slopes
slopendf = df.loc[(df.aX > inflection1 * 0.95)&(df.aX < inflection1 * 0.2), ['aX', 'dose']]
goslope1, m1 = R2(slopendf.aX, slopendf.dose)
slopepdf = df.loc[(df.aX > inflection2 * 0.2)&(df.aX < inflection2 * 0.95), ['aX', 'dose']]
goslope2, m2 = R2(slopepdf.aX, slopepdf.dose)

#Normalized to inflection points hight
inflectionh = df.loc[(df.aX == inflection1)|(df.aX == inflection2), 'dose'].mean()
df['nidose'] = df.dose / inflectionh * 100
dfn['nidose'] = dfn.ndose / inflectionh * 100

#calculate new 20% and new 80%
lni20n = dfn.loc[(dfn.nidose > 20), 'aX'].min()
lni20p = dfn.loc[(dfn.nidose > 20), 'aX'].max()
lni80n = dfn.loc[(dfn.nidose > 80), 'aX'].min()
lni80p = dfn.loc[(dfn.nidose > 80), 'aX'].max()

#calculate penumbras
penumbrandfn = dfn.loc[(dfn.aX>lni20n) & (dfn.aX<lni80n), ['aX', 'nidose']]
penumbrandf = dfn.loc[(dfn.aX>lni20n) & (dfn.aX<lni80n), ['aX', 'ndose']]
penumbrapdfn = dfn.loc[(dfn.aX>lni80p) & (dfn.aX<lni20p), ['aX', 'nidose']]
penumbrapdf = dfn.loc[(dfn.aX>lni80p) & (dfn.aX<lni20p), ['aX', 'ndose']]

#calculate new field size
edgen2 = dfn.loc[dfn.nidose>50, 'aX'].min()
edgep2 = dfn.loc[dfn.nidose>50, 'aX'].max()
fieldsize2 = -edgen2 + edgep2

#FINAL PLOTS

#fig1
fig1 = px.scatter(df, x='aX', y='dose')
fig1.add_traces(hdline)
fig1.add_vline(x=inflection1, line_dash='dash', line_color='green')
fig1.add_vline(x=inflection2, line_dash='dash', line_color='red')
fig1.add_vline(x=0, line_color='black', line_dash='dash')
fig1.add_vline(x=l20n, line_color='orange', line_dash='dash')
fig1.add_vline(x=l20p, line_color='DarkGreen', line_dash='dash')
fig1.add_annotation(x=(xcenter + l20n)/2-2, 
                    y=50, 
                    text='Area=%.2f%%' % percentarea1, 
                    font=dict(color='blue'), 
                    showarrow=False)
fig1.add_traces(px.area(areandf, x='aX', y='ndose', markers=False).data)
fig1.add_traces(px.area(areapdf, 
                        x='aX', 
                        y='ndose', 
                        color_discrete_sequence=['LIghtGreen'],
                        markers=False).data)
fig1.add_annotation(x=(xcenter + l20p)/2-2, 
                    y=50, 
                    text='Area=%.2f%%' % percentarea2, 
                    font=dict(color='green'), 
                    showarrow=False)
fig1.add_traces(goslope1)
fig1.add_annotation(x=(-slopendf.aX.max()+slopendf.aX.min())/2-2,
                      y=slopendf.dose.max()-2,
                    text='m = %.2f' %m1,
                   font_color='blue',
                   showarrow=False)
fig1.add_traces(goslope2)
fig1.add_annotation(x=(slopepdf.aX.max()+slopepdf.aX.min())/2+2,
                      y=slopepdf.dose.max()-2,
                    text='m = %.2f' %m2,
                   font_color='green',
                   showarrow=False)
fig1.add_vline(x=inflection3, line_dash='dash', line_color='DarkGreen')
fig1.add_vline(x=inflection4, line_dash='dash', line_color='DarkRed')
fig1.add_annotation(x=0.2,
                    y=95,
                    text='peak=%.2f%%'%percentagepeak,
                   font_color='red',
                   font_size=12,
                   showarrow=False)
fig1.add_annotation(x=0,
                    y=60,
                   text='Field Size=%.2fcm' %fieldsize1,
                   font_color='BlueViolet',
                   font_size=15,
                   showarrow=False)
fig1.add_traces(go.Scatter(x=penumbrandf.aX,
                           y=penumbrandf.ndose,
                           fill='tozeroy',
                          showlegend=False))
fig1.add_traces(go.Scatter(x=penumbrapdf.aX,
                           y=penumbrapdf.ndose,
                           fill='tozeroy',
                          fillcolor='LightPink',
                          showlegend=False))
fig1.add_annotation(x=lni80n+0.1,
                    y=40,
                    text='penumbra = %.2fcm' %(lni80n - lni20n),
                   font_color='red',
                   showarrow=False)
fig1.add_annotation(x=lni80p+0.1,
                    y=40,
                    text='penumbra = %.2fcm' %(lni20p - lni80p),
                   font_color='red',
                   showarrow=False)


#fig2
fig2 = px.scatter(df, x='aX', y='nidose', title='Normalized to Inflection Points')
fig2.add_vline(x=inflection1, line_dash='dash', line_color='red')
fig2.add_vline(x=inflection2, line_dash='dash', line_color='green')
fig2.add_vline(x=0, line_dash='dash', line_color='black')
fig2.add_vline(x=lni20n, line_dash='dash', line_color='orange', opacity=0.5)
fig2.add_vline(x=lni80n, line_color='Magenta', line_dash='dash', opacity=0.5)
fig2.add_vline(x=lni20p, line_color='orange', line_dash='dash', opacity=0.5)
fig2.add_vline(x=lni80p, line_color='Magenta', line_dash='dash', opacity=0.5)
fig2.add_traces(go.Scatter(x=penumbrandfn.aX, y=penumbrandfn.nidose,
                          fill='tozeroy',
                          showlegend=False))
fig2.add_traces(go.Scatter(x=penumbrapdfn.aX, y=penumbrapdfn.nidose,
                           fill='tozeroy',
                           fillcolor='Pink',
                           showlegend=False))
fig2.add_annotation(x=lni80n+0.1,
                    y=40,
                    text='penumbra = %.2fcm' %(lni80n - lni20n),
                   font_color='red',
                   showarrow=False)
fig2.add_annotation(x=lni80p+0.1,
                    y=40,
                    text='penumbra = %.2fcm' %(lni20p - lni80p),
                   font_color='red',
                   showarrow=False)
fig2.add_annotation(x=0,
                    y=60,
                   text='Field Size=%.2fcm' %fieldsize2,
                   font_color='BlueViolet',
                   font_size=15,
                   showarrow=False)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
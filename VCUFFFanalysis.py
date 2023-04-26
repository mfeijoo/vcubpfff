import streamlit as st
import pandas as pd
#import mpld3
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import streamlit.components.v1 as components

def R2 (x, y, axn, xmax=False, r2offset=0):
    coeff, cov = np.polyfit(x,y,1, cov=True)
    stdcoeff = np.sqrt(np.diag(cov))
    #print ("std of coeff = ", stdcoeff)
    polinomio = np.poly1d(coeff)
    yhat = polinomio(x)
    ybar = np.sum(y)/len(y)
    sstot = np.sum((y - ybar)**2)
    ssres = np.sum((y - yhat)**2)
    R2 = 1 - ssres/sstot
    if xmax:
        xline = np.linspace(x.min(), xmax)
    else:
        xline = np.linspace(x.min(), x.max())
    axn.plot(xline, polinomio(xline), color='r', ls='--')
    #stotext = 'y = %.4f x + %.4f \n$R^2 = %.4f$' %(coeff[0], coeff[1], R2)
    stotext = 'm = %.3f' %(coeff[0])
    axn.text((x.min()+x.max())/2+1, (y.min()+y.max())/2, stotext, fontsize=8)
    return coeff[0]

st.title("VCU & Blue Physics FFF profiles analysis")

listofdirectories = glob('*/')

directory1 = st.selectbox('Select Directory', listofdirectories)

listoffiles = glob('%s*.csv' %directory1)

filenow =  st.selectbox('Select File', listoffiles)


#st.write(filenow)

if 'Crossline' in filenow:
    newcolumns = ['aX', 'Y', 'Z', 'dose', 'dummy']
else:
    newcolumns = ['X', 'aX', 'Z', 'dose', 'dummy']

df = pd.read_csv(filenow, skiprows=89, skipfooter=2, engine='python')

df.columns = newcolumns

#st.write(df.head())


#ax1 = df.plot.scatter(x='aX', y='dose', marker='o')

#increase resolution interpolating 100 new points between raw data

dfhd = pd.DataFrame({'aX':np.linspace(df.aX.min(), df.aX.max(), df.shape[0]*100)})
dfn = dfhd.merge(df, how='outer', sort=True)
dfn['ndose'] = dfn.dose.interpolate()
#dfn.plot.scatter(x='aX', y='ndose', marker='.', color='c', ax=ax1)

#calculate derivative to find inflection points from raw data

df['deltax'] = df.aX.diff()
df['deltadose'] = df.dose.diff()
df['derivative'] = df.deltadose / df.deltax

#calculated derivative after 10 points

df['deltax10'] = df.aX.diff(10)
df['deltadose10'] = df.dose.diff(10)
df['derivative10'] = df.deltadose10/df.deltax10

#find inflection points

inflectioncriteria =  st.slider('inflection criteria', 15, 60, 50)

inflection1 = df.loc[df.derivative > inflectioncriteria, 'aX'].max()
inflection2 = df.loc[df.derivative < -inflectioncriteria, 'aX'].min()

xcenter = 0

#Draw thd 20% lines using the hd plot

l20n = dfn.loc[(dfn.ndose > 19.9)&(dfn.ndose < 20.1) & (dfn.aX < 0), 'aX'].median()
l20p = dfn.loc[(dfn.ndose > 19.9)&(dfn.ndose < 20.1) & (dfn.aX > 0), 'aX'].median()

#Draw the final plot

fig5, ax5 = plt.subplots()
df.plot.scatter(x='aX', y='dose', marker='.', ax=ax5)
ax5.axvline(inflection1, color='r', linestyle='dashed', alpha=0.5)
ax5.axvline(inflection2, color='g', linestyle='dashed', alpha=0.5)
ax5.axvline(0, color='k', linestyle='dashed', alpha=0.5)
ax5.axvline(l20n, color='orange', linestyle='dashed', alpha=0.5)
ax5.axvline(l20p, color='darkgreen', linestyle='dashed', alpha=0.5)
ax5.grid(True)

#find areas positive and negative
areandf = df.loc[(df.aX > l20n) & (df.aX <= 0), ['aX', 'dose']]
ax5.fill_between(areandf.aX, areandf.dose, color='y', alpha=0.5)
ax5.text((xcenter + l20n)/2-2, 50, 'Area=%.2f' % areandf.dose.sum(), color='r' )
areapdf = df.loc[(df.aX > 0)&(df.aX < l20p),['aX', 'dose']]
ax5.fill_between(areapdf.aX, areapdf.dose, color='lightgreen', alpha=0.5)
ax5.text((l20p + xcenter)/2-2, 50, 'Area=%.2f' %areapdf.dose.sum(), color='g')

#Calculate slopes
slopendf = df.loc[(df.aX > inflection1 * 0.95)&(df.aX < inflection1 * 0.2), ['aX', 'dose']]
R2(slopendf.aX, slopendf.dose, axn=ax5)
slopepdf = df.loc[(df.aX > inflection2 * 0.2)&(df.aX < inflection2 * 0.95), ['aX', 'dose']]
R2(slopepdf.aX, slopepdf.dose, axn=ax5)

#Normalized to inflection points hight
inflectionh = df.loc[(df.aX == inflection1)|(df.aX == inflection2), 'dose'].mean()
df['nidose'] = df.dose / inflectionh * 100
dfn['nidose'] = dfn.ndose / inflectionh * 100

fig6, ax6 = plt.subplots()

df.plot.scatter(x='aX', y='nidose', marker='.', ax=ax6)
ax6.axvline(inflection1, color='r', linestyle='dashed', alpha=0.5)
ax6.axvline(inflection2, color='g', linestyle='dashed', alpha=0.5)
ax6.axvline(0, color='k', linestyle='dashed', alpha=0.5)
ax6.grid(True)

lni20n = dfn.loc[(dfn.nidose > 19.9) & (dfn.nidose < 20.1) & (dfn.aX < 0), 'aX'].median()
lni20p = dfn.loc[(dfn.nidose > 19.9) & (dfn.nidose < 20.1) & (dfn.aX > 0), 'aX'].median()
lni80n = dfn.loc[(dfn.nidose > 79.9) & (dfn.nidose < 80.1) & (dfn.aX < 0), 'aX'].median()
lni80p = dfn.loc[(dfn.nidose > 79.9) & (dfn.nidose < 80.1) & (dfn.aX > 0), 'aX'].median()
ax6.axvline(lni20n, color='orange', linestyle='dashed', alpha=0.5)
ax6.axvline(lni80n, color='m', linestyle='dashed', alpha=0.5)
penumbrandfn = dfn.loc[(dfn.aX>lni20n) & (dfn.aX<lni80n), ['aX', 'nidose']]
penumbrandf = dfn.loc[(dfn.aX>lni20n) & (dfn.aX<lni80n), ['aX', 'ndose']]
penumbrapdfn = dfn.loc[(dfn.aX>lni80p) & (dfn.aX<lni20p), ['aX', 'nidose']]
penumbrapdf = dfn.loc[(dfn.aX>lni80p) & (dfn.aX<lni20p), ['aX', 'ndose']]
ax6.fill_between(penumbrandfn.aX, penumbrandfn.nidose, color='y', alpha=0.5)
ax5.fill_between(penumbrandf.aX, penumbrandf.ndose, color='r', alpha=0.5)
ax6.text(lni80n+0.1, 50, 'penumbra = %.2fcm' %(lni80n - lni20n), color='r')
ax5.text(lni80n+0.1, 40, 'penumbra = %.2fcm' %(lni80n - lni20n), color='r')
ax6.axvline(lni20p, color='orange', linestyle='dashed', alpha=0.5)
ax6.axvline(lni80p, color='m', alpha=0.5, linestyle='dashed')
ax6.fill_between(penumbrapdfn.aX, penumbrapdfn.nidose, color='lightgreen', alpha=0.5)
ax5.fill_between(penumbrapdf.aX, penumbrapdf.ndose, color='r', alpha=0.5)
ax6.text(lni80p -5, 50, 'penumbra = %.2fcm' %(lni20p - lni80p) ,color='g')
ax5.text(lni80p -5, 40, 'penumbra = %.2fcm' %(lni20p - lni80p) ,color='g')

#Format the plot beautiful
#ax5.set_title(filenow)
ax5.set_xlabel('position (cm)')
ax5.set_ylabel('Relative dose (%)')


#fig5 = ax5.get_figure()
#fig6 = ax6.get_figure()

#fig_html = mpld3.fig_to_html(fig)

#components.html(fig_html, height=600)

st.pyplot(fig5)
st.pyplot(fig6)
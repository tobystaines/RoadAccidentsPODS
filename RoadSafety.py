# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:06:49 2017

@author: Toby
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import dates as dates
from matplotlib.ticker import NullFormatter
import seaborn as sns
from sklearn.preprocessing import quantile_transform, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import stats

######################## Load data ###########################################

acc16 = pd.read_csv('Acc_2016_Tidy.csv') # Table of all accidents
cas16 = pd.read_csv('Cas_2016_Tidy.csv') # Table of all casualties reported in accidents
veh16 = pd.read_csv('Veh_2016_Tidy.csv') # Table of all vehichles involved in accidents
trafficVol = pd.read_excel('tra0307.xlsx') # Govt estimates of road use
population = pd.read_excel('pop_est_mid16.xlsx',index_col=0) # Population figues by Local Authority

############## Set time format correctly #####################################

acc16.dropna(subset = ['Time'], inplace= True) # drop 2 rows
acc16['Hour'] = acc16['Time'].apply(lambda x : int(str(x).split(':')[0]))
acc16['Date'] = pd.to_datetime(acc16['Date'], format='%d/%m/%Y').dt.date
acc16['Time'] = pd.to_datetime(acc16['Time'], format='%H:%M').dt.time
acc16['Month'] = pd.DatetimeIndex(acc16['Date']).month
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
roadTypes = ['A Rural','A Urban','Minor Rural','Minor Urban','Motorway']
#%%
################## Data Updates ##############################################
#Update Road Types in accident data to match less granular types in road use data
roadMapping = [('AUrban','A Urban'),
               ('CUrban','Minor Urban'),
               ('BUrban','Minor Urban'),
               ('UnclassifiedUrban','Minor Urban'),
               ('MotorwayUrban','Motorway'),
               ('A(M)Urban','Motorway'),
               ('ARural','A Rural'),
               ('CRural','Minor Rural'),
               ('BRural','Minor Rural'),
               ('UnclassifiedRural','Minor Rural'),
               ('MotorwayRural','Motorway'),
               ('A(M)Rural','Motorway'),
               ('AUnallocated','A Urban'),
               ('CUnallocated','Minor Urban'),
               ('BUnallocated','Minor Urban'),
               ('UnclassifiedUnallocated','Minor Urban'),
               ('MotorwayUnallocated','Motorway'),
               ('A(M)Unallocated','Motorway')]

acc16['1st_Road_Class_RU'] = acc16['1st_Road_Class'] + acc16['Urban_or_Rural_Area']
acc16['Alt_Road_Class'] = ''

for t in roadMapping:
    acc16.loc[acc16['1st_Road_Class_RU'] == t[0],'Alt_Road_Class'] = t[1]

#Add casualty numbers to main table
fatalities = pd.get_dummies(cas16['Casualty_Severity'])
fatalities = fatalities[['Fatal','Serious']]
fatalities = fatalities[(fatalities['Fatal']==1)|(fatalities['Serious']==1)]
casCount = pd.concat([cas16['Accident_Index'], fatalities], axis=1)
casCount.dropna(inplace=True)
casCount = casCount.groupby('Accident_Index').sum()
acc16 = acc16.join(casCount,on='Accident_Index')
acc16['Accident_Count']=1
acc16.fillna(value=0,inplace=True)

acc16 = acc16[(acc16['Light_Conditions']!='Data missing or out of range') &
              (acc16['Weather_Conditions']!='Data missing or out of range') &
              (acc16['Weather_Conditions']!='Unknown') &
              (acc16['Road_Surface_Conditions']!='Data missing or out of range')]

#Create table of dummy variables for relevant columns
dummies = pd.get_dummies(acc16[['Light_Conditions','Weather_Conditions',
                                'Road_Surface_Conditions','Alt_Road_Class']])
dummies = pd.concat([acc16[['Local_Authority_(District)','Accident_Count',
                            'Number_of_Casualties','Serious','Fatal','Date',
                            'Day_of_Week','Hour']], dummies],axis=1)

#Reset index
dummies.reset_index(inplace=True)
dummies.drop('index',axis=1,inplace=True)

#Add row for national totals
dummySum = dummies.sum(axis=0)
dummies = dummies.append(dummySum.transpose(),ignore_index=True)
dummies.loc[len(dummies)-1,'Local_Authority_(District)'] = 'National Total'
dummies.fillna(value=0,inplace=True)

#%%
################## Prepare Initial visualisation ##############################

#Calculate distribution of accidents throughout the year
#Group accidents by date
byDate = acc16[['Date','Accident_Index']].groupby('Date').count()
byDate['Date'] = byDate.index

#Identify outliers
dailyAccMean = byDate['Accident_Index'].mean()
dailyAccSD = byDate['Accident_Index'].std()

check_high = dailyAccMean + 2*dailyAccSD
check_low = dailyAccMean - 2*dailyAccSD
byDate['isOutlier'] = (byDate['Accident_Index'] > check_high)|(byDate['Accident_Index'] < check_low)
byDate['pltColour'] = '#B9BCC0'
byDate.loc[byDate['isOutlier'] == True, 'pltColour'] = 'r'

#Calculate the distribution of accidents throughout the day
byHour = acc16[['Hour','Accident_Index']].groupby('Hour').count()
byHour['Accident_Index'] = byHour['Accident_Index'].apply(lambda x: x/366) # Transform total to per day value

#%%
############### Work with data by time #######################################

#Calculate the distribution of accidents throughout each day of the week
#for the whole year, and for June and December specifically (the lightest and 
#darkest months of the year)
byHourDay = acc16[['Hour','Day_of_Week','Accident_Count','Number_of_Casualties','Serious','Fatal']].groupby(['Hour','Day_of_Week']).sum().reset_index()
byHourDayJune = acc16.loc[acc16['Month'] == 6,['Hour','Day_of_Week','Accident_Count','Number_of_Casualties','Serious','Fatal']].groupby(['Hour','Day_of_Week']).sum().reset_index()
byHourDayDec = acc16.loc[acc16['Month'] == 12,['Hour','Day_of_Week','Accident_Count','Number_of_Casualties','Serious','Fatal']].groupby(['Hour','Day_of_Week']).sum().reset_index()

#Reshape into pivot table with same layout as trafficVol
AccbyHourDay = byHourDay.pivot(index='Hour',columns='Day_of_Week',values='Accident_Count')/366
AccbyHourDay = AccbyHourDay[days]
AccbyHourDayJune = byHourDayJune.pivot(index='Hour',columns='Day_of_Week',values='Accident_Count')/30
AccbyHourDayJune = AccbyHourDayJune[days]
AccbyHourDayDec = byHourDayDec.pivot(index='Hour',columns='Day_of_Week',values='Accident_Count')/31
AccbyHourDayDec = AccbyHourDayDec[days]
CasbyHourDay = byHourDayDec.pivot(index='Hour',columns='Day_of_Week',values='Number_of_Casualties')
CasbyHourDay = CasbyHourDay[days]
SeriousbyHourDay = byHourDay.pivot(index='Hour',columns='Day_of_Week',values='Serious')
SeriousbyHourDay = SeriousbyHourDay[days]
DeathbyHourDay = byHourDay.pivot(index='Hour',columns='Day_of_Week',values='Fatal')
DeathbyHourDay = DeathbyHourDay[days]
trafficVol = trafficVol[days]
trafficVol *= 0.01
trafficVolAvg = trafficVol.sum(axis=1)/7

#Normalise daily/hourly accident rate by road traffic volume
AccbyHourDayNorm = AccbyHourDay.div(trafficVol)
AccbyHourDayDecNorm = AccbyHourDayDec.div(trafficVol)
AccbyHourDayJuneNorm = AccbyHourDayJune.div(trafficVol)
CasbyHourDayNorm = CasbyHourDay.div(trafficVol)
SeriousbyHourDayNorm = SeriousbyHourDay.div(trafficVol)
DeathbyHourDayNorm = DeathbyHourDay.div(trafficVol)
byHourNorm = pd.DataFrame()
byHourNorm['Accidents'] = AccbyHourDayNorm.sum(axis=1)/7
byHourNorm['Casualties'] = CasbyHourDayNorm.sum(axis=1)/7
byHourNorm['Serious'] = SeriousbyHourDayNorm.sum(axis=1)/7
byHourNorm['Deaths'] = DeathbyHourDayNorm.sum(axis=1)/7
byHourNormJune = AccbyHourDayJuneNorm.sum(axis=1)/7
byHourNormDec = AccbyHourDayDecNorm.sum(axis=1)/7

#Data for billions of km travelled on UK roads 2016, by road type
roadVols16 = [('A Rural',151),
              ('A Urban',80.5),
              ('Minor Rural',73.3),
              ('Minor Urban',106.9),
              ('Motorway',109.2)]

#Group accidents by the type of road they occured on
byRoadType = acc16[['Accident_Count','Alt_Road_Class', 'Number_of_Casualties','Serious','Fatal']].groupby('Alt_Road_Class').sum()
byRoadType.reset_index(inplace=True)

#Calculate statistics per distance travelled on different road types
for t in roadVols16:
    byRoadType.loc[byRoadType['Alt_Road_Class'] == t[0],'vols'] = float(t[1])

byRoadType['AccPerBillKM'] = byRoadType['Accident_Count']/byRoadType['vols']
byRoadType['CasPerBillKM'] = byRoadType['Number_of_Casualties']/byRoadType['vols']
byRoadType['SeriousPerBillKM'] = byRoadType['Serious']/byRoadType['vols']
byRoadType['FatalPerBillKM'] = byRoadType['Fatal']/byRoadType['vols']
byRoadType.sort_values('Accident_Count',inplace=True, ascending=False)

#Create table of hour against road type
RTDummies = pd.concat([acc16[['Hour','Accident_Count']],pd.get_dummies(acc16['Alt_Road_Class'])], axis=1)
byHourRoadType = RTDummies.groupby('Hour').sum().reset_index()
byHourRoadTypeNorm = byHourRoadType.copy()
for t,v in roadVols16:
    byHourRoadTypeNorm[t] /= (v/(24))

byHourRoadTypeNorm[roadTypes] = byHourRoadTypeNorm[roadTypes].divide(trafficVolAvg, axis=0)

#%%
############################ Plot results ###########################################################

sns.set_style('whitegrid')
fig, ax = plt.subplots(5,1, figsize = (12,16))
plotDates = dates.date2num(byDate['Date'])
plt.style.use('ggplot')
months = dates.MonthLocator()  # every month
monthsFmt = dates.DateFormatter('%b')
#Fit regression line for byDay data
pCoeff = np.polyfit(plotDates, byDate['Accident_Index'], 2)
regLine = np.poly1d(pCoeff)


#by day plot
ax[0].scatter(plotDates, byDate['Accident_Index'],marker= 'x', c= byDate['pltColour'])
ax[0].plot(plotDates, regLine(plotDates), 'k--')
ax[0].xaxis.set_major_locator(months)
ax[0].xaxis.set_major_formatter(monthsFmt)
ax[0].set_title('Accident Count Per Day - 2016', fontsize=16)
ax[0].set_xlabel('Date')

#by hour plot
ax[1].plot(byHour.index, byHour['Accident_Index'], c='b')
ax[1].set_title('Daily Mean Accident Count by Hour - 2016', fontsize=16)
ax[1].set_xlabel('Time of Day')

#by hour normalised plot
ax[2].plot(byHourNorm.index,byHourNorm['Accidents'],label='All of 2016')
ax[2].plot(byHourNorm.index,byHourNormJune, label='June 2016')
ax[2].plot(byHourNorm.index,byHourNormDec, label='Dec 2016')
ax[2].set_title('Accident Count by Hour, Normalised for Traffic Volume - 2016', fontsize=16)
ax[2].set_xlabel('Time of Day')
lgd2 = ax[2].legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=14)
ax[2].yaxis.set_major_formatter(NullFormatter())


#by hour and road type normalised plot
for d in days:
    ax[3].plot(AccbyHourDayNorm.index,AccbyHourDayNorm[d])
ax[3].set_title('Accident Count by Hour by Day, Normalised for Traffic Volume - 2016', fontsize=16)
ax[3].set_xlabel('Time of Day', fontsize=16)
lgd3 = ax[3].legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=14)
ax[3].yaxis.set_major_formatter(NullFormatter())

ax[4].plot(byHourRoadTypeNorm.index,byHourRoadTypeNorm[roadTypes])
ax[4].set_title('Accident Count by Hour and Road Type, Normalised for Traffic Volume - 2016', fontsize=16)
ax[4].set_xlabel('Time of Day')
lgd4 = ax[4].legend(roadTypes, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=14)
ax[4].yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()
plt.savefig('Initial Visualisations.jpg', bbox_extra_artists=(lgd2,lgd3,lgd4), bbox_inches='tight')

#%%
#Plot summary of accident counts for each value of each environmental condition
fig4,ax4 = plt.subplots(1,3,figsize=(15,5))
ax4[0].pie(acc16['Light_Conditions'].value_counts(), shadow=True)
ax4[0].set_title('Light Conditions',fontsize=18)
pieleg0 = ax4[0].legend(labels=acc16['Light_Conditions'].value_counts().index,bbox_to_anchor=(0.5,0.1), fontsize=14,loc="upper center")
ax4[1].pie(acc16['Weather_Conditions'].value_counts(), shadow=True)
ax4[1].set_title('Weather Conditions',fontsize=18)
pieleg1 = ax4[1].legend(labels=acc16['Weather_Conditions'].value_counts().index,bbox_to_anchor=(0.5,0.1), fontsize=14,loc="upper center")
ax4[2].pie(acc16['Road_Surface_Conditions'].value_counts(), shadow=True)
ax4[2].set_title('Road Surface Conditions',fontsize=18)
pieleg2 = ax4[2].legend(labels=acc16['Road_Surface_Conditions'].value_counts().index,bbox_to_anchor=(0.5,0.1), fontsize=14,loc="upper center")
plt.savefig('Condition Pies.jpg', bbox_extra_artists=(pieleg0,pieleg1,pieleg2), bbox_inches='tight')
plt.tight_layout()

#%%   Plot national number by road type

fig2, ax2 = plt.subplots(4,2, figsize = (9,18))
sns.set_style('whitegrid')
#Accidents
sns.barplot(x='Alt_Road_Class', y='Accident_Count', data=byRoadType, ax=ax2[0,0])
ax2[0,0].set_ylabel('Number of Accidents')
ax2[0,0].set_xlabel('Road Type')
ax2[0,0].set_title('Number of Accidents\nby Road Type',fontsize=14)
#Accidents per KM
sns.barplot(x='Alt_Road_Class', y='AccPerBillKM', data=byRoadType, ax=ax2[0,1])
ax2[0,1].set_ylabel('Accidents Per Billion KM Travelled')
ax2[0,1].set_xlabel('Road Type')
ax2[0,1].set_title('Accidents\nper Billion KM Travelled',fontsize=14)
#Casualties
sns.barplot(x='Alt_Road_Class', y='Number_of_Casualties', data=byRoadType, ax=ax2[1,0])
ax2[1,0].set_ylabel('Number of Casualties')
ax2[1,0].set_xlabel('Road Type')
ax2[1,0].set_title('Number of Casualties\nby Road Type',fontsize=14)
#Casualties per KM
sns.barplot(x='Alt_Road_Class', y='CasPerBillKM', data=byRoadType, ax=ax2[1,1])
ax2[1,1].set_ylabel('Casualties Per Billion KM Travelled')
ax2[1,1].set_xlabel('Road Type')
ax2[1,1].set_title('Casualties\nper Billion KM Travelled',fontsize=14)
#Serious injuries
sns.barplot(x='Alt_Road_Class', y='Serious', data=byRoadType, ax=ax2[2,0])
ax2[2,0].set_ylabel('Number of Serious Injuries')
ax2[2,0].set_xlabel('Road Type')
ax2[2,0].set_title('Number of Serious Injuries\nby Road Type',fontsize=14)
#Serious injuries per KM
sns.barplot(x='Alt_Road_Class', y='SeriousPerBillKM', data=byRoadType, ax=ax2[2,1])
ax2[2,1].set_ylabel('Serious Injuries Per Billion KM Travelled')
ax2[2,1].set_xlabel('Road Type')
ax2[2,1].set_title('Serious Injuries\nper Billion KM Travelled',fontsize=14)
#Deaths
sns.barplot(x='Alt_Road_Class', y='Fatal', data=byRoadType, ax=ax2[3,0])
ax2[3,0].set_ylabel('Number of Deaths')
ax2[3,0].set_xlabel('Road Type')
ax2[3,0].set_title('Number of Deaths\nby Road Type',fontsize=14)
#Deaths per KM
sns.barplot(x='Alt_Road_Class', y='FatalPerBillKM', data=byRoadType, ax=ax2[3,1])
ax2[3,1].set_ylabel('Deaths Per Billion KM Travelled')
ax2[3,1].set_xlabel('Road Type')
ax2[3,1].set_title('Deaths\nper Billion KM Travelled',fontsize=14)
sns.despine(fig2,top=True, right=True)
plt.tight_layout()
#plt.savefig('National Numbers by Road Type.jpg')
"""
sns.pairplot(byLAnorm[['Light_Conditions_Darkness - lighting unknown',
                       'Light_Conditions_Darkness - lights lit',
                       'Light_Conditions_Darkness - lights unlit',
                       'Light_Conditions_Darkness - no lighting',
                       'Light_Conditions_Daylight']])
plt.tight_layout()
"""
#%%
################# Group by Local Authority and Normalise ############################################

#Group by LA
byLA = dummies.groupby('Local_Authority_(District)').sum()
byLA.drop('London Airport (Heathrow)',inplace=True) # Noone lives here

#Add population estimates for local authorities
byLA = byLA.join(population,how='left')
byLA.drop('Hour',axis=1,inplace=True)

#Calculate incident rates /1000 residents
byLA['AccPerKPerson'] = (byLA['Accident_Count']/byLA['Population'])*1000
byLA['CasPerKPerson'] = (byLA['Number_of_Casualties']/byLA['Population'])*1000
byLA['SeriousPerKPerson'] = (byLA['Serious']/byLA['Population'])*1000
byLA['DeathPerKPerson'] = (byLA['Fatal']/byLA['Population'])*1000

#City of London is an outlier due to its small population and high traffic volume
#Set its stats to the mean
byLA.loc['City of London','AccPerKPerson'] = np.mean(byLA['AccPerKPerson'])
byLA.loc['City of London','CasPerKPerson'] = np.mean(byLA['CasPerKPerson'])
byLA.loc['City of London','SeriousPerKPerson'] = np.mean(byLA['SeriousPerKPerson'])
byLA.loc['City of London','DeathPerKPerson'] = np.mean(byLA['DeathPerKPerson'])

# Lists of values for the environmental conditions variables we want to investigate
lightingCols = ['Light_Conditions_Darkness - lighting unknown',
                'Light_Conditions_Darkness - lights lit',
                'Light_Conditions_Darkness - lights unlit',
                'Light_Conditions_Darkness - no lighting',
                'Light_Conditions_Daylight']
weatherCols = ['Weather_Conditions_Fine + high winds',
                'Weather_Conditions_Fine no high winds',
                'Weather_Conditions_Fog or mist',
                'Weather_Conditions_Other',
                'Weather_Conditions_Raining + high winds',
                'Weather_Conditions_Raining no high winds',
                'Weather_Conditions_Snowing + high winds',
                'Weather_Conditions_Snowing no high winds']
surfaceCols = ['Road_Surface_Conditions_Dry',
                'Road_Surface_Conditions_Flood over 3cm. deep',
                'Road_Surface_Conditions_Frost or ice',
                'Road_Surface_Conditions_Snow',
                'Road_Surface_Conditions_Wet or damp']
allCols = ['Light_Conditions_Darkness - lighting unknown',
           'Light_Conditions_Darkness - lights lit',
           'Light_Conditions_Darkness - lights unlit',
           'Light_Conditions_Darkness - no lighting',
           'Light_Conditions_Daylight',
           'Weather_Conditions_Fine + high winds',
           'Weather_Conditions_Fine no high winds',
           'Weather_Conditions_Fog or mist',
           'Weather_Conditions_Other',
           'Weather_Conditions_Raining + high winds',
           'Weather_Conditions_Raining no high winds',
           'Weather_Conditions_Snowing + high winds',
           'Weather_Conditions_Snowing no high winds',
           'Road_Surface_Conditions_Dry',
           'Road_Surface_Conditions_Flood over 3cm. deep',
           'Road_Surface_Conditions_Frost or ice',
           'Road_Surface_Conditions_Snow',
           'Road_Surface_Conditions_Wet or damp']

byLAprop = byLA[allCols].copy()
for col in byLAprop.columns.values:
    for row in byLAprop.index:
        byLAprop.loc[row,col] = byLAprop.loc[row,col]/byLA.loc[row,'Accident_Count']

byLActr = byLAprop.copy()
for col in byLActr.columns.values:
    for row in byLActr.index:
        byLActr.loc[row,col] = byLActr.loc[row,col] - byLA.loc['National Total',col]

scaler = MinMaxScaler(feature_range=(0,1),copy=True)

#Tried various methods for normalising the values
byLAnorm = pd.DataFrame(data=quantile_transform(byLAprop.copy()), index=byLAprop.index, columns=byLAprop.columns.values).copy()
#byLAnorm = scaler.fit_transform(byLAprop)
#byLAnorm = (byLAprop-byLAprop.min(axis=0))/(byLAprop.max(axis=0)-byLAprop.min(axis=0))

nationalTotals = byLA.loc['National Total']
byLA.drop('National Total',axis=0, inplace=True)
nationalTotalsprop = byLAprop.loc['National Total']
byLAprop.drop('National Total',axis=0, inplace=True)
nationalTotalsnorm = byLAnorm.loc['National Total']
byLAnorm.drop('National Total',axis=0, inplace=True)

#%%
##### Examine Correlations between conditions and accident rates ##############

#Calculate correlations between each environmental condition and each incident rate measure
accCorr = pd.DataFrame(index=allCols,columns=('PearsonCorr','PearsonPval','SpearmanCorr','SpearmanPval'))
casCorr = pd.DataFrame(index=allCols,columns=('PearsonCorr','PearsonPval','SpearmanCorr','SpearmanPval'))
seriousCorr = pd.DataFrame(index=allCols,columns=('PearsonCorr','PearsonPval','SpearmanCorr','SpearmanPval'))
deathCorr = pd.DataFrame(index=allCols,columns=('PearsonCorr','PearsonPval','SpearmanCorr','SpearmanPval'))

for c in allCols:
    for t in [(accCorr,'AccPerKPerson'),(casCorr,'CasPerKPerson'),(seriousCorr,'SeriousPerKPerson'),(deathCorr,'DeathPerKPerson')]:
        for s in [('SpearmanCorr','SpearmanPval'),('PearsonCorr','PearsonPval')]:
            t[0].loc[c,s[0]], t[0].loc[c,s[1]]= stats.pearsonr(byLAprop[c], byLA[t[1]])
        

#%%
####### See if we can find sensible clusters for conditions ###################
#Cluster local authorities by different attibute selections
cluster1 = KMeans(n_clusters=3).fit(byLAprop[allCols])
cluster2 = KMeans(n_clusters=3).fit(byLAprop[lightingCols])
cluster3 = KMeans(n_clusters=3).fit(byLAprop[weatherCols])
cluster4 = KMeans(n_clusters=3).fit(byLAprop[surfaceCols])

#Add cluster labels to byLA and byLAnorm tables
byLA['cluster1'] = cluster1.labels_
byLA['cluster2'] = cluster2.labels_
byLA['cluster3'] = cluster3.labels_
byLA['cluster4'] = cluster4.labels_

byLAprop['cluster1'] = cluster1.labels_
byLAprop['cluster2'] = cluster2.labels_
byLAprop['cluster3'] = cluster3.labels_
byLAprop['cluster4'] = cluster4.labels_

#Identify the local authority closest to the cluster centres
closest1, _ = pairwise_distances_argmin_min(cluster1.cluster_centers_, byLAprop[allCols])
closest2, _ = pairwise_distances_argmin_min(cluster2.cluster_centers_, byLAprop[lightingCols])
closest3, _ = pairwise_distances_argmin_min(cluster3.cluster_centers_, byLAprop[weatherCols])
closest4, _ = pairwise_distances_argmin_min(cluster4.cluster_centers_, byLAprop[surfaceCols])

closest1 = byLAprop.iloc[closest1]
closest2 = byLAprop.iloc[closest2]
closest3 = byLAprop.iloc[closest3]
closest4 = byLAprop.iloc[closest4]

# PLot relationships out
sns.lmplot(x='Road_Surface_Conditions_Wet or damp',
            y='AccPerKPerson',
            data=pd.concat((byLAprop['Road_Surface_Conditions_Wet or damp'],
                          byLA['AccPerKPerson']),axis=1))

fig3, ax3 = plt.subplots(18,4,figsize=(17,80))

i=0
j=0

for c in allCols:
    for v in ['AccPerKPerson','CasPerKPerson','SeriousPerKPerson','DeathPerKPerson']:
        sns.regplot(x=c,y=v,data=pd.concat((byLA[[v,'cluster1']],byLAprop[allCols]),axis=1),ax=ax3[i,j])
        #ax3[i,j].set_ylim([0,0.0002])

        if j < 3:
            j +=1
        else:
            i+=1
            j=0
plt.tight_layout()


#%% Checking for colinearity between variables

# Compute the correlation matrix
colin = byLAprop[['Weather_Conditions_Raining no high winds',
                  'Weather_Conditions_Fine no high winds',
                  'Road_Surface_Conditions_Dry',
                  'Road_Surface_Conditions_Wet or damp']]
labels = np.array(['Raining',
                   'Fine',
                   'Dry',
                   'Wet'])
corr = colin.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.set(font_scale=1.5, style='white')
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, annot=True,linewidths=.5, xticklabels=labels,yticklabels=labels,cbar_kws={"shrink": .5},annot_kws={"size":20})
ax.set_title('Colinearity Check')
plt.savefig('Colinearity Check.jpg',bbox_inches='tight')

#%%
#Linear Regression Model for Strongest Correlation

X = byLAprop['Weather_Conditions_Raining no high winds']
y = byLA['AccPerKPerson']
pCoeff = np.polyfit(X, y, 2)
polynomial = np.poly1d(pCoeff)
sns.set_style("whitegrid", {'axes.grid' : True})

# Plot Model Results
xp = np.linspace(X.min(), X.max(), 100)
fig5 = plt.figure()
axes = fig5.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
axes.scatter(X,y,c='blue')
axes.plot(xp, polynomial(xp), linewidth = 3, c='red')
axes.set_title('Weather Condition: Raining, no high winds\nAgainst Accident Rate')
axes.set_xlabel('Proportion of Accidents During Rain, no High Winds')
axes.set_ylabel('Road Accidents per 1000 People')
plt.savefig('Raining, no high winds vs Accident.jpg')

# Calculate model prediction for each LA, the residual from true value
evaluatedCurve = np.polyval(pCoeff, X)
byLA['AccPrediction'] = evaluatedCurve
byLA['Residual'] = byLA['AccPerKPerson'] - byLA['AccPrediction']

uk = gpd.read_file('Local_Authority_Districts_December_2016_Full_Clipped_Boundaries_in_Great_Britain.shp')
uk = uk.join(byLA[['AccPerKPerson','Residual']],on='lad16nm')
uk[['AccPerKPerson','Residual']].fillna(value=str(0), inplace=True)

#Plot residuals on choropleth map
sns.set_style("whitegrid", {'axes.grid' : False})
fig6 = plt.figure(figsize=(6,7))
axes6 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
uk.plot(column='Residual',cmap='OrRd', ax=axes6, legend=True)
sns.despine(fig6 ,top=True, right=True, bottom=True, left=True)
axes6.set_title('Variation from Predictive Model')
axes6.set(yticklabels=[])
axes6.set(xticklabels=[])
plt.savefig('Variation from Predictive Model 2')

#%%
# Clustering

#Cluster local authorities by different attibute selections
cluster4 = KMeans(n_clusters=3).fit(byLAprop[surfaceCols])

#Add cluster labels to byLA and byLAnorm tables
byLA['cluster4'] = cluster4.labels_
byLAprop['cluster4'] = cluster4.labels_
byLA['cluster4'] = byLA['cluster4'].astype('str')
byLAprop['cluster4'] = byLAprop['cluster4'].astype('str')
uk = gpd.read_file('Local_Authority_Districts_December_2016_Full_Clipped_Boundaries_in_Great_Britain.shp')
uk = uk.join(byLA[['AccPerKPerson','CasPerKPerson','SeriousPerKPerson',
                   'DeathPerKPerson','cluster1','cluster2','cluster3',
                   'cluster4']],on='lad16nm')
uk.drop(49, inplace=True)

uk[['AccPerKPerson','CasPerKPerson','SeriousPerKPerson','DeathPerKPerson',
    'cluster1','cluster2','cluster3','cluster4']].fillna(value=str(0), inplace=True)


#Plot Choropleth
sns.set_style("whitegrid", {'axes.grid' : False})
fig2, ax2 = plt.subplots(1,1, figsize=(8,10))
uk.plot(column='cluster4',cmap='viridis',legend=True, ax=ax2, categorical=True)
ax2.set_title('Clustering on Surface Conditions')
sns.despine(fig2 ,top=True, right=True, bottom=True, left=True)
ax2.set(yticklabels=[])
ax2.set(xticklabels=[])
plt.savefig('Clustering on Surface Conditions')

#Identify the local authority closest to the cluster centres
closest4, _ = pairwise_distances_argmin_min(cluster4.cluster_centers_, byLAprop[surfaceCols])
closest4 = byLAprop.iloc[closest4]

#Plot Parallel Coordinates PLot
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
parallel_coordinates(pd.concat((closest4[surfaceCols],closest4['cluster4']), axis=1), 'cluster4',colormap='viridis')
plt.title('Road Surface Conditions\nCluster Profiles')
plt.legend(labels=closest4.index,frameon=True,framealpha=1,fontsize=12)

for tick in axes.get_xticklabels():
    tick.set_rotation(45)
    tick.set_fontsize(12)

plt.savefig('Surface PCP',bbox_inches='tight')
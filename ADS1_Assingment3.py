# -*- coding: utf-8 -*-
"""
Created on Thu May 11 02:00:09 2023

@author: admin
"""
###############################Import python Packages #########################
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import cluster_tools as ct

############################Data Preparation ##################################
economic_indicator = pd.read_csv("API_3_DS2_en_csv_v2_5361610.csv", skiprows = 4)
economic_indicator_GDP = economic_indicator[economic_indicator['Indicator Code'].isin(['NY.GDP.MKTP.KD.ZG'])]
economic_indicator_GDP_2021 = economic_indicator_GDP.loc[:, ['Country Name','Indicator Code','2021']]
economic_indicator_export = economic_indicator[economic_indicator['Indicator Code'].isin(['NE.EXP.GNFS.ZS'])]
economic_indicator_export_2021 = economic_indicator_export.loc[:, ['Country Name','Indicator Code','2021']]
indicator_data_combined = pd.merge(economic_indicator_export_2021, economic_indicator_GDP_2021, left_on='Country Name', right_on='Country Name')
indicator_data_combined.rename(columns={'2021_x': 'Export', '2021_y': 'GDP'}, inplace=True)
del indicator_data_combined['Indicator Code_x']
del indicator_data_combined['Indicator Code_y']

indicator_data_combined_Na = indicator_data_combined.dropna()
indicator_data_combined_Na.info()
indicator_data_combined_Na.describe(include = "all")

# Drop NAN values
indicator_data_combined_Na = indicator_data_combined.dropna()

# Ploting Not normalized Data #################################################
customers_scaled_UN = indicator_data_combined_Na[['GDP','Export']]
pd.DataFrame(customers_scaled_UN, columns = ['GDP', 'Export'])

###############################################################################
from sklearn.cluster import KMeans
wcss = []

for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(customers_scaled_UN)
    wcss.append(km.inertia_)

wcss_series = pd.Series(wcss, index = range(2, 11))

plt.figure(figsize = (8, 6))
ax = sns.lineplot(y = wcss_series, x = wcss_series.index)
ax = sns.scatterplot(y = wcss_series, x = wcss_series.index, s = 150)
ax = ax.set(xlabel = 'Number of Clusters (k)',
            ylabel = 'Within Cluster Sum of Squares (WCSS)')

###############################################################################
km = KMeans(n_clusters=7, n_init = 25, random_state=1234)
km.fit(customers_scaled_UN)
centers = km.cluster_centers_
# it finds the optimal values of the centroids.
###############################################################################
simpleClusterInfo = pd.Series(km.labels_).value_counts().sort_index()
print(simpleClusterInfo) # return the unique values and sort them in ascending order
###############################################################################
centers = km.cluster_centers_
xcen = centers[:, 0]
ycen = centers[:, 1]
###############################################################################
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(data = customers_scaled_UN,
                     x = 'GDP',
                     y = 'Export',
                     hue = km.labels_,
                     palette = 'bright',
                     alpha = 0.8,
                     s = 150,
                     legend = True )

plt.scatter(xcen, ycen, 90, "k", marker="d")
###############################################################################

############## Normalize the data #############################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
customers_scaled = scaler.fit_transform(indicator_data_combined_Na[['GDP','Export']])
customers_scaled = pd.DataFrame(customers_scaled, columns = ['GDP_N', 'Export_N'])

############ Ploting Normalized Data ##########################################
from sklearn.cluster import KMeans
wcss = []

for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(customers_scaled)
    wcss.append(km.inertia_)

wcss_series = pd.Series(wcss, index = range(2, 11))

plt.figure(figsize = (8, 6))
ax = sns.lineplot(y = wcss_series, x = wcss_series.index)
ax = sns.scatterplot(y = wcss_series, x = wcss_series.index, s = 150)
ax = ax.set(xlabel = 'Number of Clusters (k)',
            ylabel = 'Within Cluster Sum of Squares (WCSS)')


# elbow point is the point after which the within cluster sum of squares doesn't decrease significantly with every iteration, so the optimal number of clusters is the number of clusters at the elbow point, i.e. 7

# Plotting clusters ###########################################################
km = KMeans(n_clusters=7, n_init = 25, random_state=1234)
km.fit(customers_scaled)
centers = km.cluster_centers_
# it finds the optimal values of the centroids.
simpleClusterInfo = pd.Series(km.labels_).value_counts().sort_index()
print(simpleClusterInfo) # return the unique values and sort them in ascending order

centers = km.cluster_centers_
xcen = centers[:, 0]
ycen = centers[:, 1]

plt.figure(figsize=(10, 8))
ax = sns.scatterplot(data = customers_scaled,
                     x = 'GDP_N',
                     y = 'Export_N',
                     hue = km.labels_,
                     palette = 'bright',
                     alpha = 0.8,
                     s = 150,
                     legend = True )

plt.scatter(xcen, ycen, 90, "k", marker="d")
plt.show()
###########Updating the dataframe with cluster number #########################

indicator_data_combined_Na['Cluster'] = km.labels_.tolist()

##################Data Preparation for Bar graph ##############################
indicator_data_combined_Na_Bar = indicator_data_combined_Na.groupby('Cluster').agg({
    'Export': 'max',
    'GDP': 'max'})
############################ Ploting Bar Graph ################################
barWidth = 0.25

# set heights of bars
bars1 = indicator_data_combined_Na_Bar['Export']
bars2 = indicator_data_combined_Na_Bar['GDP']



# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]


# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Export')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='GDP')


# Add xticks on the middle of the group bars
plt.xlabel('Clusture ------>', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))],['Clusture 0', 'Clusture 1', 'Clusture 2', 'Clusture 3', 'Clusture 4','Clusture 5','Clusture 6'],rotation=45, ha='right')

# Create legend & Show graphic
plt.legend()
plt.show()

###################Ploting pie chart for Mean Export###########################
indicator_data_combined_Na_pie = indicator_data_combined_Na.groupby('Cluster').agg({
    'Export': 'mean',
    'GDP': 'mean'}).round(2)

fig = plt.figure(figsize=(15,6))
my_labels = indicator_data_combined_Na_pie.index
explode=[0,.05,0,0,0,0,0]
ax =plt.pie(indicator_data_combined_Na_pie.Export, labels=my_labels,explode=explode,autopct='%1.1f%%')
ax = plt.title('Mean Export percentage for Clusters')
plt.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, .7))
plt.show()
############################ Ploting Pie chart for GDP growth##################
fig = plt.figure(figsize=(15,6))
my_labels = indicator_data_combined_Na_pie.index
explode=[0,0,0,0,0,0,0.05]
ax =plt.pie(abs(indicator_data_combined_Na_pie.GDP), labels=my_labels,explode=explode,autopct='%1.1f%%')
ax = plt.title('Mean GDP Value for All Clusters')
plt.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, .7))
plt.show()

###################################CURVE FIT FOR India GDP Growth #############

############################ Data Preparation for curve fit###################
economic_indicator_GDP_Ind =  economic_indicator_GDP[economic_indicator_GDP['Country Name'].isin(['India'])]
economic_indicator_GDP_Ind_T = economic_indicator_GDP_Ind.T
economic_indicator_GDP_Ind_T.columns.names = ['Year']
economic_indicator_GDP_Ind_T.drop(index=economic_indicator_GDP_Ind_T.index[:4], axis=0, inplace=True)
economic_indicator_GDP_Ind_T = economic_indicator_GDP_Ind_T.dropna()
economic_indicator_GDP_Ind_T = economic_indicator_GDP_Ind_T.reset_index()
economic_indicator_GDP_Ind_T.rename(columns = {27734:'GDP'}, inplace = True)

######################Definition of function for curve fit#####################
def Logarithmic_growth(t, a, b, c):
     return a*np.log(t) + b

# Define the time variable
t = economic_indicator_GDP_Ind_T['index'].astype(int)

# Define the dependent variable
y = abs(economic_indicator_GDP_Ind_T['GDP']).astype(int)

# Fit the model to the data
popt, pcov = curve_fit(Logarithmic_growth, t, y, maxfev = 5000)

######################################Plot for data fitting ###################
# Plot the data and the fitted model
plt.plot(t, y, 'bo', label='Data')
plt.plot(t, Logarithmic_growth(t, *popt), 'r-', label='Fit')

# Add axis labels and a legend
plt.xlabel('Year')
plt.ylabel('GDP Growth Rate (%)')
plt.legend()

# Show the plot
plt.show()

########################################Prediction#############################
# Define the time values for the predictions
t_pred = np.array([2025, 2030, 2035, 2040, 2045, 2050])

# Use the model to make predictions
y_pred = Logarithmic_growth(t_pred, *popt)

# Print the predicted values and their confidence intervals
print('Predicted GDP Growth Rate (%) of INDIA in 2025: {:.2f} ({:.2f}, {:.2f})'.format(y_pred[0], y_pred[0] - 1.96 * np.sqrt(np.diag(pcov))[0], y_pred[0] + 1.96 * np.sqrt(np.diag(pcov))[0]))
print('Predicted GDP Growth Rate (%) of INDIA 2030: {:.2f} ({:.2f}, {:.2f})'.format(y_pred[1], y_pred[1] - 1.96 * np.sqrt(np.diag(pcov))[1], y_pred[1] + 1.96 * np.sqrt(np.diag(pcov))[1]))
print('Predicted GDP Growth Rate (%) of INDIA 2035: {:.2f} ({:.2f}, {:.2f})'.format(y_pred[2], y_pred[2] - 1.96 * np.sqrt(np.diag(pcov))[1], y_pred[2] + 1.96 * np.sqrt(np.diag(pcov))[2]))

#####################################The End #############################################





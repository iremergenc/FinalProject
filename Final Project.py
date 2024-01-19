import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import Figs as fg      #A seperate figure plotting file is created to make the file less crowded

prodat = pd.read_csv("Projectdata.csv")
year=('2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022')
yearsh = ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22')
countries = ('USA', 'Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Italy', 'Spain', 'Sweden', 'Finland', 'Norway',
'Bulgaria', 'Poland', 'Ukraine', 'Lithuania', 'Romania', 'Slovakia', 'Slovenia', 'Croatia', 'Estonia', 'Latvia')
feature_names= ['GDPgrow', 'GDPpC', 'FB_HP', 'M_P','HDI','FM_Capex' ]
target_name=['FM_Capex']
num_countries=len(countries)
num_years=len(yearsh)
num_features=len(feature_names)
GDPgrow=np.empty((num_countries,num_years))
GDPpC=np.empty((num_countries,num_years))
FB_HP=np.empty((num_countries,num_years))
M_P=np.empty((num_countries,num_years))
FM_Capex=np.empty((num_countries,num_years))
HDI=np.empty((num_countries,num_years))
years_arr=np.empty((num_countries,num_years))
countrynum=np.empty((num_countries,num_years))
data_organized=np.empty((num_countries*num_years,num_features))
target_organized=np.empty((num_countries*num_years,1))

""" Data read from csv file has to be reorganized to be used in regression methods, thus first it is seperated by features and then clustered according to years"""

for i in range(num_countries):
 GDPgrow[i]=np.asarray(prodat.values[i,:],dtype=np.float64)
 GDPpC[i] = np.asarray(prodat.values[i + num_countries, :], dtype=np.float64)
 FB_HP[i] = np.asarray(prodat.values[i + 2*num_countries, :], dtype=np.float64)
 M_P[i] = np.asarray(prodat.values[i + 3*num_countries, :], dtype=np.float64)
 FM_Capex[i] = np.asarray(prodat.values[i + 4*num_countries, :], dtype=np.float64)
 HDI[i]=np.asarray(prodat.values[i+5*num_countries,:],dtype=np.float64)
 years_arr[i]=np.asarray(yearsh)
 countrynum[i] = [j+1 for j in range(0, num_years)]

for j in range(num_years):
 for i in range(num_countries):
  data_organized [i+j*num_countries]= [GDPgrow[i,j],GDPpC[i,j],FB_HP[i,j],M_P[i,j],HDI[i,j],FM_Capex[i,j]]
  if HDI[i,j]>=0.925:
   target_organized[i+j*num_countries]= 1
  if 0.925>HDI[i,j]>=0.875:
   target_organized[i+j*num_countries]= 2
  if 0.875 > HDI[i, j] >= 0.825:
   target_organized[i + j * num_countries] = 3
  if 0.825 > HDI[i, j] >= 0.775:
   target_organized[i + j * num_countries] = 4
  if HDI[i, j] < 0.775:
   target_organized[i + j * num_countries] = 5

""" Figures for yearly change in features """
fg.figplot(yearsh,GDPgrow,num_countries,'Year','GDP Growth(annual %)','Yearly Change of GDP Growth',countries,'plt')
fg.figplot(yearsh,GDPpC,num_countries,'Year','GDP per Capita (current K US$)','Yearly Change of GDP per Capita ',countries,'plt')
fg.figplot(yearsh,FB_HP,num_countries,'Year','fixed broadband – household penetration','Yearly Change of fixed broadband – household penetration',countries,'plt')
fg.figplot(yearsh,M_P,num_countries,'Year','Mobile - penetration total','Yearly Change of Mobile - penetration total',countries,'plt')
fg.figplot(yearsh,FM_Capex,num_countries,'Year','Fixed + Mobile Capex per Capita','Yearly Change of Fixed + Mobile Capex per Capita',countries,'plt')
fg.figplot(yearsh,HDI,num_countries,'Year','Human Development Index Trends','Yearly Change of Human Development Index Trends',countries,'plt')
""" Figures for cross properties of features """
fg.figplot(FM_Capex,HDI,num_countries,'Fixed + Mobile Capex per Capita','Human Development Index Trends',' ',countrynum,'sct')
fg.figplot(GDPpC,HDI,num_countries,'GDP per Capita (current K US$)','Human Development Index Trends',' ',countrynum,'sct')
fg.figplot(GDPgrow,HDI,num_countries,'GDP growth (annual %)','Human Development Index Trends',' ',countrynum,'sct')
fg.figplot(GDPpC,HDI,GDPgrow,'GDP per Capita (current K US$)','GDP growth (annual %)',' ',countrynum,'sct')
fg.figplot(FB_HP,HDI,num_countries,'fixed broadband – household penetration','Human Development Index Trends',' ',countrynum,'sct')
fg.figplot(FB_HP,M_P,num_countries,'fixed broadband – household penetration','Mobile - penetration total',' ',countrynum,'sct')

Data_F=pd.DataFrame(data=data_organized, columns=feature_names)
print(Data_F.head())
print(len(Data_F))
X=data_organized[0:num_years*num_countries,0:num_features]
y=data_organized[0:num_years*num_countries:,num_features-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
### all feauture plotting
pd.plotting.scatter_matrix(Data_F,c=target_organized, grid=True, figsize=[10,10], s=420, marker='.')
plt.show()

# # # OLS section
reg = linear_model.LinearRegression()  # linear regression fitting
reg.fit(X_train, y_train)
preddata=np.empty((2,5))
targdata=np.empty((2,1))
preddata=[X[42,:],X[63,:]]
targdata=[y[42],y[63]]
reg_pred = reg.predict(preddata)
print("reg_pred:{}".format(reg_pred))
print("target:{}".format(targdata))
reg_score=reg.score(X_test, y_test)
# cv_results = cross_val_score(reg, X, y, cv=10)
# print("cv_results:{}".format(cv_results))
print("Regression Score:{}".format(reg_score))
print("Regression Coefficients:{}".format(reg.coef_))
rmse_reg= np.sqrt(mean_squared_error(reg_pred, targdata))
print("Regression Root Mean Squared Error: {}".format(rmse_reg))
# # #Ridge Regression Section
ridge = linear_model.Ridge(alpha=0.1) #ridge regression fitting with penalty 0.1
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_score=ridge.score(X_test, y_test)
print("Ridge Score:{}".format(ridge_score))
print("Ridge Coefficients:{}".format(ridge.coef_))
rmse_ridge = np.sqrt(mean_squared_error(ridge_pred, y_test))
print("Ridge Root Mean Squared Error: {}".format(rmse_ridge))

# # # Although regression depends on 5 variables, plots are created for single dimension for Fixed broadband-household penetration vs Human Dev. Ind.
plt.scatter(FB_HP, HDI, c=countrynum,marker='o')
XX=data_organized[:,2].reshape(-1,1)
yy=data_organized[:,4]
prediction_space = np.linspace(XX.min(),XX.max()).reshape(-1,1)
reg.fit(XX, yy)
plt.plot(prediction_space, reg.predict(prediction_space), color='red', linewidth=4,label='OLS')
ridge.fit(XX, yy)
plt.plot(prediction_space, ridge.predict(prediction_space), color='blue', linewidth=3,label='Ridge')
plt.grid(which='both',axis='both',linestyle='-.',linewidth=1)
plt.legend(loc='lower left', fontsize='xx-small')
plt.xlabel('fixed broadband – household penetration')
plt.ylabel('Human Development Index Trends')
plt.show()
# # # Although regression depends on 5 variables, plots are created for single dimension for Fixed + Mobile Capex per Capita vs Fixed broadband-household penetration
plt.scatter(FM_Capex, FB_HP, c=countrynum,marker='o')
XX=data_organized[:,5].reshape(-1,1)
yy=data_organized[:,2]
prediction_space = np.linspace(XX.min(),XX.max()).reshape(-1,1)
reg.fit(XX, yy)
plt.plot(prediction_space, reg.predict(prediction_space), color='red', linewidth=4,label='OLS')
ridge.fit(XX, yy)
plt.plot(prediction_space, ridge.predict(prediction_space), color='blue', linewidth=3,label='Ridge')
plt.grid(which='both',axis='both',linestyle='-.',linewidth=1)
plt.legend(loc='lower left', fontsize='xx-small')
plt.xlabel('Fixed + Mobile Capex per Capita')
plt.ylabel('Fixed Broadband – Household Penetration')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# regr = MLPRegressor(random_state=42, max_iter=15000,hidden_layer_sizes=30,activation='tanh').fit(X_train,y_train)
# print("test in : {}".format(targdata))
# print("output: {}".format(regr.predict(preddata)))
# regscore=regr.score(X_test, y_test)
# print("regscore: {}".format(regscore))

# Kernel Ridge SVR section
from sklearn.kernel_ridge import KernelRidge
print("test in : {}".format(targdata))
krr=KernelRidge(alpha=1, kernel='poly',gamma=1,degree=2)
krr.fit(X_train,y_train)
print("output: {}".format(krr.predict(preddata)))
krr_score=krr.score(X_test, y_test)
print("KRR Score:{}".format(krr_score))

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Financial_Data = pd.read_csv('Financial-Data.csv')

print(Financial_Data)

Financial_Data.describe()

#B) Compute the mean, median, min, max, and standard deviation for each of the quantitative variables. 
#Age - Age of Applicant
print('Mean: ', Financial_Data.age.mean())
print('Median: ', Financial_Data.age.median())
print('Min: ', Financial_Data.age.min())
print('Max: ', Financial_Data.age.max())
print('Std. dev.: ', Financial_Data.age.std())

#Income - Income of Applicant
print('Mean: ', Financial_Data.income.mean())
print('Median: ', Financial_Data.income.median())
print('Min: ', Financial_Data.income.min())
print('Max: ', Financial_Data.income.max())
print('Std. dev.: ', Financial_Data.income.std())

#Months_Employed - How many months at current job
print('Mean: ', Financial_Data.months_employed.mean())
print('Median: ', Financial_Data.months_employed.median())
print('Min: ', Financial_Data.months_employed.min())
print('Max: ', Financial_Data.months_employed.max())
print('Std. dev.: ', Financial_Data.months_employed.std())

#Years_employed - How many years at current job
print('Mean: ', Financial_Data.years_employed.mean())
print('Median: ', Financial_Data.years_employed.median())
print('Min: ', Financial_Data.years_employed.min())
print('Max: ', Financial_Data.years_employed.max())
print('Std. dev.: ', Financial_Data.years_employed.std())

#Current_Address_Year - How many years at current address
print('Mean: ', Financial_Data.current_address_year.mean())
print('Median: ', Financial_Data.current_address_year.median())
print('Min: ', Financial_Data.current_address_year.min())
print('Max: ', Financial_Data.current_address_year.max())
print('Std. dev.: ', Financial_Data.current_address_year.std())

#Personal_account_m - How many months applicant had a personal account
print('Mean: ', Financial_Data.personal_account_m.mean())
print('Median: ', Financial_Data.personal_account_m.median())
print('Min: ', Financial_Data.personal_account_m.min())
print('Max: ', Financial_Data.personal_account_m.max())
print('Std. dev.: ', Financial_Data.personal_account_m.std())

#Personal_account_y - How many years applicant had a personal account
print('Mean: ', Financial_Data.personal_account_y.mean())
print('Median: ', Financial_Data.personal_account_y.median())
print('Min: ', Financial_Data.personal_account_y.min())
print('Max: ', Financial_Data.personal_account_y.max())
print('Std. dev.: ', Financial_Data.personal_account_y.std())

#Amount_Requested - How much applicant applied for
print('Mean: ', Financial_Data.amount_requested.mean())
print('Median: ', Financial_Data.amount_requested.median())
print('Min: ', Financial_Data.amount_requested.min())
print('Max: ', Financial_Data.amount_requested.max())
print('Std. dev.: ', Financial_Data.amount_requested.std())

#Risk_score - Applicants risk score
print('Mean: ', Financial_Data.risk_score.mean())
print('Median: ', Financial_Data.risk_score.median())
print('Min: ', Financial_Data.risk_score.min())
print('Max: ', Financial_Data.risk_score.max())
print('Std. dev.: ', Financial_Data.risk_score.std())

#Risk_score_2
print('Mean: ', Financial_Data.risk_score_2.mean())
print('Median: ', Financial_Data.risk_score_2.median())
print('Min: ', Financial_Data.risk_score_2.min())
print('Max: ', Financial_Data.risk_score_2.max())
print('Std. dev.: ', Financial_Data.risk_score_2.std())

#Risk_score_3
print('Mean: ', Financial_Data.risk_score_3.mean())
print('Median: ', Financial_Data.risk_score_3.median())
print('Min: ', Financial_Data.risk_score_3.min())
print('Max: ', Financial_Data.risk_score_3.max())
print('Std. dev.: ', Financial_Data.risk_score_3.std())

#Risk_score_4
print('Mean: ', Financial_Data.risk_score_4.mean())
print('Median: ', Financial_Data.risk_score_4.median())
print('Min: ', Financial_Data.risk_score_4.min())
print('Max: ', Financial_Data.risk_score_4.max())
print('Std. dev.: ', Financial_Data.risk_score_4.std())

#Risk_score_5
print('Mean: ', Financial_Data.risk_score_5.mean())
print('Median: ', Financial_Data.risk_score_5.median())
print('Min: ', Financial_Data.risk_score_5.min())
print('Max: ', Financial_Data.risk_score_5.max())
print('Std. dev.: ', Financial_Data.risk_score_5.std())

#ext_quality_score - Additional applicant quality score
print('Mean: ', Financial_Data.ext_quality_score.mean())
print('Median: ', Financial_Data.ext_quality_score.median())
print('Min: ', Financial_Data.ext_quality_score.min())
print('Max: ', Financial_Data.ext_quality_score.max())
print('Std. dev.: ', Financial_Data.ext_quality_score.std())

#ext_quality_score_2
print('Mean: ', Financial_Data.ext_quality_score_2.mean())
print('Median: ', Financial_Data.ext_quality_score_2.median())
print('Min: ', Financial_Data.ext_quality_score_2.min())
print('Max: ', Financial_Data.ext_quality_score_2.max())
print('Std. dev.: ', Financial_Data.ext_quality_score_2.std())

#Inquiries_last_month - How many inquiries the applicant made in the previous month
print('Mean: ', Financial_Data.inquiries_last_month.mean())
print('Median: ', Financial_Data.inquiries_last_month.median())
print('Min: ', Financial_Data.inquiries_last_month.min())
print('Max: ', Financial_Data.inquiries_last_month.max())
print('Std. dev.: ', Financial_Data.inquiries_last_month.std())

#Histogram variable Age
fig, ax = plt.subplots()
ax.hist(Financial_Data.age)
ax.set_axisbelow(True) 
plt.title('Age')

#Histogram variable Income
fig, ax = plt.subplots()
ax.hist(Financial_Data.income)
ax.set_axisbelow(True) 
plt.title('Income')

#Histogram variable amount_requested
fig, ax = plt.subplots()
ax.hist(Financial_Data.amount_requested)
ax.set_axisbelow(True) 
plt.title('Amount Requested')

#Histogram variable months_employed
fig, ax = plt.subplots()
ax.hist(Financial_Data.months_employed)
ax.set_axisbelow(True) 
plt.title('Months Employed')

#Histogram variable years_employed
fig, ax = plt.subplots()
ax.hist(Financial_Data.years_employed)
ax.set_axisbelow(True) 
plt.title('Years Employed')

#Histogram variable current_address_year
fig, ax = plt.subplots()
ax.hist(Financial_Data.current_address_year)
ax.set_axisbelow(True) 
plt.title('Current Address (years)')

#Histogram variable personal_account_m
fig, ax = plt.subplots()
ax.hist(financial_data_df.personal_account_m)
ax.set_axisbelow(True) 
plt.title('Personal Account(months)')

#Histogram variable personal_account_y
fig, ax = plt.subplots()
ax.hist(Financial_Data.personal_account_y)
ax.set_axisbelow(True) 
plt.title('Personal Account(years)')


#Histogram variable risk_score
fig, ax = plt.subplots()
ax.hist(Financial_Data.risk_score)
ax.set_axisbelow(True) 
plt.title('Risk Score')

fig, ax = plt.subplots()
ax.hist(Financial_Data.risk_score_2)
ax.set_axisbelow(True) 
plt.title('Risk Score 2')

fig, ax = plt.subplots()
ax.hist(Financial_Data.risk_score_3)
ax.set_axisbelow(True) 
plt.title('Risk Score 3')

fig, ax = plt.subplots()
ax.hist(Financial_Data.risk_score_4)
ax.set_axisbelow(True) 
plt.title('Risk Score 4')

fig, ax = plt.subplots()
ax.hist(Financial_Data.risk_score_5)
ax.set_axisbelow(True) 
plt.title('Risk Score 5')

fig, ax = plt.subplots()
ax.hist(Financial_Data.ext_quality_score)
ax.set_axisbelow(True) 
plt.title('External Quality Score')

fig, ax = plt.subplots()
ax.hist(Financial_Data.ext_quality_score_2)
ax.set_axisbelow(True) 
plt.title('External Quality Score 2')

#Histogram variable inquiries_last_month
fig, ax = plt.subplots()
ax.hist(Financial_Data.inquiries_last_month)
ax.set_axisbelow(True) 
plt.title('Inquiries Last Month')


Financial_Data_Amounts = Financial_Data.filter(['income','amount_requested'], axis=1)

# Since we only want to create a boxplot for two features, we can create a new dataframe that is a subset of the previous one. 
# We do this by using the filter function and specify axis=1 to ensure that we filter by columns rather than rows.

boxplot = sns.boxplot(data=Financial_Data_Amounts)

# We now have a boxplot comparing income with amount requested. Clearly, income has a bigger distribution of values from about 1000 to just under 8000. 
# Then there are several outliers above this range. Conversely, amount requested is concentrated below 2000 with a large range of outliers from about 2000 to over 10,000.

# In this portion we took all of the quantitative variables from our data in order to generate a correlation table.
Financial_Data_2 = Financial_Data[['age', 'income','months_employed', 'years_employed', 'personal_account_m',
                                               'personal_account_y', 'amount_requested', 'risk_score', 'risk_score_2',
                                               'risk_score_3', 'risk_score_4', 'risk_score_5', 'ext_quality_score',
                                               'ext_quality_score_2', 'inquiries_last_month']]

Financial_Data_3 = Financial_Data_2.corr()
Financial_Data_3

# Used to mask the upper triangle half of the heat map to reduce visual clutter
mask = np.zeros_like(Financial_Data_3)
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
mask

# Creating the heatmap and resizing the model and figures to fit the screen.
plt.figure(figsize=(25,25))
sns.heatmap(Financial_Data_3,mask=mask, annot = True, annot_kws = {"size": 21}, fmt = '.2f')
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 25)
plt.show()

Financial_Data_Numeric = Financial_Data.filter(['age', 'income','months_employed', 'years_employed', 'personal_account_m',
                                               'personal_account_y', 'amount_requested', 'risk_score', 'risk_score_2',
                                               'risk_score_3', 'risk_score_4', 'risk_score_5', 'ext_quality_score',
                                               'ext_quality_score_2', 'inquiries_last_month'], axis=1)

# Before we perform principal component analysis, we would like to isolate and standardize the numeric features. We do this first by filtering. 

column_names = ['age', 'income','months_employed', 'years_employed', 'personal_account_m',
                                               'personal_account_y', 'amount_requested', 'risk_score', 'risk_score_2',
                                               'risk_score_3', 'risk_score_4', 'risk_score_5', 'ext_quality_score',
                                               'ext_quality_score_2', 'inquiries_last_month']

# Since the method we are about to use involves transforming the dataframe into an array, we will save the column names to revert back to a dataframe later.

Financial_Data_Numeric_Stand = StandardScaler().fit_transform(Financial_Data_Numeric)
Financial_Data_Numeric_Stand = pd.DataFrame(Financial_Data_Numeric_Stand, columns = column_names)
Financial_Data_Numeric_Stand['e_signed'] = Financial_Data['e_signed']

# To normalize the dataframe we use the StandardScaler and fit_transform functions from sklearn. 
# The output of these functions is a numpy array, so to transform it back to a dataframe we use the DataFrame function from pandas, 
# specifying the column names as those names we saved in a list just prior. Finally, we would like to append the target variable e_signed from our first dataframe.

Financial_Data_Numeric_Stand.head()

# Here we want to ensure our dataframe was successfully transformed.

corr = Financial_Data_Numeric_Stand.corr()

# Now we want to create a correlation matrix and use the corr function on our dataframe.

fig, ax = plt.subplots()
fig.set_size_inches(11, 7)
sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu", center=0, ax=ax)

# Here we plot the correlation matrix and can now see how the features correlate with eachother. 
# In particular it appears that many of the risk scores are positively correlated with eachother. 
# Interestingly, it does not appear that there are strong correlations with our target variable and any other variable.

pcs = PCA(n_components=2)
pcs.fit(Financial_Data_Numeric_Stand[['risk_score_4', 'risk_score_5']])

# Given how highly correlated risk_score_4 and risk_score_5 are, we will use principal component analysis on them.

pcs_summary = pd.DataFrame({'Standard deviation' : np.sqrt(pcs.explained_variance_), 
                            'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion' : np.cumsum(pcs.explained_variance_ratio_)})

pcs_summary = pcs_summary.transpose()
pcs_summary.columns = ['PC1', 'PC2']
pcs_summary.round(4)

pcs_components_df = pd.DataFrame(pcs.components_.transpose(), columns=['PC1', 'PC2'], index=['risk_score_4', 'risk_score_5'])
pcs_components_df

# The weights for Z1 are given by (0.707107, 0.707107) and for Z2 are given by (0.707107, 0.707107). 
# Z1 accounts for 79.41% of the total variability whereas Z2 accounts for the remaining 20.59%.

scores = pd.DataFrame(pcs.transform(Financial_Data_Numeric_Stand[['risk_score_4', 'risk_score_5']]), columns=['PC1', 'PC2'])
scores.head()





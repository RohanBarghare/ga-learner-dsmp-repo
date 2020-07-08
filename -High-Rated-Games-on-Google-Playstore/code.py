# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Code starts here
data=pd.read_csv(path)
data['Rating'].plot(kind='hist')

data=data[data['Rating']<=5]
data['Rating'].plot(kind='hist')
#Code ends here


# --------------
#Code starts here

#Sum of null values of each column
total_null = data.isnull().sum()

#Percentage of null values of each column
percent_null = (total_null/data.isnull().count())

#Concatenating total_null and percent_null values
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])

print(missing_data)

#Dropping the null values
data.dropna(inplace = True)

#Sum of null values of each column
total_null_1 = data.isnull().sum()

#Percentage of null values of each column
percent_null_1 = (total_null_1/data.isnull().count())

#Concatenating total_null and percent_null values
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total', 'Percent'])

print(missing_data_1)

#Code ends here


# --------------

#Code starts here

sns.catplot(x="Category",y="Rating",data=data,kind='box',height = 10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].astype(int)
le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
sns.regplot(x="Installs", y="Rating" , data=data)
plt.title('Rating vs Installs [RegPlot]')
#Code ends here



# --------------
#Code starts here
data['Price'].value_counts()
data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype(float)
sns.regplot(x="Price", y="Rating" , data=data)
plt.title('Rating vs Price [RegPlot]')
#Code ends here


# --------------

#Code starts here
print(len(data['Genres'].unique()),'genres')
data['Genres']=data['Genres'].str.split(";").str[0] 
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

print(gr_mean.describe())

gr_mean=gr_mean.sort_values('Rating')
print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated'] ).dt.days
plt.figure(figsize = (10,10)) 
sns.regplot(x="Last Updated Days", y="Rating" , data=data)
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here



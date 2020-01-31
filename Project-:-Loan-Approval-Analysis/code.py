# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank =pd.read_csv(path)

categorical_var=bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var=bank.select_dtypes(include = 'number')
print(numerical_var)
# code ends here


# --------------
# code starts here

banks=bank.drop(['Loan_ID'],axis=1)
print(banks.isnull().sum())
#code ends here
bank_mode=banks.mode
banks=banks.fillna(value=bank_mode)



# --------------
# Code starts here

avg_loan_amount = banks.pivot_table(index = ['Gender', 'Married', 'Self_Employed'] ,values = 'LoanAmount',aggfunc=np.mean)
# code ends here



# --------------
# code starts here




loan_approved_se = ((banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')).sum()
loan_approved_nse = ((banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')).sum()
Loan_Status = banks['Loan_Status'].count()

percentage_se = loan_approved_se/Loan_Status *100
percentage_nse = loan_approved_nse/Loan_Status *100
# code ends here




# --------------
# code starts here
loan_term=banks['Loan_Amount_Term'].apply(lambda x : int(x) / 12)



big_loan_term=len(loan_term[loan_term>=25])
print(big_loan_term)

# code ends here


# --------------
# code starts here

columns_to_show = ['ApplicantIncome', 'Credit_History']
 
loan_groupby=banks.groupby(['Loan_Status'])

loan_groupby=loan_groupby[columns_to_show]

# Check the mean value 
mean_values=loan_groupby.agg([np.mean])

print(mean_values)

# code ends here



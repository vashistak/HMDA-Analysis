# -*- coding: utf-8 -*-
"""
DATA MUGGING
"""
#%%
import os
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('ggplot')


direc_=os.getcwd()
print(direc_)
#%%
"""

Reads Loan and institution csv files.
Merged loan and institution DataFrames and returns a left merged dataframe 
[by As_of_Year, Respondent_ID and Agency_Code] 

returns full data frame pandas DataFrame

"""
def read_files():
    lo_data = pd.read_csv("2012_to_2014_loans_data.csv", low_memory= False,skipinitialspace=True)    
    inst_data = pd.read_csv("2012_to_2014_institutions_data.csv", low_memory= False,skipinitialspace=True)
    return lo_data, inst_data

def hmda_init():
    loan_data,institution_data=read_files()
    result = pd.merge(loan_data,institution_data,how='left',on=['As_of_Year','Respondent_ID','Agency_Code'],sort=False)
    return result
if __name__ == '__main__':
    full_data = hmda_init()

#%% 

 """
 Buckets loan_amount_000 into quantiles of 10 groups

 """
full_data['Loan_Amount_Size']=pd.qcut((full_data.Loan_Amount_000), 10,labels=["Small","Very_Low","Low","medium","Average","Above_Average",
                                                         "High_1","High_2","High_3","Very_High"])
#full_data['Loan_Amount_Size']=pd.qcut((full_data.Loan_Amount_000), 10)
#count_loan=pd.value_counts(num_bins)
#print(count_loan)
#%%


#%%
def hmda_to_json(data,states=None, conventional_conforming=None):
    
    """
    With states and conventional conforming parameters as optional
    hmda_to_json selects data of grouped by product segment and exports into
    json files for each state selected.
    parameter states: list of states['',''] or None=select all states
    parameter conventional_conforming: Boolean True, False or None(select all without filtering)
    returns True if json files are created
     """
    
    states_data= list(data['State'].unique())
    states_list=[]
    if conventional_conforming is True:
        data_CC = data.loc[data['Conventional_Conforming_Flag'] == 'Y']
    else:
        data_CC=data
        
    if states is None:
        states_list = states_data
    else:
        if type(states) == str:
            given_list = [state.strip().upper() for state in states.split(',')]
        else:
            given_list = [state.upper() for state in states]
        
        for i in given_list:
            if i in states_data:
                states_list.append(i)
    states_list =  pd.DataFrame(states_list,columns=['States'])
            
    print(states_list)
   
    for state in states_list['States']:
        data_param = data_CC.loc[data_CC['State'] == state]
        
        filename = "loans_by_product_of_{}_{}.json".format(state,datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        output_dire_ = os.path.join(direc_, filename)
        try:
            data_json = json.loads(data_param.to_json(orient='records'))
            with open(output_dire_, 'w+', encoding='utf-8') as f:
                json.dump(data_json, f, sort_keys=True)
        except:
            raise Exception("Export to json Unsuccessful")
      
    return True

parametered_data= hmda_to_json(full_data,conventional_conforming=True)
#%%
full_data['Tract_to_MSA_MD_Income_Pct'].describe()
#%%
print(sum(full_data['Loan_Amount_000'].isnull()))
print(sum(full_data['Respondent_Name_TS'].isnull()))
print(sum(full_data['Applicant_Income_000'].isnull()))
print(sum(full_data['FFIEC_Median_Family_Income'].isnull()))
print(sum(full_data['Tract_to_MSA_MD_Income_Pct'].isnull()))
print(sum(full_data['Respondent_ZIP_Code'].isnull()))
print(sum(full_data['Conforming_Limit_000'].isnull()))
#%%
"""
After checking data types of required columns in full_data (merged data)
certain columns are having null values are replaced with '0'
and changed into numeric.
Applicant_Income_000 has no null values
FFIEC_Median_Family_Income  has no null values
Number_of_Owner_Occupied_Units has no null values
Tract_to_MSA_MD_Income_Pct 900 null values
"""
        

def convert_(df,column):
    """
    If column given is string, check length or think it is numeric and find nan rows
    
    To check for Loan_Amount_000 and Respondent_Name.
    :return:
    """
    
    if df[column].dtype in ['str']:
        df[column] = df[column].str.split(',') 
        null_values = df[df[column].str.strip().str.len() <= 3][df[column].str.upper().str.contains('NA')]
        null_values = null_values.index.values
        '''
        print(null_values)
        '''
        #identifies null values and prints the index values 
        df[column].replace(r'\s+', np.nan, regex=True)
        #replaces null values with NAN     
    else:
        #df[column] = df[column].str.strip()
        df[column].fillna(0, inplace=True)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        null_values = np.where(list(np.isnan(df[column])))[0]
        '''
        print(null_values)
        '''
    return df
full_data = convert_(full_data, 'Applicant_Income_000')
full_data = convert_(full_data, 'FFIEC_Median_Family_Income')
full_data = convert_(full_data, 'Number_of_Owner_Occupied_Units')
full_data = convert_(full_data, 'Tract_to_MSA_MD_Income_Pct')
full_data=convert_(full_data,'FFIEC_Median_Family_Income')


#%%  
def market_size_Loans(given_states=None):
    """
     Loan volume or Number of loans prossesd based on  Year and the passed parameter states
    :parameters
    :parameter given_states:list of states['',''] or None will give all states
                                          pass a 2-d input
    :return: plots
    """
    #data for conventional conforming Y
    data_Conforming = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']
    #grouping data by year and state
    market_Y_S = pd.DataFrame({'Total_Confirmed_loans': data_Conforming.groupby(['As_of_Year', 'State']).size()
                                }).reset_index()
    #converting year va.lues to datetime format to plot graph for timeline
    market_Y_S['As_of_Year'] = pd.to_datetime(market_Y_S['As_of_Year'].astype(str))
    #if list of states are given the data for the states is extrated
    #or None is passes the whole states data is passed for plots
    if given_states is None:
            states_unique =  pd.DataFrame(market_Y_S['State'].unique(),columns=['State'])
    else:
            states_unique =  pd.DataFrame(np.asarray(given_states),columns=['State'])
    
    #for each state plots are drawn for trend in lOan volume
    for state in states_unique['State']:
        
        states = market_Y_S.loc[market_Y_S['State'] == state]
        print(states)
        plt.plot(states['As_of_Year'], states['Total_Confirmed_loans'], label=state, linewidth=2,color='r')
    
        plt.xlabel('Years')
        plt.xticks(rotation=90)
        plt.ylabel('Number of Loans in each state')
        plt.title("Loan Volume")
        plt.legend(loc='best')
    # save the plot, then close it.
        plt.show()
        
        year_percent = market_Y_S['As_of_Year'].unique()
    
    for year in year_percent:
        print(year)
        market_percent = market_Y_S.loc[market_Y_S['As_of_Year'] == year]
        sum1=market_percent["Total_Confirmed_loans"].sum()
        market_percent['percent'] = market_percent["Total_Confirmed_loans"].apply(lambda c:np.round(c/sum1 *100,2))
        
        size = (np.array(market_percent.shape[::-1]) + np.array([0, 1])) * np.array([2, 0.725])
        fig, bx = plt.subplots(figsize=size)
        bx.axis('off')

        bx.table(cellText=market_percent.values, bbox=[0, 0, 1, 1], 
                colLabels=market_percent.columns,fontsize=35)
        plt.show()
market_size_Loans()
#%%
def total_money_market(given_states=None):
    """
     Toal Loan amount prossesd based on  Year and the passed parameter states
    :parameters
    :parameter given_states:list of states['',''] or None will give all states

    :return: plots
    """
    #data for conventional conforming Y
    data_Conforming=data = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']
        #grouping data by year and state

    market_Y_S = pd.DataFrame({'Total_loan_amount': data_Conforming.groupby(['As_of_Year','State',
                                    ])['Loan_Amount_000'].sum()}).reset_index()
   
  #if list of states are given the data for the states is extrated
    #or None is passes the whole states data is passed for plots
      
    if given_states is None:
            states_unique =  pd.DataFrame(market_Y_S['State'].unique(),columns=['State'])
    else:
            states_unique =  pd.DataFrame(np.asarray(given_states),columns=['State'])
    year_percent = market_Y_S['As_of_Year'].unique()
    
    #for each state plots are drawn for trend in lOan volume

    for state in states_unique['State']:
       
        market_Y_S_L = market_Y_S.loc[market_Y_S['State'] == state]
        print(market_Y_S_L)
        n=3
        move = 0.5  
        fig, ax = plt.subplots(figsize=plt.figaspect(1.0))
        # set the x axis sequence (0, 1, 2, 3...9)
        corner = np.arange(n)
      
        ax.bar(corner, market_Y_S_L['Total_loan_amount'], color='b', label=state)
        plt.xlabel('Years')
        plt.xticks(corner,market_Y_S_L['As_of_Year'])
        plt.ylabel('Total amount of loans issued in the state')
        plt.title("")
        plt.legend(loc='best')
    # save the plot, then close it.
        # save the plot, then close it.

        plt.show()
        for year in year_percent:
            market_percent = market_Y_S.loc[market_Y_S['As_of_Year'] == year]
            sum1=market_percent["Total_loan_amount"].sum()
            market_percent['percent'] = market_percent["Total_loan_amount"].apply(lambda c:np.round(c/sum1 *100,2))
        
            size = (np.array(market_percent.shape[::-1]) + np.array([0, 1])) * np.array([2, 0.725])
            fig, bx = plt.subplots(figsize=size)
            bx.axis('off')

            bx.table(cellText=market_percent.values, bbox=[0, 0, 1, 1], 
                colLabels=market_percent.columns,fontsize=35)
            plt.show()
total_money_market()

#%% 
   
def market_size_Respondent(state_given=None,top=10):
    """
     Number of Loans or loan volume prossesd based on respondents in Year and the passed parameter states
    :parameters
    :parameter state_given:list of states['',''] or None will give all states
    :top: Number of companies willing to study in plot
    :return: plots
    """
    
    data_Conforming=data = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']
    
    market_S_Y  = pd.DataFrame({'loans_Count': data_Conforming.groupby(['As_of_Year', 'Respondent_Name_TS',
                                                      'State']).size()}).reset_index()
    
#if list of states are given the data for the states is extrated
    #or None is passes the whole states data is passed for plots
      
   
    if state_given is None:
            states_unique =  pd.DataFrame(market_S_Y['State'].unique(),columns=['State'])
    else:
            states_unique =  pd.DataFrame(np.asarray(state_given),columns=['State'])
    
    
    for state in states_unique['State']:
        
        market_S_Y_state = market_S_Y.loc[market_S_Y['State'] == state]
        market_S_Y_state = market_S_Y_state.sort_values(['As_of_Year', 'loans_Count'], ascending=[False,False])
        #print(market_S_Y_state.head())
        
        #Top companies in local region of state are filtered byy most recent year 
        #for evry state and plotted for market share for every year
        
        top_companys = market_S_Y_state['Respondent_Name_TS'][:top]
        top_companys_data = market_S_Y_state[market_S_Y_state['Respondent_Name_TS'].isin(top_companys)].reset_index()
        top_companys_grouped = top_companys_data.pivot(index='Respondent_Name_TS', columns='As_of_Year',
        values='loans_Count').reset_index()
        #print(top_respondents_grouped.head())
        n = int(top)
        #data for each year is taken
        data_2012 = np.array(top_companys_grouped[2012])
        data_2013 = np.array(top_companys_grouped[2013])
        data_2014 = np.array(top_companys_grouped[2014])
        fig, ax = plt.subplots(figsize=(10,10))
        corner = np.arange(n)
        ax.bar(corner, data_2012, label='2012')
        ax.bar(corner, data_2013,bottom=data_2012,  color='b', label='2013')
        ax.bar(corner, data_2014, bottom=np.array(data_2012) + np.array(data_2013), color='r', label='2014')
        
        
        plt.legend(loc='best')
        plt.title('Number of loans in %s issued by Top companies' % state)
        fig.subplots_adjust(bottom=0.28)
        plt.xticks(corner,top_companys_grouped['Respondent_Name_TS'],rotation=90, fontsize=8)
        plt.show()

market_size_Respondent()  


#%%

def market_size_Purpose(given_states=None):
    
    """
     Number of Loans or loan volume prossesd for purpose of loan PURCHASE OR REFINANCE
     based on respondents in Year and the passed parameter states
    :parameters
    :parameter state_given:list of states['',''] or None will give all states
    :return: top 10 companies to study in plot by state

    """
   #  conventional conforming data is filtered
    data_Conforming=data = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']
    
    #loan data is filtered based on the purpose of loans and companies offering 
    #in each state. To study top
    
    market_S_Y  = pd.DataFrame({'Loan_Purpose': data_Conforming.groupby(['Respondent_Name_TS','Loan_Purpose_Description',
                                                      'State']).size()}).reset_index()
    
    #print(market_S_Y.head())
    #if list of states are given the data for the states is extrated
    #or None is passes the whole states data is passed for plots
    if given_states is None:
            states_unique =  pd.DataFrame(market_S_Y['State'].unique(),columns=['State'])
    else:
            states_unique =  pd.DataFrame(np.asarray(given_states),columns=['State'])
    
    for state in states_unique['State']:
        
        market_S_Y_state = market_S_Y.loc[market_S_Y['State'] == state]
        market_S_Y_state = market_S_Y_state.sort_values(['Loan_Purpose'], ascending=False)
       # print(market_S_Y_state.head(15))
        top_companies = market_S_Y_state['Respondent_Name_TS'][:10]
        #print(top_companies)
        top_companies_data = market_S_Y_state[market_S_Y_state['Respondent_Name_TS'].isin(top_companies)].reset_index()
        top_companies_grouped = top_companies_data.pivot(index='Respondent_Name_TS', columns='Loan_Purpose_Description',
        values='Loan_Purpose').reset_index()
        #print(top_respondents_grouped)
                                               
        n = len(top_companies_grouped.Respondent_Name_TS.value_counts())
        data_Purchase = np.array(top_companies_grouped.Purchase)
        data_Refinance = np.array(top_companies_grouped.Refinance)
        
        fig, ax = plt.subplots(figsize=plt.figaspect(1.0))
        corner = np.arange(n)
        ax.bar(corner, data_Purchase,color='r', label='Purchase')
        ax.bar(corner, data_Refinance,bottom=data_Purchase,  color='b', label='Refinance')
        
        plt.legend(loc='best')
        plt.title('Loans Types in %s each year' % state)
        fig.subplots_adjust(bottom=0.28)
        plt.xticks(corner,top_companies_grouped['Respondent_Name_TS'],rotation=90, fontsize=8)      
        plt.show()

market_size_Purpose()    


#%%
def purpose_market():
    """
     Number of Loans or loan volume prossesd for purpose of loan PURCHASE OR REFINANCE
    :return: plots
    """
    #  conventional conforming data is filtered
    data_Conforming = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']
    
    market_S_Y = pd.DataFrame({'Purpose_of_loan': data_Conforming.groupby(['As_of_Year','State',
                                    'Loan_Purpose_Description']).size()}).reset_index()
   #if list of states are given the data for the states is extrated
    #or None is passes the whole states data is passed for plots
    
    year_unique =  pd.DataFrame(market_S_Y['As_of_Year'].unique(),columns=['As_of_Year'])
    
    for year in year_unique['As_of_Year']:
        
        market_S_Y_year = market_S_Y.loc[market_S_Y['As_of_Year'] == year]
        
        market_S_Y_year = market_S_Y_year.sort_values(['As_of_Year', 'Purpose_of_loan'], ascending=[False,False])
        loan_types = market_S_Y_year['Loan_Purpose_Description']
        
        data_loan_types = market_S_Y_year[market_S_Y_year['Loan_Purpose_Description'].isin(loan_types)].reset_index()
        
        
        loan_types_grouped = data_loan_types.pivot(index='State', columns='Loan_Purpose_Description',values='Purpose_of_loan').reset_index()
        print(loan_types_grouped.head())
                                                
        n = 5
        data_Purchase = np.array(loan_types_grouped.Purchase)
        data_Refinance = np.array(loan_types_grouped.Refinance)
        
        fig, ax = plt.subplots(figsize=plt.figaspect(1.0))
        corner = np.arange(n)
        ax.bar(corner, data_Purchase,color='r', label='Purchase')
        ax.bar(corner, data_Refinance,bottom=data_Purchase,  color='b', label='Refinance')
       
        plt.xlabel('year')
        plt.legend(loc='best')
        plt.title('Total types of Loans in %s each year' % year)
        fig.subplots_adjust(bottom=0.28)
        plt.xticks(corner,loan_types_grouped['State'])
        plt.show()
purpose_market()
#%%
import seaborn as sns
from matplotlib.pyplot import cm
def market_metrics(given_col,kind='area'):
    """
     Market size by Year and the passed parameter given_col
    :param data_Confirming: confirming or non-confirmong or full data
    param given_col, kind of plot area or barh:

    """
    data_Conforming = full_data.loc[full_data['Conventional_Conforming_Flag'] == 'Y']

    loan_amount_group = pd.DataFrame({'Total_loan_amount': data_Conforming.groupby(['As_of_Year',given_col,
                                    ])['Loan_Amount_000'].sum()}).reset_index().sort_values(by=['As_of_Year', given_col])
   
    loan_volume_group=pd.DataFrame({'Total_loan_volume': data_Conforming.groupby(['As_of_Year',given_col,
                                    ])['Loan_Amount_000'].size()}).reset_index().sort_values(by=['As_of_Year', given_col])
   
    result = pd.merge(loan_amount_group,loan_volume_group,how='left',on=['As_of_Year',given_col],sort=False)
    #print(result.head())
    
    #to select top 10 number of group by items from column given or minimum
    #number of rows present in table if they are less than 10.
    
    top_filter = data_Conforming[['Loan_Amount_000', given_col]].groupby(given_col).count().reset_index()
    top_filter = top_filter.sort_values(by='Loan_Amount_000', ascending=False) \
               .iloc[:min(10, result.shape[0])]
    top_filter = list(top_filter[given_col].unique())

    # Calculating Percentage change year to year
    cols = ['Total_loan_amount', 'Total_loan_volume']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2)
    sns.set_style("whitegrid")
    cmap = cm.get_cmap('Dark2', 11)
    fig.suptitle("Loan Metrics of {}".format(given_col), fontsize=16, verticalalignment='top')

    for i in range(len(cols)):
        market_given_col = result.sort_values(by='Total_loan_volume', ascending=False)

        # Reducing the number of digits for better display of the y-ticks in plot
        market_given_col[cols[i]] /= 1000
        market_given_col = market_given_col.sort_values(by=['As_of_Year', given_col])[['As_of_Year', cols[i], given_col]] \
            .pivot('As_of_Year', given_col, cols[i])
        market_given_col.index = [str(x) for x in market_given_col.index]
        market_given_col = market_given_col[top_filter]
        #print(market_given_col)
        # Calculating the percentage of each group in Volume of Loans
        market_given_perc= market_given_col.apply(lambda c: np.round(c / c.sum() * 100, 2), axis=1)
        axes[i].set_title(cols[i])
        fig.tight_layout(pad=3)
        market_given_col.plot(kind=kind, stacked=True, legend=None, ax=axes[i], cmap=cmap)
        handles, labels = axes[0].get_legend_handles_labels()
        lg = axes[1].legend(handles, labels, bbox_to_anchor=(1.3, 1), loc=0, fontsize=14)
        for lo in lg.legendHandles:
            lo.set_linewidth(12)
        plt.show()

    fig, bx = plt.subplots(figsize=size)
    bx.axis('off')

    bx.table(cellText=market_given_perc.values, bbox=[0, 0, 1, 1], 
                colLabels=market_given_perc.columns,rowLabels=data_Conforming['As_of_Year'].unique(),fontsize=35)
    plt.show()
    
    

#%%
'''
To study loan metrics based on county name
'''
market_metrics('County_Name',kind='bar')
#%%
'''
To study loan metrics in metropolitan statistical areas
'''

market_metrics('MSA_MD_Description',kind='bar')
#%%
'''
To study loan metrics based on loan purpose
'''
market_metrics('Loan_Purpose_Description,kind='bar')

#%%
'''
To know total number of companies based on loan purpose description every year
'''
loan_volume_purpose=pd.DataFrame({'Total_loan_volume': data_Conforming.groupby(['As_of_Year','Loan_Purpose_Description'
                                    ]).size()}).reset_index()
sns.factorplot(x="As_of_Year", y="Total_loan_volume", col='Loan_Purpose_Description',
               data=loan_volume_purpose,kind="bar",saturation=.5,col_wrap=2)
#%%
import tkinter
from tkinter import *
def user_interface():
    
    def market_size_Respondent_ALL():
        market_size_Respondent()
    def market_size_Respondent_VA():
        market_size_Respondent(['VA'])
    def market_size_Respondent_MD():
        market_size_Respondent(['MD'])
   
    def purpose_market_ALL():
        purpose_market()
    
    def market_size_Loans_VA():
        market_size_Loans(['VA'])
    def market_size_Loans_MD():
        market_size_Loans(['MD'])
    def market_size_Loans_ALL():
        market_size_Loans()
    def market_metric_ALL():
        market_metrics('County_Name',kind='area')

    main_menu = tkinter.Tk()
    c=Canvas(main_menu,width=300, height=100)
    c.pack()
    
    main_m =tkinter.Menu(main_menu)
    main_menu.config(menu = main_m)
    funcMenu =tkinter.Menu(main_m)
    main_m.add_cascade(label='LOAN VOLUME',menu=funcMenu)
    funcMenu.add_command(label='Maryland', command=market_size_Loans_VA)
    funcMenu.add_command(label='Varginia', command=market_size_Loans_MD)
    funcMenu.add_separator()
    funcMenu.add_command(label='ALL States', command=market_size_Loans_ALL)
    
    main_m.add_separator()

    purposeMenu = tkinter.Menu(main_m)
   
    main_m.add_cascade(label='LOAN PURPOSE BY STATE AND YEAR',menu=purposeMenu)
    purposeMenu.add_command(label='ALL States', command=purpose_market_ALL)
    
    respondentMenu = tkinter.Menu(main_m)
   
    main_m.add_cascade(label='TOP COMPANIES GROUPED BY LOAN VOLUME IN STATE AND YEAR',menu=respondentMenu)
    respondentMenu.add_command(label='Varginia', command=market_size_Respondent_VA)
    respondentMenu.add_command(label='Maryland', command=market_size_Respondent_MD)
    respondentMenu.add_separator()
    respondentMenu.add_command(label='ALL States', command=market_size_Respondent_ALL)
    
    countyMenu = tkinter.Menu(main_m)
    main_m.add_cascade(label='CLUSTER OF LOAN VOLUME ON COUNTY NAMES',menu=countyMenu)
    countyMenu.add_command(label='Every Year', command=market_metric_ALL)
    

    tkinter.Label(main_m, text='Select from the above drop down menu', bg="red", fg="black").pack()
    main_m.mainloop()

user_interface()
#%%

'''
PART4:
    the following functions are for checking duplicate data and any oulier data while updated csv files with new rows of data

'''
#%%
def is_outlier(value, p25, p75):
    """Check if value is an outlier
    """
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return value <= lower or value >= upper
 
 
def get_indices_of_outliers(values):
    """Get outlier indices (if any)
    """
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
     
    indices_of_outliers = []
    for ind, value in enumerate(values):
        if is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)
    return indices_of_outliers
 

indices_of_outliers = get_indices_of_outliers(dist)
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dist, 'b-', label='distances')
ax.plot(
    indices_of_outliers,
    values[indices_of_outliers],
    'ro',
    markersize = 7,
    label='outliers')
ax.legend(loc='best')
def data_outlier(df,column_df,threshold=3):
    """
    Using Median Absolute Deviation to identify outliers

    :param df:
    :param column:
    :return:
    """
    
    col = column.copy()
    diff = np.abs(column-np.median(column))
    median_diff = np.median(diff)
    
    if median.diff==0:
        score=0
    else:
        score = difference / float(median_difference)
    compare = score > threshold
    col[mask] = np.median(col)
    
    return col

#%%
def outliers(tmp):
    """tmp is a list of numbers"""
    outs = []
    mean = sum(tmp)/(1.0*len(tmp))
    var = sum((tmp[i] - mean)**2 for i in range(0, len(tmp)))/(1.0*len(tmp))
    std = var**0.5
    outs = [tmp[i] for i in xrange(0, len(tmp)) if abs(tmp[i]-mean) > 1.96*std]
    return outs


lst = [random.randrange(-10, 55) for _ in range(40)]
print lst
print outliers(lst)
#%%
def update_duplicate(data, name):
    """
    Checks for duplicate records in the passed DataFrame
    :param data:
    :return:
    If loans data file is getting updated Agency_Code', 'Respondent_ID', 'As_of_Year' are the unique columns or primary keys to identify
    unique rows in data. If institutions data is getting updated 'Agency_Code', 'Respondent_ID', 'As_of_Year', 'Sequence_Number'
    are main columns.
    """
    if name==loans_data:
        imp_columns = ['Agency_Code', 'Respondent_ID', 'As_of_Year']
    else:
        imp_columns = ['Agency_Code', 'Respondent_ID', 'As_of_Year', 'Sequence_Number']

    data_unique_row = data[imp_columns].count().reset_index()

    columns = data_unique_row.columns.values
    columns[-1] = 'Row_Count'
    data_unique_row.columns = columns
    duplicate_data = data_unique_count[data_unique_row.Row_Count > 1]
    
    return duplicate_data if duplicate_data.shape[0] > 0 else None
#%%

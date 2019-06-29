''' Common includes '''
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn import ensemble,tree,linear_model,preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor
from sklearn.feature_selection import chi2,SelectKBest,RFECV

from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,log_loss,roc_curve, auc,log_loss,precision_recall_curve,average_precision_score
from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.externals import joblib
from collections import OrderedDict
import gc
import random
import seaborn as sns
import string
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats.stats import pearsonr,spearmanr
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import r2_score
from scipy.special import boxcox, inv_boxcox
import os

def describe_table(DF):
    df=DF.copy()
    df.reset_index(inplace=True)
    df_input_dtypes = pd.DataFrame(df.dtypes,columns=['Variable Type'])
    df_input_dtypes = df_input_dtypes.reset_index()
    df_input_dtypes['Variable Name'] = df_input_dtypes['index']
    df_input_dtypes = df_input_dtypes[['Variable Name','Variable Type']]
    #df_input_dtypes['Sample Value'] = df.loc[0].values
    
    unique_val=pd.DataFrame(df.nunique()).reset_index()
    unique_val.rename(columns={'index':'Variable Name',
                               0:'Unique Values'},
    inplace=True)
    
    mis_val = df.isnull().sum()
    mis_val_percent = df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    #mis_val_table[2]=map(lambda x: "{0:.2f}%".format(x * 100),mis_val_table[1])
    mis_val_table[2]=mis_val_table[1].apply(lambda x: "{0:.2f}%".format(x * 100))
    mis_val_table[1]=mis_val_table[1].apply(lambda x: round(float(x),2))
    #mis_val_table[1]=map(lambda x: round(float(x),2),mis_val_table[1])
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Number of Missing Value', 
               1 : 'Float of Missing Value',
               2 : 'Percentage of Missing Value'}
               )
    mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
    mis_val_table_ren_columns['Variable Name'] = mis_val_table_ren_columns['index']
    mis_val_table_ren_columns = mis_val_table_ren_columns[['Variable Name','Number of Missing Value', 'Float of Missing Value','Percentage of Missing Value']]
	 
    df_input_dtypes=df_input_dtypes.merge(unique_val,left_on='Variable Name', right_on='Variable Name',how='left')
    df_input_dtypes=df_input_dtypes.merge(mis_val_table_ren_columns, left_on='Variable Name', right_on='Variable Name',how='left')
    
    df_input_dtypes['Variable Type']=df_input_dtypes['Variable Type'].astype(str)
    df_input_dtypes.sort_values(['Number of Missing Value'],ascending=False,inplace=True)
    df_input_dtypes.reset_index(inplace=True,drop=True)
        
    return df_input_dtypes.drop(['Number of Missing Value','Float of Missing Value'],1)




def KS_Lift(predict_score, target,n_decile=10,new_leads=None,redecile=True):
    '''calcualte KS and top 1 decile lift'''
    data=pd.DataFrame()
    data['prescore']=predict_score
    data['target']=target.values
    #unique_length=len(np.unique(predict_score))
    if redecile:
        data=campaign_decile(new_leads,data,'decile','prescore')
        
    #elif unique_length<10:            
        #data['decile'] =unique_length-1 - pd.qcut(data['prescore'], unique_length-1, labels=False)
    else:
        data['decile'] =n_decile- pd.qcut(data['prescore'], n_decile, labels=False,duplicates='drop')

    
    temp=pd.DataFrame(index=range(1,n_decile+1))
    lead_col='Model Lead Num'
    resp_col='Model Book Num'
    
    temp[lead_col]=data.groupby(['decile'])['target'].count()
    temp[resp_col]=data.groupby(['decile'])['target'].sum()    
    temp['non_responders']=temp[lead_col]-temp[resp_col]
    total_responders=temp[resp_col].sum()
    total_nonresponders=temp['non_responders'].sum()
    temp.set_value(1,'cumulative responders',temp.loc[1,resp_col])
    temp.set_value(1,'cumulative non_responders',temp.loc[1,'non_responders'])
    for i in range(2,11):
        temp.set_value(i,'cumulative responders',temp.loc[i-1,'cumulative responders']+temp.loc[i,resp_col])
        temp.set_value(i,'cumulative non_responders',temp.loc[i-1,'cumulative non_responders']+temp.loc[i,'non_responders'])
    temp['%cum resp']=(temp['cumulative responders'].astype(float))/total_responders
    temp['%cum non_resp']=(temp['cumulative non_responders'].astype(float))/total_nonresponders
    temp['diff']=temp['%cum resp']-temp['%cum non_resp']
    temp['resp rate']=(temp[resp_col].astype(float))/temp[lead_col]
    overall_rate=float(total_responders)/temp[lead_col].sum()
    temp['lift']=temp['resp rate']/overall_rate
      
    fpr, tpr, threshold = roc_curve(data['target'].values, data['prescore'].values)
    roc_auc = auc(fpr, tpr)
        
    roc_plot={}  
    roc_plot['fpr']=fpr
    roc_plot['tpr']=tpr
    roc_plot['auc']=roc_auc    
    
    
    performance={'KS':round(temp['diff'].max(),2),
                 'ROC':round(roc_auc,2),
                 'Top 1 Decile Lift':round(temp.loc[1,'lift'],2),
                 'Top 3 Deciles Capture':round(temp.loc[3,'%cum resp'],2),  
                 'Top 5 Deciles Capture':round(temp.loc[5,'%cum resp'],2)
    }
    return temp,performance,roc_plot
    

def campaign_decile(new_leads,datasets,name,score_name):        
    temp=datasets.copy()
    temp[name]=np.NAN
    temp.sort_values([score_name],ascending=False,inplace=True)
    temp.reset_index(inplace=True)
    for decile in range(0,10):
        if decile==0:
            temp[name][temp.index<=new_leads[decile]-1]=decile
        else:
            temp[name][(temp.index>sum(new_leads[:decile])-1) & (temp.index<=sum(new_leads[:decile+1])-1)]=decile
    temp[name]=temp[name]+1
    return temp
 
      
def chk_corr(dataframe,columns, threshold=0.80):
    corr_temp=dataframe[columns].corr()
    result=pd.DataFrame(columns=['Var1','Var2','Correlation'])
    i=0
    checked=[]
    for index in corr_temp.index:
        checked.append(index)
        for column in corr_temp.columns:
            if column not in checked:
                correlation=corr_temp.loc[index,column]
                if (abs(correlation)>=threshold and correlation<1.0) or correlation==-1.0 :                    
                    result.loc[i,'Var1'] = index
                    result.loc[i,'Var2'] = column
                    result.loc[i,'Correlation'] = correlation
                    i+=1
    return result
                    

def two_overlap(df,decile1,decile2,customerid=None):   
    if customerid is not None:    
        leads=df.groupby([decile1,decile2])[customerid].nunique().to_frame().reset_index()
    else:
        leads=df.groupby([decile1,decile2]).size().to_frame().reset_index()
    leads.fillna(0,inplace=True)
    
    length=df[decile1].nunique()
    '''Index is Decile1, Column is Decile2'''
    leads_df=pd.DataFrame(index=sorted(df[decile1].unique()),columns=sorted(df[decile2].unique()))

    for rs in leads[decile1].values:
        for pp in leads[decile2].values:
            value=leads[(leads[decile1]==rs) & (leads[decile2]==pp)][0].values.astype(int)
            leads_df.set_value(rs,pp,value)
    print ('Column is %s and Row is %s'%(decile2,decile1)) 
    return leads_df

                   
def outliers_z_score(ys):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

       
def outliers_modified_z_score(value,threshold):
    ys=pd.DataFrame(list(value),columns=['value'])
    median_y = ys['value'].median()
    ys['MAD']=map(lambda x:np.abs(x - median_y), ys['value'])
    median_absolute_deviation_y=ys['MAD'].median()
    if median_absolute_deviation_y>0:
        ys['mz_score']=map(lambda x: 0.6745 * (x - median_y) / median_absolute_deviation_y,ys['value'])
        correct=ys['value'][np.abs(ys['mz_score'])<=threshold]
        correct_min=correct.min()
        correct_max=correct.max()
        num_outliers=ys.shape[0]-correct.shape[0]
    
        ys['value'][ys['mz_score']<-threshold]=correct_min
        ys['value'][ys['mz_score']> threshold]=correct_max
        return ys['value'].values, int(num_outliers)
    else:
        return value, 0
        
        
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:05:15 2018

@author: wwu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
import sys
warnings.filterwarnings("ignore")






#numeric=list(df.select_dtypes(exclude=['object']).columns)
final_feature_col=[col for col in feature_rank]
target='target'

final_feature_col

var_dict=pd.read_excel('E:\wwu\Check\Checking Prospensity IV\Full List of Attributes.xlsx')


import os, errno

try:
    os.remove(output_path)
except OSError:
    pass
'''
if os.path.exists(output_path):
    os.remove(output_path)
'''





def Decile_Avg_Mid(var_name, df, buc):
    if len(df[var_name].unique())>2:
        RR = df.groupby(pd.qcut(df[var_name], buc, duplicates='drop').astype(str))[target].mean().to_frame(name='RR')
        Min = df.groupby(pd.qcut(df[var_name], buc, duplicates='drop').astype(str))[var_name].min().to_frame(name='Min')
        Max = df.groupby(pd.qcut(df[var_name], buc, duplicates='drop').astype(str))[var_name].max().to_frame(name='Max')
    else:
        RR = df.groupby(df[var_name].astype(str))[target].mean().to_frame(name='RR')
        Min = df.groupby(df[var_name].astype(str))[var_name].min().to_frame(name='Min')
        Max = df.groupby(df[var_name].astype(str))[var_name].max().to_frame(name='Max')

    result = RR.merge(Min,left_index=True,right_index=True,how='inner')
    result= result.merge(Max,left_index=True,right_index=True,how='inner')
		
  
    return result



def univariate_rr(model_data, buc, colorc,):
    
    writer = pd.ExcelWriter(output_path)
    workbook  = writer.book
    cell_format = workbook.add_format({'align': 'left',
                                       'valign': 'vcenter',
                                       'border': 1,
                                       'bold': False,
                                       'text_wrap': True})
    percentage_format = workbook.add_format({'num_format': '0.0%'})
 
    for index_num, col in enumerate(final_feature_col):
        result	= Decile_Avg_Mid(col, model_data,  buc)
        #scoring_analysis= Decile_Avg_Mid(col, New_Scoring_data, 'Score_decile')
        #result = pd.concat([mdl_analysis, scoring_analysis], keys=[mdl_name, scr_name],axis=1)
        result.index.name = col  
        result=result.sort_values(by=['Min'])
        result.to_excel(writer,'Univariate Analysis',index=True,startrow=(buc+5)*index_num,startcol=0)
    		
        
        start=(buc+5)*index_num+2
        end=(buc+5)*index_num+result.shape[0]+1
        desc=''.join(var_dict[var_dict['Name'].str.lower()==col.lower()]['Description'])
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
                'categories': '''='Univariate Analysis'!$A$%i:$A$%i'''%(start,end),
                'values':     '''='Univariate Analysis'!$B$%i:$B$%i'''%(start,end), 
                'fill': {'color': colorc},
                         'gap':    50
                })
        #chart.add_series({'values': '''='Univariate Analysis'!$D$%i:$D$%i'''%(start,end),'name':'%s Average Score'%scr_name})
    
        chart.set_x_axis({'name': 'Group'})
        chart.set_title({'name':'%s '%desc,'name_font': {'size': 12}})
        chart.set_y_axis({
                'major_gridlines': {
                        'visible': False
                        },
                'num_format': '0.00%'
                })
        
        chart.set_legend({'none': True})
        #chart.set_legend({ 'font': {'size': 9},'position': 'top'})
        writer.sheets['Univariate Analysis'].insert_chart('I%i'%(start-1), chart)		 
    
    #formater = workbook.add_format({'border':0})
    #writer.sheets['04-Univariate Analysis'].set_column('A:DC',None,formater)
    writer.sheets['Univariate Analysis'].hide_gridlines(2)
    
    
    
    writer.save()
    writer.close()
    print ('')
    print ('Feature Univariate Analysis here! %s'%output_path)


univariate_rr(df, n_bar, '#FFB6C1')



PLL=pd.read_csv('New_to_Bank_PLL.csv')
PLL['Customer_Flag']='c'
PLL.head()
PLL.to_csv('New_to_Bank_PLL.csv',index=False)


'''
del Mdl_Dev_data,New_Scoring_data
gc.collect()
'''
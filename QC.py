
'''
This script is used for ADS model execution QC
Author: Cheng Qian
Date: 08/26/2017
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
import sys
warnings.filterwarnings("ignore")


def describe_table(df):
    if df is None:
        return None
    else:
        df_input_dtypes = pd.DataFrame(df.dtypes,columns=['Variable Type'])
        df_input_dtypes = df_input_dtypes.reset_index()
        df_input_dtypes['Variable Name'] = df_input_dtypes['index']
        df_input_dtypes = df_input_dtypes[['Variable Name','Variable Type']]

	
        mis_val = df.isnull().sum()
        mis_val_percent = df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table[2]=list(map(lambda x: "{0:.2f}%".format(x * 100),mis_val_table[1]))
        mis_val_table[1]=list(map(lambda x: round(float(x),2),mis_val_table[1]))
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Number of Missing Value', 
                   1 : 'Float of Missing Value',
                   2 : 'Percentage of Missing Value'}
                   )
        mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
        mis_val_table_ren_columns['Variable Name'] = mis_val_table_ren_columns['index']
        mis_val_table_ren_columns = mis_val_table_ren_columns[['Variable Name','Number of Missing Value', 'Float of Missing Value','Percentage of Missing Value']]
	
        df_input_dtypes=df_input_dtypes.merge(mis_val_table_ren_columns, left_on='Variable Name', right_on='Variable Name',how='left')
        df_input_dtypes['Variable Type']=df_input_dtypes['Variable Type'].astype(str)
        df_input_dtypes.sort_values(['Float of Missing Value'],ascending=False,inplace=True)
        df_input_dtypes['Number of Records']=df.shape[0]
        df_input_dtypes.drop(['Float of Missing Value'],1,inplace=True)
    return df_input_dtypes


def data_QC(df,varlist,name):
    description1=pd.DataFrame()
    for col in varlist:
        t1=df[col].describe()
        t1=pd.DataFrame({'Index':t1.index,name:t1.values})
        t1=t1[1:]
        #t1.loc[-1]={'Index':'Skew',name:df[col].skew()}
        #t1.loc[-2]={'Index':'Kurtosis',name:df[col].kurtosis()}
        t1.loc[-1]={'Index':'%Missing',name:round((1-df[col].count().astype(float)/df.shape[0]),2)}
        t1.loc[-2]={'Index':'Variable Name',name:col}
        t1=t1.sort_index()
        description1 = pd.concat([description1,t1] ,axis=0)      
    description1=pd.DataFrame(description1)
    description1.reset_index(inplace=True,drop=True)
    return description1


def Var_Check(df,varlist,name):
    if df is None:
        return True
    else:
        wrong=[]
        for col in varlist:
            if col not in df:            
                print ('Variable %s not found in %s'%(col,name))
                wrong.append(col)
    
        if len(wrong)>0:
            print ('--------')
            return False
        else:
            return True
            
        

Score_Var_to_Check=[str(x).upper() for x in Score_Var_to_check]


def col_check(x):
    if x is not None:
        return str(x).upper()
    else:
        return None


MDL_Score=col_check(MDL_Score)
Data_Score=col_check(Data_Score)
MDL_Decile=col_check(MDL_Decile)
Data_Decile=col_check(Data_Decile)

'''#####################################################  Data Path Check ######################################################'''
Check=True
print ('Checking attributes and datasets...')


def load_data(path,message,chunksize=100000):
   if not os.path.exists(path):
       print (message)
       raise ValueError
       return None, False
   else:
       data_chunk=pd.read_csv(path,chunksize=chunksize)
       data=pd.DataFrame()
       data=pd.concat([chunk for chunk in data_chunk],ignore_index=True)
       data.columns=[str(x).upper() for x in data.columns]
       return data, True


message='Error: Model development data not found'
Mdl_Dev_data,Check	   = load_data(mdl_transform_path,message,chunksize=100000)
message='Error: Scoring data not found'
New_Scoring_data,Check = load_data(score_transform_path,message,chunksize=100000)


'''####################################################  Variable Check  ########################################################'''
if Check:
    check1=Var_Check(Mdl_Dev_data,   Score_Var_to_Check,'Modeling Development Datasets')
    check2=Var_Check(New_Scoring_data,Score_Var_to_Check,'New Scoring Datasets')
    if False in [check1,check2]:
        Check=False    



score_check=True
if None in (MDL_Score,Data_Score):
    score_check=False

decile_check=True
if None in (MDL_Decile,Data_Decile):
    decile_check=False

if Check:
    
    print ('Data and Attributes read!')
    print ('')
    
    df_mdl_desc = describe_table(Mdl_Dev_data[Score_Var_to_Check])
    df_score_desc = describe_table(New_Scoring_data[Score_Var_to_Check])
    
    '''''################################################## Variable Classification ###############################################'''
    
        
    var_category={'Category':[],'Continuous':[]}

    var_category['Category']  =[col for col in list(Mdl_Dev_data.select_dtypes(include=['object']).columns) if col in Score_Var_to_Check]
    var_category['Continuous']=[col for col in list(Mdl_Dev_data.select_dtypes(exclude=['object']).columns) if col in Score_Var_to_Check]

    
    removeal=[]
    for column in var_category['Continuous']:
        unique_num=Mdl_Dev_data[column].nunique()
        if int(unique_num)<10:
            #print (column)
            var_category['Category'].append(column)      
            removeal.append(column)            
    #var_category['Continuous'].remove(column)
    continuous2=[col for col in var_category['Continuous'] if col not in removeal]
    var_category['Continuous']=continuous2
    '''''################################################## Transform&Scoring Data QC ###############################################'''

    print ('Writing descriptions of datasets...')
    continuous_check=[col for col in list(Mdl_Dev_data.select_dtypes(exclude=['object']).columns) if col in Score_Var_to_Check]
    description1=data_QC(Mdl_Dev_data,continuous_check,mdl_name)
    description2=data_QC(New_Scoring_data,continuous_check,scr_name)

    Transform_QC = description1.merge(description2,left_index=True,right_index=True, how='inner')[['Index_x',mdl_name,scr_name]]


    for index in Transform_QC.index:
        try:            
            p=abs((Transform_QC.loc[index,scr_name]-Transform_QC.loc[index,mdl_name])/(Transform_QC.loc[index,mdl_name]+0.00001))
            Transform_QC.set_value(index,'Percentage Change',round(p,2))
        except:
            Transform_QC.set_value(index,'Percentage Change',0)
    Transform_QC.rename(columns={'Index_x':'Index'},inplace=True)        
    
    

        

    if score_check:
        if decile_check:
            Mdl_Dev_data['MDL_Decile'] = Mdl_Dev_data[MDL_Decile]
            New_Scoring_data['Score_decile'] = New_Scoring_data[Data_Decile]
            decile_range=sorted(Mdl_Dev_data['MDL_Decile'].unique())
        else:  
            Mdl_Dev_data['MDL_Decile'] = 10-pd.qcut(Mdl_Dev_data[MDL_Score], 10, labels=False)
            New_Scoring_data['Score_decile'] = 10-pd.qcut(New_Scoring_data[Data_Score], 10, labels=False)
            decile_range=range(1,11)
                    
        print ('Checking model decile scores...')
        a=Mdl_Dev_data.groupby(['MDL_Decile'])[MDL_Score].min()
        b=Mdl_Dev_data.groupby(['MDL_Decile'])[MDL_Score].median()
        c=New_Scoring_data.groupby(['Score_decile'])[Data_Score].min()
        d=New_Scoring_data.groupby(['Score_decile'])[Data_Score].median()
        
        
        
        Score_QC=pd.concat([a,c,b,d] ,keys=[mdl_name+'_Min',scr_name+'_Min',mdl_name+'_Median',scr_name+'_Median'],axis=1)
        Score_QC['%Min_Change']=abs(Score_QC[mdl_name+'_Min']-Score_QC[scr_name+'_Min'])/Score_QC[mdl_name+'_Min']
        Score_QC['%Min_Change']=list(map(lambda x: "{0:.2f}%".format(x*100),Score_QC['%Min_Change']))

        Score_QC['%Median_Change']=abs(Score_QC[mdl_name+'_Median']-Score_QC[scr_name+'_Median'])/Score_QC[mdl_name+'_Median']
        Score_QC['%Median_Change']=list(map(lambda x: "{0:.2f}%".format(x*100),Score_QC['%Median_Change']))
        
        Score_QC['Decile']=decile_range
        Score_QC=Score_QC[['Decile',mdl_name+'_Min',scr_name+'_Min','%Min_Change',mdl_name+'_Median',scr_name+'_Median','%Median_Change']]

    




    '''''#####################################################PSI###############################################################'''



    print ('Checking PSI value...')

            
    columns=['Variable','Values','Actual','Expected','A-E','ln(A/E)','PSI']
    psi_result=pd.DataFrame(columns=columns)
    
    
    
    for variable in var_category['Category']:
        result=pd.DataFrame(columns=columns)
        expected=(Mdl_Dev_data[variable].value_counts(normalize=True,dropna=False)).sort_index().to_frame(name='Expected')
        actual=(New_Scoring_data[variable].value_counts(normalize=True,dropna=False)).sort_index().to_frame(name='Actual')
                
        data=actual.merge(expected,left_index=True,right_index=True,how='left')  
        

        
        result['Values']=data.index
        result['Expected']=list(map(lambda x: round(x,2),data['Expected']))
        result['Actual']=list(map(lambda x: round(x,2),actual['Actual']))
        result['A-E']=list(map(lambda x: round(x,2),result['Actual']-result['Expected']))
        result['ln(A/E)']=list(map(lambda x: round(x,2),np.log(result['Actual']/result['Expected'])))
        result['PSI']=result['A-E'] * result['ln(A/E)']
        result['Variable']=variable
        variable_psi=np.nansum(list(filter(lambda x: x not in [np.Inf,np.nan],result['PSI'])))
        result.set_value(result.index.max()+1,'ln(A/E)','Sum')
        result.set_value(result.index.max(),'PSI',variable_psi)
        result.loc[result.index.max()+1]=np.NaN
        psi_result=pd.concat([psi_result,result],axis=0)
        



    for variable in var_category['Continuous']:
        result=pd.DataFrame(columns=columns)
        quantile_values=Mdl_Dev_data[variable].quantile(q=np.linspace(0.0,1.0,11)).values
        cutoff=sorted(list(set(quantile_values)))
        
        values,expected,actual=[],[],[]
        
        for index in range(len(cutoff)-1): 
            if cutoff[index]<0.001:
                n=6
            else:
                n=2
            #minimum=round(cutoff[index],n)
            #maximum=round(cutoff[index+1],n)
            minimum=cutoff[index]
            maximum=cutoff[index+1]

            
            if index==0:
                values.append('<%s'%str(maximum))
                expected.append(float(Mdl_Dev_data[Mdl_Dev_data[variable]<maximum].shape[0]) / Mdl_Dev_data.shape[0])       
                actual.append(float(New_Scoring_data[New_Scoring_data[variable]<maximum].shape[0]) / New_Scoring_data.shape[0]) 
            elif index<=len(cutoff)-3:
                values.append('%s - %s'%(str(minimum),str(maximum)))
                expected.append(float(Mdl_Dev_data[(Mdl_Dev_data[variable]>=minimum) & (Mdl_Dev_data[variable]<maximum)].shape[0]) / Mdl_Dev_data.shape[0])       
                actual.append(float(New_Scoring_data[(New_Scoring_data[variable]>=minimum) & (New_Scoring_data[variable]<maximum)].shape[0]) / New_Scoring_data.shape[0])       
            else:
                values.append('>=%s'%str(minimum))
                expected.append(float(Mdl_Dev_data[Mdl_Dev_data[variable]>=minimum].shape[0]) / Mdl_Dev_data.shape[0])       
                actual.append(float(New_Scoring_data[New_Scoring_data[variable]>=minimum].shape[0]) / New_Scoring_data.shape[0])  
        if Mdl_Dev_data[variable].isnull().sum()>0 or New_Scoring_data[variable].isnull().sum()>0:
            values.append('Missing')
            expected.append(float(Mdl_Dev_data[variable].isnull().sum()) / Mdl_Dev_data.shape[0])
            actual.append(float(New_Scoring_data[variable].isnull().sum()) / New_Scoring_data.shape[0])
        
        result['Values']=values
        result['Expected']=list(map(lambda x: round(x,2),expected))
        result['Actual']=list(map(lambda x: round(x,2),actual))
        result['A-E']=list(map(lambda x: round(x,2),result['Actual']-result['Expected']))
        result['ln(A/E)']=list(map(lambda x: round(x,2),np.log(result['Actual']/result['Expected'])))
        result['PSI']=result['A-E'] * result['ln(A/E)']
        result['Variable']=variable
        variable_psi=np.nansum(list(filter(lambda x: x not in [np.Inf,np.nan],result['PSI'])))
        result.set_value(result.index.max()+1,'ln(A/E)','Sum')
        result.set_value(result.index.max(),'PSI',variable_psi)  
        result.loc[result.index.max()+1]=np.NaN
        psi_result=pd.concat([psi_result,result],axis=0)    
        

    psi_result['PSI']=list(map(lambda x: round(x,3),psi_result['PSI']))

    psi_result.reset_index(inplace=True,drop=True)
    
    psi_result.rename(columns={'Actual':'Actual:'+scr_name,'Expected':'Expected:'+mdl_name},inplace=True)





    '''''#####################################################Write into Excel###############################################################'''

    print ('')
    print ('Writing into Excel...')
    writer = pd.ExcelWriter(output_path)
    workbook  = writer.book
    cell_format = workbook.add_format({'align': 'left',
                                       'valign': 'vcenter',
                                       'border': 1,
                                       'bold': False,
                                       'text_wrap': True})
    percentage_format = workbook.add_format({'num_format': '0.0%'})
    
    df_mdl_desc.to_excel(writer,'01-Data Overview',index=False)
    
    df_score_desc.to_excel(writer,'01-Data Overview',index=False,startrow=df_mdl_desc.shape[0]+3)
    worksheet=writer.sheets['01-Data Overview'] 
    
    text='Overview of %s'%mdl_name                                 
    worksheet.merge_range('H2:M5', text, cell_format)
    
    
    text='Overview of %s' %scr_name                                
    worksheet.merge_range('H%i:M%i'%(df_mdl_desc.shape[0]+4,df_mdl_desc.shape[0]+7), text, cell_format)
    
    
    Transform_QC.to_excel(writer,'02-Data Attributes QC',index=False)
    text='''This Tab is to compare distribution after data transformation;\n
    Column B is {0} Value;\n
    Column C is New Scoring Datasets Value;\n
    Column D is Percentage Change,values higher than 20% will be highlighted red;'''.format(mdl_name)
    worksheet=writer.sheets['02-Data Attributes QC']                               
    worksheet.merge_range('F3:M11', text, cell_format)    
    
    
    
    max_length=Transform_QC.shape[0]+1
    font_format = workbook.add_format({'font_color': 'red','num_format': '0.0%'})
    border_format = workbook.add_format({'border':1})                               
    QC_sheet='02-Data Attributes QC'
    
    worksheet=writer.sheets[QC_sheet]
    worksheet.conditional_format('D2:D%i'%max_length, {'type':     'cell',
                                                       'criteria': '>=',
                                                       'value':    0.0,
                                                       'format':   percentage_format})
    worksheet.conditional_format('D2:D%i'%max_length, {'type':     'cell',
                                                       'criteria': 'greater than',
                                                       'value':    0.2,
                                                       'format':   font_format})
        
    for row in range(2,Transform_QC.shape[0],9):
        worksheet.conditional_format('A%i:D%i'%(row,row),{'type':     'cell',
                                                          'criteria': '>=',
                                                          'value':    0,
                                                          'format':  border_format})

    '''Score Decile plot'''
    if score_check:                               
        Score_QC=Score_QC[['Decile',mdl_name+'_Min',scr_name+'_Min','%Min_Change',mdl_name+'_Median',scr_name+'_Median','%Median_Change']]
        Score_QC.to_excel(writer,'03-Score QC Plot',index=False)
        position=len(decile_range)+1

        worksheet = writer.sheets['03-Score QC Plot']

        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({'values': '''='03-Score QC Plot'!$B$2:$B$%i'''%position,'name':'%s Score'%mdl_name})
        chart.add_series({'values': '''='03-Score QC Plot'!$C$2:$C$%i'''%position,'name':'%s Score'%scr_name})

        chart.set_x_axis({'name': 'Model Decile'})
        chart.set_title({'name':'Minimum Score by Decile Comparison','name_font': {'size': 12}})

        worksheet.insert_chart('B13', chart)



    '''PSI'''

    psi_result.to_excel(writer,'04-PSI',index=False)
    worksheet = writer.sheets['04-PSI']
    font_format = workbook.add_format({'font_color': 'red'})
    worksheet.conditional_format('G2:G%i'%(psi_result.shape[0]+1),{'type':     'cell',
                                                      'criteria': '>=',
                                                      'value':    0.1,
                                                      'format':  font_format})

    text=''' This tab is to calculate PSI(Population Stability Index) of each variable \n
PSI<0.1 : Minor Shift \n
PSI Between 0.1 and 0.25: Some Minor Change \n
PSI>0.25 : Major Shift \n

Actual: Percentage of customers by range in the %s \n
Expected: Percentage of customers by range in the %s \n
'''    %(scr_name,mdl_name)
    worksheet.merge_range('J6:P20', text, cell_format)

    
    '''Univariate Analysis'''
    def Decile_Avg_Mid(var_name, df, decile_name):
        avg = df.groupby(decile_name)[var_name].mean().to_frame(name='Average')
        mid = df.groupby(decile_name)[var_name].median().to_frame(name='Median')

        result = avg.merge(mid,left_index=True,right_index=True,how='inner')
		
      
        return result



    numeric=list(Mdl_Dev_data.select_dtypes(exclude=['object']).columns)
    univariate_check_cols=[col for col in numeric if col in Score_Var_to_Check]
    
 
    for index_num, col in enumerate(univariate_check_cols):
        mdl_analysis	= Decile_Avg_Mid(col, Mdl_Dev_data,     'MDL_Decile')
        scoring_analysis= Decile_Avg_Mid(col, New_Scoring_data, 'Score_decile')
        result = pd.concat([mdl_analysis, scoring_analysis], keys=[mdl_name, scr_name],axis=1)
        result.index.name = col        
        result.to_excel(writer,'04-Univariate Analysis',index=True,startrow=(result.shape[0]+8)*index_num,startcol=0)
		
        
        start=(result.shape[0]+8)*index_num+4
        end=(result.shape[0]+8)*index_num+13
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({'values': '''='04-Univariate Analysis'!$B$%i:$B$%i'''%(start,end),'name':'%s Average Score'%mdl_name})
        chart.add_series({'values': '''='04-Univariate Analysis'!$D$%i:$D$%i'''%(start,end),'name':'%s Average Score'%scr_name})

        chart.set_x_axis({'name': 'Model Decile'})
        chart.set_title({'name':'%s Average by Decile'%col,'name_font': {'size': 12}})
        
        chart.set_legend({'font': {'size': 9},'position': 'top'})
        writer.sheets['04-Univariate Analysis'].insert_chart('I%i'%(start-3), chart)		 
    
    #formater = workbook.add_format({'border':0})
    #writer.sheets['04-Univariate Analysis'].set_column('A:DC',None,formater)
    writer.sheets['04-Univariate Analysis'].hide_gridlines(2)
    writer.sheets['03-Score QC Plot'].hide_gridlines(2)
    writer.sheets['02-Data Attributes QC'].hide_gridlines(2)
    writer.sheets['01-Data Overview'].hide_gridlines(2)


    writer.save()
    writer.close()
    
    print ('')
    print ('Complete! QC report is here! %s'%output_path)
    del Mdl_Dev_data,New_Scoring_data
    gc.collect()

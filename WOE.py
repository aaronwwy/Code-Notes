# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:49:32 2017

@author: wwu
"""


class WOE():
    """ 
        Calculate WOE and Information Value of Category and Continuous Variables;
        df: Pandas DataFrame
        self.target: Variable name of self.target in dataframe
        category_max: the maximum number of distinct values to determine category variable
        bins: number of intervals for continuous variable
    """    
    def __init__(self, df, target, category_max=5, bins=10):
        #import numpy as np
        
        self.data=df

        self.IV={}
        self.WOE={}
        self.var_category={'Category':[],'Continuous':[]}
        self.category_max=category_max
        self.bins=bins
        self.target=target
        #self.good_columns=self.data.select_dtypes(exclude=['object']).columns   
        
     
    def var_classification(self):  
        '''Classify variables into different groups, category and continuous vars '''
        self.var_category['Category']=list(self.data.select_dtypes(include=['object']).columns)
        self.var_category['Continuous']=list(self.data.select_dtypes(exclude=['object']).columns)
        
        con_to_category=[]
        for col in self.var_category['Continuous']:
            if self.data[col].nunique() <= self.category_max:
                #print (col)
                con_to_category.append(col)
                self.var_category['Category'].append(col)
        self.var_category['Continuous']=[col for col in self.var_category['Continuous'] if col not in con_to_category]
        
        
        if self.target in self.var_category['Category']:
            self.var_category['Category'].remove(self.target)
        if self.target in self.var_category['Continuous']:
            self.var_category['Continuous'].remove(self.target)
        

    def woe_category(self,column,save_name,chart=False):
        '''Calculate Information Value of Category Variable'''
        data=self.data.dropna(subset=[column])
        
        
        unique=np.unique(data[column])
        result={}
        IV=0
        freq_table=data.groupby([self.target,column])[column].count().to_frame(name='Count').reset_index()
        
        temp=freq_table.groupby(self.target)['Count'].sum().to_frame(name='#obs').reset_index()
        events=temp[temp[self.target]==1]['#obs'].values[0]
        non_events=temp[temp[self.target]==0]['#obs'].values[0]
        
        for value in unique:
            if value in freq_table[freq_table[self.target]==1][column].values:
                Events_temp=freq_table[(freq_table[self.target]==1) & (freq_table[column]==value)]['Count'].values[0]
            else:
                Events_temp=0.5
            if value in freq_table[freq_table[self.target]==0][column].values:
                Non_Events_temp=freq_table[(freq_table[self.target]==0) & (freq_table[column]==value)]['Count'].values[0]
            else:
                Non_Events_temp=0.5
       
                       
            Non_events_p=(float(Non_Events_temp)/non_events)
            events_p=(float(Events_temp)/events) 

            result[value]= np.log(events_p / Non_events_p)
            IV+=(events_p - Non_events_p) * result[value]
        
        
        
        self.WOE[save_name]=result
        self.IV[save_name]=IV
        
        if chart==True:
            lists = sorted(result.items())
            x,y = zip(*lists)    
            plt.bar(range(len(result)), y, align='center',color='grey')
            plt.xticks(range(len(result)), x)
            plt.title('WOE Plot: %s, IV:%f'%(save_name,IV))
            plt.show()
        
        return result
    
    
        
    def woe_continuous(self,column,chart=False):
        ''' Calculate Information Value of Continuous Variable'''
        
        self.data['decile'] =pd.qcut(self.data[column], self.bins, labels=False,duplicates='drop')
        return self.woe_category('decile',column,chart)
        
    ''' 
        
        freq_table=self.data.groupby([self.target,'decile'])[column].count().to_frame(name='Count').reset_index()
       
        
        
        
        thresholds=np.linspace(0.0, 1.0, num=self.bins+1,endpoint=True)      
        intervals=[self.data[column].quantile(q=quantile) for quantile in thresholds]
        intervals=sorted(list(set((intervals))))
        
        result=OrderedDict()
        IV=0    
        for index in range(len(intervals)-1):
            beginpoint=intervals[index]
            endpoint=intervals[index+1]
            if index==len(intervals)-2:
                Events_temp=self.data[(self.data[self.target]==1) & (self.data[column]>=beginpoint) & (self.data[column]<=endpoint)].shape[0]
                Non_Events_temp=self.data[(self.data[self.target]==0) & (self.data[column]>=beginpoint) & (self.data[column]<=endpoint)].shape[0]
            else:
                Events_temp=self.data[(self.data[self.target]==1) & (self.data[column]>=beginpoint) & (self.data[column]<endpoint)].shape[0]
                Non_Events_temp=self.data[(self.data[self.target]==0) & (self.data[column]>=beginpoint) & (self.data[column]<endpoint)].shape[0]                
            
            if Events_temp==0 or Non_Events_temp==0:
                Non_Events_temp+=0.5
                Events_temp+=0.5

            Non_events_p=(float(Non_Events_temp)/self.non_events)
            events_p=(float(Events_temp)/self.events) 
            
            interval=str(round(beginpoint,2)) + '-' + str(round(endpoint,2))
            result[interval]=np.log( events_p / Non_events_p )
            
            IV+=(events_p - Non_events_p) * result[interval]
        
        self.WOE[column]=result
        self.IV[column]=IV    
        
        if chart==True:    
            plt.bar(range(len(result)), result.values(), align='center',color='grey')
            plt.xticks(range(len(result)), range(1,len(result)+1))
            plt.title('WOE Plot: %s, IV:%f'%(column,IV))
            plt.show()
        
        return result
    '''
    def fit(self):
        '''Calculate all variables Information value and rank, plot 15 most important variables'''
		
        self.var_classification()
        for count_cat,col in enumerate(self.var_category['Category']):
            self.woe_category(column=col,save_name=col)
            if count_cat >=100 and count_cat % 100==0:
                print ('%i completed'%count_cat)
        for count,col in enumerate(self.var_category['Continuous']):
            self.woe_continuous(col)
            if count+count_cat >=100 and (count+count_cat) % 100==0:
                print ('%i completed'%(count+count_cat))                
        self.IV=OrderedDict(sorted(self.IV.items(), key=lambda x: x[1],reverse=True))
        '''
        pos = np.arange(15,0,-1)-0.5
        plt.figure()
        plt.title("Information Value Rank")
        plt.barh(pos, list(self.IV.values()[0:15]), align='center',color='r')
        plt.yticks(pos,list(self.IV.keys()[0:15]))
        plt.xlabel('Information Value')
        plt.show()
        '''

    def export(self,path):
        '''Export Information Value Rank to Excel file'''
        result=pd.DataFrame(columns=['Rank','Variable Name','Information Value'])
        result['Variable Name']=self.IV.keys()
        result['Information Value']=self.IV.values()
        result['Rank']=result.index+1
        result=result[['Rank','Variable Name','Information Value']]
        result.to_excel(path,index=False)

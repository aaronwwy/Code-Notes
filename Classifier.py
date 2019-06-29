class classifiers():
    """
    different models for binary classification
    including grid search for feature selection, parameter optimization, campaign performance  
    """
    def __init__(self, df, target): 
        self.train=df.copy()
        self.target=target.copy()
        self.optimal_vars={}
        self.rfe_scores={}
        self.grid_scores={}
            
        exec(open(r'E:\wwu\Modules\Common_Includes.py').read())

    
    def RFE_Feature(self,model,modelname,parameters):
        '''
        Recursive feature selection step
        Example: step=1, cv=2,scoring='log_loss'
        '''
        RFE_selector = RFECV(model, step=parameters['step'],cv=parameters['cv'],scoring=parameters['scoring'])
        RFE_selector.fit(self.train, self.target)
        
        
        self.optimal_vars[modelname]=self.train.columns[RFE_selector.get_support()]
        self.rfe_scores[modelname]=RFE_selector.grid_scores_        
        
        plt.figure()
        plt.title('Model %s Feature Selection, Optimal %d'% (modelname,RFE_selector.n_features_))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(RFE_selector.grid_scores_) + 1), RFE_selector.grid_scores_)
        plt.show()

        

    
    def grid_parameter(self,model,modelname,model_parameters,parameters):
        '''
        Grid search for optimal parameters       
        '''
        Grid_optimal = GridSearchCV(model,model_parameters,cv=parameters['cv'],scoring=parameters['scoring'])

        Grid_optimal.fit(self.train, self.target)
        
        print (Grid_optimal.best_params_)
        self.grid_scores=Grid_optimal.grid_scores_
    
    
    def performance(self,newmodel,variables,n,random_num,validation=0.30,chart=True):
        '''
        Plot top 10 important features, Capture model performance
        '''
        X_train, X_validation, y_train, y_validation =train_test_split(self.train[variables],
                                                                       self.target, 
                                                                       test_size=validation,
                                                                       random_state=random_num)
        newmodel.fit(X_train,y_train)
        
        predict_score_train=newmodel.predict_proba(X_train)[:,1]  
        predict_score_validation=newmodel.predict_proba(X_validation)[:,1]
        try:
            importances = newmodel.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names=np.array(X_train.columns)
        
            importance_rank=importances[indices]
            feature_rank=feature_names[indices]
        
            feature_output=pd.DataFrame()
            feature_output['feature_rank']=feature_rank        
            feature_output['importance_rank']=importance_rank
        except:
            coef = pd.Series(newmodel.coef_[0], index = variables)
            feature_output = pd.concat([coef.sort_values()])
            
        if chart:
            try:
                indices = np.argsort(importances)
                pos = np.arange(indices[len(indices)-n:].shape[0]) + .5
                plt.figure()
                plt.title("Feature importances")
                plt.barh(pos,importances[indices[len(indices)-n:]] , align='center',color='r')
                plt.yticks(pos, feature_names[indices[len(indices)-n:]])
                plt.xlabel('Relative Importance')
                plt.title('Variable Importance')
                plt.show()                       
            except:
                
                plt.figure(figsize=(10,10))
                feature_output.plot(kind = "barh",color='red')
                plt.title("Coefficients in the Logistic Regression Model")
                plt.show()               
                

        feature_output=pd.DataFrame(feature_output)
        campaign_table_train, performanc_result_train,roc_train=KS_Lift(predict_score=predict_score_train, target=y_train,redecile=False)
        campaign_table_validation, performanc_result_validation,roc_val=KS_Lift(predict_score=predict_score_validation, target=y_validation,redecile=False)
        
        plt.title('ROC Curve')
        plt.plot(roc_train['fpr'], roc_train['tpr'], 'b', label = 'Training AUC = %0.2f' % roc_train['auc'])
        plt.plot(roc_val['fpr'], roc_val['tpr'], 'y', label = 'Validation AUC = %0.2f' % roc_val['auc'])
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()        
        
        
        performance_result={}
        performance_result['feature_output']=feature_output
        performance_result['campaign_table_train']=campaign_table_train
        performance_result['campaign_table_validation']=campaign_table_validation        
        performance_result['performanc_result_train']=performanc_result_train 
        performance_result['performanc_result_validation']=performanc_result_validation 
        
        
        return newmodel,performance_result
        
        
    def oot_test(self,model,ootdata,variables,test_target,new_leads=None):
        predict_score_test=model.predict_proba(ootdata[variables])[:,1]
        if new_leads is None:
            campaign_table_test, performanc_result_test,roc_oot=KS_Lift(predict_score=predict_score_test, target=test_target,new_leads=None, redecile=False)
        else:
            campaign_table_test, performanc_result_test,roc_oot=KS_Lift(predict_score=predict_score_test, target=test_target,new_leads=new_leads, redecile=True)            
        return campaign_table_test, performanc_result_test
        
    def incremental_vars(self,model,variables,test_size=0.3,begin=5,end=15):
        '''This function is to test incremental contributions of each varible
         Default: begin with top 5 variables, end with 15 variables
        '''
        if len(variables)<end:
            print ('Error: not enought variables for selection')
        else:
            metrics={'ROC':[],'KS':[],'Log Loss':[],'Top 1 Decile Lift':[]}
            X_train, X_validation, y_train, y_validation =train_test_split(self.train[variables],
                                                                                             self.target, 
                                                                                             test_size=test_size)
            print ('Begin test variables----:')
            for index in range(begin,end+1):
                model.fit(X_train[variables[0:index]],y_train)
                predict_score_test=model.predict_proba(X_validation[variables[0:index]])[:,1]
                #campaign_table, performanc_result,roc_train=KS_Lift(predict_score=predict_score_test, target=y_validation,redecile=False)
                '''Save Performance Metric'''
                #metrics['ROC'].append(performanc_result['ROC'])
                #metrics['KS'].append(performanc_result['KS'])
                #metrics['Top 1 Decile Lift'].append(performanc_result['Top 1 Decile Lift'])
                metrics['Log Loss'].append(log_loss(y_validation,predict_score_test))
                print ('%i Variables Tested')%index
        '''
        plt.figure(figsize=(50,50))
        fig, ax = plt.subplots(2,2)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5,hspace = 0.5)
        ax[0, 0].plot(range(begin,end+1),metrics['KS'])
        ax[0, 0].set_title('KS')
        ax[0, 1].plot(range(begin,end+1),metrics['ROC'])
        ax[0, 1].set_title('ROC')
        ax[1, 0].plot(range(begin,end+1),metrics['Top 1 Decile Lift'])
        ax[1, 0].set_title('Top 1 Decile Lift')
        ax[1, 1].plot(range(begin,end+1),metrics['KS'])
        ax[1, 1].set_title('Log Loss')
        plt.show()  
        '''
        return metrics
                 
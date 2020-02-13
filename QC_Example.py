# -*- coding: utf-8 -*-
"""
Checking Balance Model Scoring QC
Created at 04/04/2018
@author: cqian
"""


mdl_transform_path=r'E:\Santander_CAR\Cheng Qian\Checking Balance Model\MAY18QC.csv'
score_transform_path=r'E:\Santander_CAR\Cheng Qian\Checking Balance Model\QC\C418_CHK_QC.csv'


'''Specify variable name of socres'''
MDL_Score='bal_qc_score'      #the variable name of scores in model development datasets
Data_Score='bal_qc_score' # the variable name of scores in new scoring datasets

mdl_name='Model Development'
score_name='Scoring Dataset'

MDL_Decile=None
Data_Decile=None
''' Specify variables you need to check'''

Score_Var_to_check=['n9356', 'AP000446', 'nP005121', 'n9358', 'nP005196', 'AP001294',
'avg_deposit', 'nflc1000', 'per_deposit', 'nP005134', 'n9351',
'n7607', 'nP004976', 'hh_mma', 'nP005106', 'compete7', 'n7616',
'nflcf998', 'compete15', 'nP004971','bal_qc_score']


''' Specify the output file, output must be excel format'''
#output_path=r'E:\Santander_CAR\Cheng Qian\PLL Cross Sell Targeting Model\PLL Model Scoring\Q4_PLL_Distribution.xlsx'

output_path=r'E:\Santander_CAR\Cheng Qian\Checking Balance Model\example.xlsx'


exec(open(r'E:\Santander_CAR\Cheng Qian\Modules\Model Scoring Data QC Automation.py').read())
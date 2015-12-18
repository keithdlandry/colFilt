# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:58:02 2015

@author: keithlandry
"""


import pandas as pd


def readArchPred(pos, weeknumber):
    
    path = '/Users/keithlandry/Desktop/fantasyFootballAnalysis/colFilt/projections/' + pos + 'week_' + str(weeknumber) + '.cvs'    
    
    df = pd.read_csv(path)
    return df    
    
    
#Currently im achiving the predicitons in a csv. They are in he raw format outputted by the learing function.
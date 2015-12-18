# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:12:31 2015

@author: keithlandry
"""



from __future__ import division

import numpy as np
import pandas as pd
import itertools



def getWeekProj(df, weekNumber):

    #load file containg schedule grid
    #make sure the team nicknames are the same as I use when getting stats from urls
    fullSched = pd.read_csv("/Users/keithlandry/Desktop/fantasyFootballAnalysis/colFilt/nflSched2015.csv", header = None)

    weekGames = np.vstack((fullSched[0] , fullSched[weekNumber])).T #stackthe two arrays onto eachother then transpose so it looks like a nice list
    
    matchups = [t1[0].replace("@","") + t1[1].replace("@","") for t1 in weekGames]

    weekProj = df[(df.team+df.opp).isin(matchups)]
    
    return weekProj
    


#testdf = getWeekProj(finalDF, 12)
    

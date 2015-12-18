# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:35:34 2015

@author: keithlandry
"""

from __future__ import division

import numpy as np
import pandas as pd
import itertools
import re


from colabFiltering import colabFilCostFunc
from colabFiltering import colabFilGrad
from colabFiltering import trainColabFilter
from colabFiltering import thinMatrix
from colabFiltering import varOfListOfMat
from colabFiltering import mean

from footballColabFilterLearning import findBestLambdaNfeats
from footballColabFilterLearning import predToDataFrame
from footballColabFilterLearning import exportWeeklyResults
from footballColabFilterLearning import organizeProject


from gettingStats import getPlayerURLs
from gettingStats import getPlayerNames
from gettingStats import getPlayerWeeklyStats
from gettingStats import fillMatrixFromDF
from gettingStats import getPlayerStatsFromURL

from weeklyProj import getWeekProj

import matplotlib.pyplot as plt

import time




year = 2015
weeknumber = 15
pathToSalaries = '/Users/keithlandry/Downloads/'
pathToSalaries = pathToSalaries + 'FanDuel-NFL-2015-12-20-13996-players-list.csv'
#FanDuel-NFL-2015-12-13-13913-players-list.csv
#FanDuel-NFL-2015-12-20-13996-players-list.csv

#first step is to get stats and create the matrix 

positions = ['QB','RB','WR','TE','K']

predictionsByPos = []

for pos in positions:

    dataFrames = []
    players, teams = getPlayerNames(pos)
    
    #control for busta... god damnit dude he's a tight end and his name is fuckd up so my code doesn't work
    if "'Busta'_Anderson,_Ro" in players:
        indexOfBusta = players.index("'Busta'_Anderson,_Ro")
        del players[indexOfBusta]
        del teams[indexOfBusta]
    
    URLs = getPlayerURLs(pos, 'FD') #needed if using getPlayerStatsFromURL
    
    #create dictionary so I can add team name to finalDF later on
    playerDict = dict(itertools.izip(players,teams))
    
    for player, team in zip(players, teams):
     
        print player, team
        time.sleep(1) #delay before accesing website
        playerDF = getPlayerStatsFromURL(URLs, player, team, pos) #More efficient. Don't have to look up urls every time.   
        #playerDF = getPlayerWeeklyStats(player, pos, team)
        
        dataFrames.append(playerDF)
    
    totalDF = pd.concat(dataFrames, ignore_index = True)
    totalDF = totalDF.convert_objects(convert_numeric = True)
    
    #totalDF = totalDF[totalDF.week < 13] #take stats only up to week 12
    
    
    nflTeams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
                "GB", "HOU", "IND", "JAC", "KC", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
                "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEN", "WAS"]
    
    playerNames = list(pd.unique(totalDF["player"]))
    
    playerMatrix = np.matrix(np.zeros(shape = (len(nflTeams), len(playerNames))))          
    
    
    for player in pd.unique(totalDF["player"]):
        
        tempMat = fillMatrixFromDF(totalDF, player)
        
        playerMatrix = playerMatrix + tempMat
        
    
    print "moving to learning"
    
    #second step is using the matrix to learn
    
    R = (playerMatrix!=0)
    
    #do cv if we want
    #playerMatrixTrain, RTrain, cvAns, indForCV = thinMatrix(playerMatrix,R,.15)
    
    num_teams = np.size(R,0)
    num_players = np.size(R,1)        
    
    #automated cross val
    bestL, bestNum_feat, bestMiss, YTrain = findBestLambdaNfeats(playerMatrix,R,.15)
    
    print "best values:"
    print "nFeats = %d" % bestNum_feat
    print "lmbd = %d" % bestL
    print "accuracy is %d" % bestMiss
    
    
    #start bootstrapping
    nBoot = 100
    percOfDataRem = .15
    
    bootT = []
    bootX = []
    
    bootPred = []
    
    for i in range(0,nBoot):
        
        #change a percentage of the true values in R to false ones 
        #to get a smaller data set to learn on
        #either for cross validation or boot strapping 
        #if bootstrapping I can just ignore cvAns and indForCv
        Y_thin, R_thin, cvAns, indForCV = thinMatrix(playerMatrix,R,percOfDataRem)    
        
        learnedT, learnedX = trainColabFilter(Y_thin,R_thin,bestL,bestNum_feat,num_teams,num_players)
    
        bootT.append(learnedT)
        bootX.append(learnedX)
     
        #strange way to add row means allows this to work fo matrix or array
        predictionb = np.dot(learnedX,learnedT.T)
     
        bootPred.append(predictionb)
        
    meanPred = mean(bootPred)
    varPred  = varOfListOfMat(bootPred)
    
    
    bootstrappedDF = predToDataFrame(playerNames, playerDict, meanPred)
    bootstrappedVar = predToDataFrame(playerNames, playerDict, varPred)
    
    predAndErrorDF = bootstrappedDF.merge(bootstrappedVar, on = ["player", "team", "opp"])
    
    predAndErrorDF = predAndErrorDF.rename(columns = {"fantasy_points_x":"fantasy_points"})
    predAndErrorDF = predAndErrorDF.rename(columns = {"fantasy_points_y":"variance"})
    predAndErrorDF = predAndErrorDF.convert_objects(convert_numeric = True) #must convert to numeric or averaging will not work
    
    
    exportWeeklyResults(predAndErrorDF, weeknumber, pos)
    
    predErrWeek = getWeekProj(predAndErrorDF, weeknumber)
    
    predErrWeek.variance = np.sqrt(predErrWeek.variance)
    
    #attach the salaries
    salaries = pd.read_csv(pathToSalaries)
    salaries['Last Name'] = salaries['Last Name'].str.replace('.', '')
    salaries['Last Name'] = salaries['Last Name'].str.replace(' ', '_')
    
    
    salaries['player'] = salaries['First Name'] + "_" + salaries['Last Name']
    salaries = salaries[['player', 'Salary']]
    
    predWithSals = predErrWeek.merge(salaries, on = ["player"])

    predictionsByPos.append(predWithSals)
    
    


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
from scipy.stats import gaussian_kde
import time




year = 2016
weeknumber = 10
pathToSalaries = '/Users/keithlandry/Downloads/'
pathToSalaries = pathToSalaries + 'FanDuel-NFL-2015-12-20-13996-players-list.csv'
#FanDuel-NFL-2015-12-13-13913-players-list.csv
#FanDuel-NFL-2015-12-20-13996-players-list.csv

#first step is to get stats and create the matrix 

#positions = ['QB','RB','WR','TE','K']
positions = ['QB']
predictionsByPos = []

for pos in positions:

    dataFrames = []
    players, teams = getPlayerNames(pos)
    
    #control for busta, name is strange so code doesn't work (he also never plays so just remove)
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
    totalDF = totalDF[-totalDF.week.isnull()]

    totalDF = totalDF[totalDF.week < 10] #take stats only up to week 12
    if pos == 'RB':
        remPlayers = ['Ronnie_Hillman','Kyle_Juszczyk','Jalston_Fowler','Travaris_Cadet','John_Kuhn','Will_Tukuafu','Michael_Burton']
        totalDF = totalDF[~totalDF.player.isin(remPlayers)]
    if pos == 'TE':
        remPlayers = ['Luke_Stocker','Jermaine_Gresham','Nick_O\'Leary','Luke_Stocker','David_Johnson','David_Johnson','Khari_Lee','Phillip_Supernaw','C.J._Uzomah']    
        totalDF = totalDF[~totalDF.player.isin(remPlayers)]
    if pos == 'K':
        remPlayers = ['Robbie_Gould']
        totalDF = totalDF[~totalDF.player.isin(remPlayers)]

        
    
    nflTeams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
                "GB", "HOU", "IND", "JAC", "KC", "LAR", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
                "OAK", "PHI", "PIT", "SD", "SEA", "SF", "TB", "TEN", "WAS"]
    
    playerNames = list(pd.unique(totalDF["player"]))
    
    playerMatrix = np.matrix(np.zeros(shape = (len(nflTeams), len(playerNames))))          
    
    
    for player in pd.unique(totalDF["player"]):
        
        tempMat = fillMatrixFromDF(totalDF, player)
        
        playerMatrix = playerMatrix + tempMat
        
    
    print "moving to learning"
    
    #second step is using the matrix to learn
    
    R = (playerMatrix!=0)
    
    #do cv if we want, now set up to do this automatically
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
    nBoot = 6000
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
     
        #strange way to add row means allows this to work for matrix or array
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
 
B = np.array(bootPred) 
plyrPreds = B[:,27,4]  
xpoint = np.linspace(6,24,100)
kernDen = gaussian_kde(plyrPreds)
x, y, _ = plt.hist(plyrPreds, facecolor = 'green')
plt.plot(xpoint, kernDen(xpoint)*max(x)/max(kernDen(xpoint)), 'r-', linewidth = 2.0)
plt.xlabel('Predicted Fantasy Points')
plt.title('Tom Brady Week 10 Fantasy Points, N = 1000')
#plt.savefig('tomBweek10Pred.pdf')
plt.show()




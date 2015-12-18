# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:42:49 2015

@author: keithlandry
"""
from __future__ import division

import pandas as pd
import numpy as np
from lxml.html import parse
from urllib2 import urlopen
import time



def getPlayerURLs(pos,scoring):

    league_dict = {'FD': 174424 , 'DK': 174978, 'PPR': 107644, 'NOPPR': 1}
    league_id = league_dict[scoring]

    if pos not in ["QB","WR","RB","TE","K"]:
        print "incorrect possition"
    else:
        url_str = "http://fftoday.com/stats/players?Pos=" + pos
        parsed = parse(urlopen(url_str))
        page = parsed.getroot()
        tables = page.findall('.//table')
        
        main_table = tables[3] #table 3 is the one with the hyperlinks
        
        URLs = []
    
        for a in main_table.xpath('.//td/*/a'):
            URLs.append("http://fftoday.com" + a.get('href') + "?LeagueID=" + str(league_id)) #last part for ppr scoring 
            
        #for a in main_table.xpath('.//td/*/a'):
        #print 'found "%s" link to href "%s"' % (a.text, a.get('href'))
    
    return URLs
    
def getPlayerNames(pos):
    
    if pos not in ["QB","WR","RB","TE","K"]:
        print "incorrect possition"
    else:
        url_str = "http://fftoday.com/stats/players?Pos=" + pos
        parsed = parse(urlopen(url_str))
        page = parsed.getroot()
        tables = page.findall('.//table')
        
        main_table = tables[3] #table 3 is the one with the hyperlinks
        
        rows = main_table.findall('.//table')
                
        playerNames = []
        teams = []
        
        for i,row in enumerate(rows[3:]): #rows[3] is first player entry
            elements = row.findall('.//a') #find all elements in the row
            values = [val.text_content() for val in elements] #make a list of all the values from the row
            #print values
            
            for plyr in values:

                team = plyr.split()[-1]
                firstName = plyr.split()[-2]

                #need to deal with people that have stange names like Jr.
                #probably should just do this by parsing the url directly
                if len(plyr.split()) == 3:
                    lastName = plyr.split()[0][:-1] #take off last character becuase its a comma
                    totalName = firstName + "_" + lastName
                else:
                    lastName = plyr.split()[0] 
                    otherName = plyr.split()[1][:-2] #comma is on the other (Jr.) also remove .
                    totalName = firstName + "_" + lastName + "_" + otherName
                
                #take out the free agents becasuse they don't have stats typically
                if team != "FA":
                    playerNames.append(totalName)
                    teams.append(team)
        
        return playerNames, teams


def getPlayerWeeklyStats(name, pos, team):
    
    URLs = getPlayerURLs(pos)
    
    playerURL = [s for s in URLs if name in s]
    
    url_str = playerURL[0]
    
      
    #check if they have stats for 2015
    html = urlopen(url_str).read()   
    if "2015 Gamelog Stats" in html:

    
        #for test use dion lewis hyperlink
        #url_str = "http://fftoday.com/stats/players/12003/Dion_Lewis?LeagueID=107644"    
        #url_str = "http://fftoday.com/stats/players/11776/Seyi_Ajirotutu"
        
        parsed = parse(urlopen(url_str))
        page = parsed.getroot()
        tables = page.findall('.//table')
    
        #print name
        #print len(tables)    
        
        if len(tables) > 16: #protect against players that don't have any 2015 stats
        
            main_table = tables[16]
            rows = main_table.findall('.//tr')
            
        
            stats = []    
            
            for i,row in enumerate(rows[3:]): #rows[3] is first player entry
                elements = row.findall('.//td') #find all elements in the row
                values = [val.text_content() for val in elements] #make a list of all the values from the row
    
                #insert home or away
                if len(values[1].split()) == 2:
                    values.insert(2,"home")
                else:
                    values.insert(2,"away")
    
                #format the opponent name 
                values[1].split()
                values[1] = values[1].split()[-1]
    
                #insert player name            
                values.insert(0,name)
                
                #insert team name
                values.insert(1,team)                
                
                stats.append(values)
                
                
            if pos == "WR":
                colNames = ["player", "team", "week", "opp", "loc", "res", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "rush_attempts", "rush_yards", "yards_per_rush", "rush_TD", "fantasy_points"]
            if pos == "RB":
                colNames = ["player", "team", "week", "opp", "loc", "res", "rush_attempts", "rush_yards", "yards_per_rush", "rush_TD", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "fantasy_points"]
            if pos == "QB":
                colNames = ["player", "team", "week", "opp", "loc", "res", "completions", "pass_attempts", "comp_perc", "pass_yards", "pass_TD", "ints", "rush_att", "rush_yards", "yards_per_rush", "rush_TD", "fantasy_points"]
            if pos == "TE":
                colNames = ["player", "team", "week", "opp", "loc", "res", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "fantasy_points"]
            if pos == "K":
                colNames = ["player", "team", "week", "opp", "loc", "res", "fgm", "fga", "fg_perc", "EPM", "EPA", "fantasy_points"]
            statsDF = pd.DataFrame(stats)

            #print np.size(statsDF,1)
            #print np.size(colNames)
            
            #protect against player whos html files are essed up (donald brown, etc.)
            if np.size(statsDF,1) == np.size(colNames):
    
                statsDF.columns = colNames
                statsDF = statsDF.convert_objects(convert_numeric = True) #must convert to numeric or averaging will not work

                return statsDF
        
        #QB: week, opp, result, comp, att, comp%, pass yards, pass td, int, rush att, rush yrd, avg, rush TD, fantasy points
        #WR: week, opp, result, target, rec, rec yrds, avg, rec TD, rush att, rush yrd, avg, rush td, fantasy points
        #RB: week, opp, result, rush att, yrds, avg, rush td, targets, rec, rec yrd, avg, rec td, fantasy points
        #TE: week, opp, result, target, rec, rec yrds, avg, rec TD, fantasy points


def getPlayerStatsFromURL(URLs, name, team, pos):
    
    playerURL = [s for s in URLs if name in s]
    url_str = playerURL[0]
    
    #check if they have stats for 2015
    html = urlopen(url_str).read()   
    if "2015 Gamelog Stats" in html:
        
        parsed = parse(urlopen(url_str))
        page = parsed.getroot()
        tables = page.findall('.//table')
    
        #print name
        #print len(tables)    
        
        if len(tables) > 16: #protect against players that don't have any 2015 stats
        
            main_table = tables[16]
            rows = main_table.findall('.//tr')
            
        
            stats = []    
            
            for i,row in enumerate(rows[3:]): #rows[3] is first player entry
                elements = row.findall('.//td') #find all elements in the row
                values = [val.text_content() for val in elements] #make a list of all the values from the row

                #insert home or away
                if len(values[1].split()) == 2:
                    values.insert(2,"home")
                else:
                    values.insert(2,"away")
    
                #format the opponent name 
                values[1].split()
                values[1] = values[1].split()[-1]
    
                #insert player name            
                values.insert(0,name)
                
                #insert team name
                values.insert(1,team)                
                
                #print values
                stats.append(values)
                
                
            if pos == "WR":
                colNames = ["player", "team", "week", "opp", "loc", "res", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "rush_attempts", "rush_yards", "yards_per_rush", "rush_TD", "fantasy_points"]
            if pos == "RB":
                colNames = ["player", "team", "week", "opp", "loc", "res", "rush_attempts", "rush_yards", "yards_per_rush", "rush_TD", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "fantasy_points"]
            if pos == "QB":
                colNames = ["player", "team", "week", "opp", "loc", "res", "completions", "pass_attempts", "comp_perc", "pass_yards", "pass_TD", "ints", "rush_att", "rush_yards", "yards_per_rush", "rush_TD", "fantasy_points"]
            if pos == "TE":
                colNames = ["player", "team", "week", "opp", "loc", "res", "targets", "receptions", "rec_yards", "yards_per_rec", "rec_TD", "fantasy_points"]
            if pos == "K":
                colNames = ["player", "team", "week", "opp", "loc", "res", "fgm", "fga", "fg_perc", "EPM", "EPA", "fantasy_points"]

            statsDF = pd.DataFrame(stats)

            #print np.size(statsDF,1)
            #print np.size(colNames)
            
            #protect against player whos html files are messed up (donald brown, etc.)
            if np.size(statsDF,1) == np.size(colNames):
    
                statsDF.columns = colNames
                statsDF = statsDF.convert_objects(convert_numeric = True) #must convert to numeric or averaging will not work

                return statsDF



def fillMatrixFromDF(df, player):
    
    nflTeams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
                "GB", "HOU", "IND", "JAC", "KC", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
                "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEN", "WAS"]

    playerNames = list(pd.unique(df["player"]))

    matrix = np.matrix(np.zeros(shape = (len(nflTeams), len(playerNames))))          


    
    opponents = pd.unique(df.loc[df["player"] == player]["opp"])
    
    dfPlayer = df.loc[df["player"] == player]
        
    for opp in opponents:
        #print opp
        fantasyPoints = dfPlayer.loc[dfPlayer["opp"] == opp]["fantasy_points"]
        
        #print fantasyPoints        
        #print np.mean(fantasyPoints)
 
       #right now I'm taking the mean if a player has played an opponent multiple times
        matrix[nflTeams.index(opp), playerNames.index(player)] = np.mean(fantasyPoints) #must have df as numeric values or this step will not work
        
    return matrix

    


########################
########################



#
#pos = "K"
#dataFrames = []
#players, teams = getPlayerNames(pos)
#URLs = getPlayerURLs(pos,'FD') #fanduel scoring
#
#for player, team in zip(players, teams):
#
#    print player, team
#    time.sleep(3) #three second delay before accesing website
#
#    #playerDF2 = getPlayerWeeklyStats(player, pos, team)    
#    playerDF2 = getPlayerStatsFromURL(URLs, player, team, pos)    
#    dataFrames.append(playerDF2)
#    
#totalDF_test = pd.concat(dataFrames, ignore_index = True)
#
#week13 = totalDF_test[totalDF_test.week == 13]
#
#
#merged =  week13.merge(predErrWeek13, on = ["player","team","opp"])
#
#merged3 = merged[merged.fantasy_points_y < 5]
#
#nums = merged.fantasy_points_x - merged.fantasy_points_y #atual - pred
#nums2 = merged2.fantasy_points_x - merged2.fantasy_points_y #atual - pred
#
#nums4 = (merged3.fantasy_points_x - merged3.fantasy_points_y)/merged3.fantasy_points_y
#
#plt.hist(nums4*100)

#
#nflTeams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
#            "GB", "HOU", "IND", "JAC", "KC", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
#            "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEN", "WAS"]
#
#playerNames = list(pd.unique(totalDF2["player"]))
#
#wrMatrix = np.matrix(np.zeros(shape = (len(nflTeams), len(playerNames))))          
#
#for player in pd.unique(totalDF2["player"]):
#    
#    tempMat = fillMatrixFromDF(totalDF2, player)
#    
#    wrMatrix = wrMatrix + tempMat


#rbDataFrames = []
#pos = "RB"
#for player in getPlayerNames(pos):
# 
#    print player   
#    playerDF = getPlayerWeeklyStats(player, pos)
#    rbDataFrames.append(playerDF)
#    
#totalRBDF = pd.concat(rbDataFrames, ignore_index = True)



#pos = "WR"
#
#wrDataFrames = []
#
#for player in getPlayerNames(pos):
# 
#    #print player
#    #print wrNames.index(player)
#    playerDF = getPlayerWeeklyStats(player, pos)
#    wrDataFrames.append(playerDF)
# 
#
#   
#    
# #playerDF = getPlayerWeeklyStats("Danny_Amendola","WR")
# 
#   
#totalWRDF = pd.concat(wrDataFrames, ignore_index = True)
#totalWRDF = totalWRDF.convert_objects(convert_numeric = True)
#
#
#
#
##stuff useful for setting up matrix
#
##totalWRDF.loc[totalWRDF["opp"] == nflTeams[5]].loc[totalWRDF["player"] == wrNames2[0]]["fantasy_points"]
##
##fps2 = totalWRDF.loc[totalWRDF["opp"] == nflTeams[0]].loc[totalWRDF["player"] == wrNames2[0]]["fantasy_points"]
##pd.unique(totalWRDF.loc[totalWRDF["player"] == wrNames2[0]]["opp"])
##
##
##wrMatrix[0,0] = fps2
##
##df = totalWRDF
##
#
#nflTeams = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
#            "GB", "HOU", "IND", "JAC", "KC", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
#            "OAK", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEN", "WAS"]
#
#playerNames = list(pd.unique(totalWRDF["player"]))
#
#wrMatrix = np.matrix(np.zeros(shape = (len(nflTeams), len(playerNames))))          
#
#
#for player in pd.unique(totalWRDF["player"]):
#    
#    tempMat = fillMatrixFromDF(totalWRDF, player)
#    
#    wrMatrix = wrMatrix + tempMat
#    
#    
#
#
#
#


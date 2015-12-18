# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:18:14 2015

@author: keithlandry
"""
from __future__ import division

import numpy as np
import pandas as pd
from scipy import optimize
import scipy.io


#Collaborative Filtering

def colabFilCostFunc(parms, Y, R, lmbd, num_feat, num_obj, num_people):
    
    
    Theta = np.reshape(parms[0:num_people*num_feat],(num_people,num_feat))
    X = np.reshape(parms[num_people*num_feat:], (num_obj,num_feat))
        
    #X is np.matrix() of dimension a x b
    #Y is np.matrix() of dimension a x c
    #Theta is np.matrix() of dimension c x b
    #R is np.matrix() of dimension a x c containing 1 for value in Y and 0 for no value in Y
    #lmbd is the coeffecient for regularization
    
    #cost is 3305.9
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    R = np.matrix(R)
    Theta = np.matrix(Theta)

    #cost function:
    a = (np.multiply(X*Theta.T,R) - Y ).T * (np.multiply(X*Theta.T,R) - Y)
    
    regTermJ = lmbd/2 * ( np.trace(Theta.T*Theta) + np.trace(X.T*X) )       
        
    J = 1/2 * np.trace(a) + regTermJ

    return J
    
    

def colabFilGrad(parms, Y, R, lmbd, num_feat ,num_obj, num_people):
    

    Theta = np.reshape(parms[0:num_people*num_feat],(num_people,num_feat))
    X = np.reshape(parms[num_people*num_feat:], (num_obj,num_feat))
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    R = np.matrix(R)
    Theta = np.matrix(Theta)
    
    
    #gradients:
    regTermTheta_grad = lmbd*Theta
    Theta_grad = ( np.multiply(Theta*X.T,R.T) - Y.T ) * X + regTermTheta_grad   

    regTermX_grad = lmbd*X;
    X_grad = ( np.multiply(X*Theta.T, R) - Y ) * Theta + regTermX_grad


    thetaArray = np.array(Theta_grad).flatten()
    xArray = np.array(X_grad).flatten()

    #return Theta_grad, X_grad
    return np.append(thetaArray, xArray)
    
     


def trainColabFilter(Y, R, lmbd, num_feat, num_obj, num_people):


    #randomize initial values
    initial_X = np.random.rand(num_obj,num_feat)
    initial_Theta = np.random.rand(num_people,num_feat)





##for testing dont set random####
#  
#    initial_X = np.array([2,3,7,8,3,2,1,9])
#    initial_X.shape = (4,2)
#    initial_Theta = np.array([1,9,8,5,3,7])
#    initial_Theta.shape = (3,2)
#
#################################

    initX_array = np.array(initial_X).flatten()
    initTheta_array = np.array(initial_Theta).flatten()



    #initial vals and argument list to pass to minimization function
    initial_Vals = np.append(initTheta_array, initX_array)
    args = (Y,R,lmbd,num_feat,num_obj,num_people)
     
    print("initial cost = ", colabFilCostFunc(initial_Vals,Y,R,lmbd,num_feat,num_obj,num_people) )
    res1 = optimize.fmin_cg(colabFilCostFunc, x0 = initial_Vals, fprime=colabFilGrad, args=args)

    finalTheta = np.reshape(res1[0:num_people*num_feat],(num_people,num_feat))

    finalX = np.reshape(res1[num_people*num_feat:], (num_obj,num_feat))
    
    return finalTheta, finalX
    
    
def normalizeY(Y, R):
        
    m = np.size(Y,0)  #rows
    n = np.size(Y,1)  #cols
        
    #normalize counting only nonzero elements
    rowSums  = np.sum(Y,1)
    nonZerosRow = (Y != 0).sum(1)
    
    colSums = np.sum(Y,0)
    nonZerosCol = (Y != 0).sum(0)
    
    rowNorms = rowSums/nonZerosRow

    #totalNonZeros = nonZerosRow + nonZerosCol    
    
    #totalSums = colSums + rowSums - Y #minus Y to offset double counting
    
    #norms = totalSums/(totalNonZeros - R)

    #Ynorm = Y - norms
    
    #return Ynorm, norms

    #do I want to normalize by row mean or by total mean?

    rowNorms = rowNorms.reshape(1,m) #reshaping allows for handling of array and matrix        
    Ynorm2 = (Y.T - rowNorms).T
    return np.multiply(Ynorm2,R), rowNorms
    

def thinMatrix(Y, R, fracToThin):
   

    num_obj = np.size(R,0)
    num_people = np.size(R,1)
    
    #print num_obj
   
    r = np.array(R).flatten()

    #find the spots in r where we actually have data
    indxOfData = np.array(np.where(r==True))
    indxOfData.shape = (np.size(indxOfData,1)) #not sure why this is needed

    indToRemove = np.random.rand(np.size(indxOfData)) < fracToThin

    #set aside indexes to cross validation data if we want to use it
    indForCV = indxOfData[indToRemove]
    y = np.array(Y).flatten()
    cvAns = y[indForCV]

    r[indxOfData[indToRemove]] = 0
    
    R_thin = r.reshape(num_obj,num_people)
    Y_thin = np.multiply(Y,R_thin)
    
    return Y_thin, R_thin, cvAns, indForCV
    
def varOfListOfMat(listM):
    
    #listM 
    listM2 = []
    
    for iM in listM:
        listM2.append(np.multiply(iM,iM))
        
    var = sum(listM2)/len(listM2) - np.multiply(sum(listM),sum(listM))/len(listM)**2
    
    return var
    
def mean(listM):
    
    mean = sum(listM)/len(listM)
    
    return mean

################################
    
    
########## 
#  let me run some test first
# first test my cost function and gradient
##########
    
##test cost function on pretrained x and theta shoud get 27918.64 according to coursera
#
#preTrained = scipy.io.loadmat('/Users/keithlandry/Downloads/machine-learning-ex8/ex8/ex8_movieParams.mat')
#
#preTrainedX = preTrained["X"]
#preTrainedTheta = preTrained["Theta"]
#
#
#preTrainX_array = np.array(preTrainedX).flatten()
#preTrainedTheta_array = np.array(preTrainedTheta).flatten()
#preTraindParams = np.append(preTrainedTheta_array, preTrainX_array)
#testCost = colabFilCostFunc(preTraindParams,Y,R,0,num_feat,num_obj,num_people)
##looks like it's workng
#
#
##test gradients  make a small problem by hand pickin y x and theta
#y = np.array([3,0,8,0,2,0,3,0,7,4,0,7])
#y.shape = (4,3)
#x = np.array([2,3,7,8,3,2,1,9])
#x.shape = (4,2)
#theta = np.array([1,9,8,5,3,7])
#theta.shape = (3,2)
#r = (y!=0)
#r = np.array([[1,0,1],[0,1,0],[1,0,1],[1,0,1]])
#
#x_array = np.array(x).flatten()
#theta_array = np.array(theta).flatten()
#parmsForGradCheck = np.append(theta_array, x_array) 
#
#num_feat = np.size(x,1)
#num_obj  = np.size(y,0)
#num_people = np.size(theta,0)
#
#
##testGrads = colabFilGrad(parmsForGradCheck,y,r,0,num_feat,num_obj,num_people)
###This is also working 
#
#
##test entire thing
#
#lT, lX = trainColabFilter(y,r,2,num_feat,num_obj,num_people) 
#pred = np.dot(lX,lT.T)
#looks like this gives just about the same answer as when I did it in matlab

#DONE WITH TESTS everything looks like it's working

######################



#real FF number
#       Hearns Calvin Odel Math Aiken Cobb Diggs Thomas
# bal
# buf
# NE
# Oak
# SD
# SF
# Den

#Y = np.matrix([[17.2,0,0,0,0,0,0,13],[13.2,0,8.8,19.35,0,0,0,0],[15,0,20.4,13.2,0,0,0,0],[0,0,0,0,11.9,0,7.6,10.5],[0,5.9,0,0,12.2,5.8,0,0],[0,0,25.4,0,11.2,0,0,0],[0,15.7,0,0,0.9,9.7,14.7,0]])
#Y = np.matrix([[17.2,0,0,0,0,0,0,13],[13.2,0,8.8,19.35,0,0,0,0],[15,0,20.4,13.2,0,0,0,0],[0,0,0,0,11.9,0,7.6,10.5],[0,5.9,0,0,0,5.8,0,0],[0,0,25.4,0,11.2,0,0,0],[0,15.7,0,0,0.9,9.7,14.7,0]])


#movies = scipy.io.loadmat('/Users/keithlandry/Downloads/machine-learning-ex8/ex8/ex8_movies.mat')
#
#
#Y = movies["Y"]
#R = movies["R"]
##R = (Y!=0)
#  
#  
#num_obj = np.size(R,0)
#num_people = np.size(R,1)    
#    
#lmbd = 10
# 
#num_feat = 10
# 
#nBoot = 1
#percOfData = .7
#
#bootT = []
#bootX = []
#
#bootPred = []
#
##start bootstrapping
#for i in range(0,nBoot):
#    
#    #change a percentage of the true values in R to false ones 
#    #to get a smaller data set to learn on
#    #either for cross validation or boot strapping 
#    #if bootstrapping I can just ignore cvAns and indForCv
#    Y_thin, R_thin, cvAns, indForCV = thinMatrix(Y,R,0)    
#    
#    
#    #dont want to do it like this because I want to keep num_obj and num_people costant
#    #Y_small = Y[ind1,:][:,ind2]
#    #R_small = R[ind1,:][:,ind2]
#    
#    
#    #normalize the "ratings"    
#    Y_norm, rowMeans = normalizeY(Y_thin,R_thin)
#    #print Y_norm
#    #Y_norm, norms = normalizeY(Y_thin, R_thin)
#    learnedT, learnedX = trainColabFilter(Y,R,lmbd,num_feat,num_obj,num_people)
#
#    bootT.append(learnedT)
#    bootX.append(learnedX)
# 
#    #strange way to add row means allows this to work fo matrix or array
#    #predictionb = np.dot(learnedX,learnedT.T)
#    #predictionb = ((np.dot(learnedX,learnedT.T)).T + rowMeans).T
#    #predictionb = np.dot(learnedX,learnedT.T) + norms
#    predictionb = np.dot(learnedX,learnedT.T)
#    #print predictionb
# 
#    bootPred.append(predictionb)
# 
##print "\n learned X " 
##print learnedX
##print "\n learned Thetea"
##print learnedT
# 
#meanPred = mean(bootPred)
#varPred  = varOfListOfMat(bootPred)
# 
#predDF  = pd.DataFrame(meanPred)
#varPredDF = pd.DataFrame(varPred)
#
#print " \n \n Var "
##print varPredDF.head()
#print " \n"
##print Y_thin
##print Y
#
#print " \n \n Prediction "
#print predDF.head(10)
#print " \n \n norms "
##print norms
#
#predictionb = np.array(predictionb).flatten()
#avgMiss = 1/len(cvAns) * sum(np.abs(predictionb[indForCV] - cvAns))
#
#print cvAns
#print predictionb[indForCV]
#print avgMiss
#
#
##learnedT, learnedX = trainColabFilter(Y,R,lmbd,num_feat,num_obj,num_people)
#
##print "\n"
##print learnedT
##print"\n"
##print learnedX
#
#
#
#
##learnedT = np.matrix(learnedT)
##learnedX = np.matrix(learnedX)
#
#
##prediction = learnedX*learnedT.T
#
##predDF = pd.DataFrame(prediction)
#
##predDF.head()
#
#
#
#
#
#
#

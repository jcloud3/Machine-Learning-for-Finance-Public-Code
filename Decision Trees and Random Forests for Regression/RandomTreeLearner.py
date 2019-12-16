import numpy as np
import random

class RandomTreeLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
		'''
		Initializes the Random Tree Learner with an empty tree, in anticipation of training
		
		Parameters:
		leaf_size (int) - maximum leaf size of the tree
		'''
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_tree(self, data):
		'''
		Constructs tree given training data.  Chooses a random feature upon which to split.
		
		Parameters:
		data (nparray) - array of observational features
		
		Returns:
		tree (nparray) - array representation of the tree.
		'''
        #print(data)
        if data.shape[0] <= self.leaf_size or np.all(data[0,-1]==data[:,-1],axis=0):
            return np.array([["leaf",np.mean(data[:,-1]),np.nan, np.nan]])
        else:
            i = random.randint(0, data.shape[1]-2)
            r1 = random.randint(0, data.shape[0]-1)
            r2 = random.randint(0, data.shape[0]-1)
            split_val = (data[r1,i]+data[r2,i])/2
            
            #split_val = np.median(data[:,i])
            
            max_val = max(data[:,i])
            if max_val == split_val:
                return np.array([["leaf", np.mean(data[:,-1]), np.nan, np.nan]])
            
            left_tree = self.build_tree(data[data[:,i]<=split_val])
            right_tree = self.build_tree(data[data[:,i]>split_val])
            root = np.array([[i, split_val, 1, left_tree.shape[0]+1]])
            return np.vstack((root, left_tree, right_tree))

    def addEvidence(self, Xtrain, Ytrain):
		'''
		Trains learners on in-sample data set.  This learner is not online, so will construct a new tree.
		
		Parameters:
		Xtrain (nparray) - Observational data in feature columns
		Ytrain (nparray) - Ground-truth values for each observation
		'''
        Ytrain = np.reshape(Ytrain, (Ytrain.shape[0],-1))
        data = np.append(Xtrain, Ytrain, axis = 1)
        self.tree =  self.build_tree(data)

    def query(self, Xtest):
		'''
		Queries learners with out-of-sample data set.
		
		Parameters:
		Xtest (nparray) - Observation data in feature columns
		
		Returns:
		Ytest (nparray) - Regression values for the query
		'''
        pred = []
        for test in Xtest:
            pred.append(float(self.search_tree(test)))
        return pred

    def search_tree(self, test):
		'''
		Helper function for queries to iteratively search tree given features
		
		Parameters:
		test (nparray) - input features
		
		Returns:
		tree_row (nparray) - row of the decision tree for that test feature
		'''
        row = 0
        while(self.tree[row,0]!="leaf"):
            rf, rs, rl, rr = self.tree[row]
            if test[int(float(rf))]<=float(rs):
                row = row + int(float(self.tree[row,2]))
            else:
                row = row + int(float(self.tree[row,3]))
        return self.tree[row,1]
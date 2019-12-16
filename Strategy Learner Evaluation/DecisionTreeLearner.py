import numpy as np

class DecisionTreeLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
		'''
		Initializes the Decision Tree Learner with an empty tree, in anticipation of training
		
		Parameters:
		leaf_size (int) - maximum leaf size of the tree
		'''
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def build_tree(self, data):
		'''
		Constructs tree given training data.  Chooses feature with highest correlation to y-value.
		
		Parameters:
		data (nparray) - array of observational features
		
		Returns:
		tree (nparray) - array representation of the tree.
		'''
        # If remaining rows equal to or less than the leaf size, it's a leaf!
        if data.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        # If the remaining Y-values are all the same, it's a leaf!
        if np.all(data[0, -1] == data[:, -1], axis=0):
            return np.array([["leaf", data[0, -1], np.nan, np.nan]])
        else:
            # Determine feature with max correlation to data_y
            correlations = np.abs(np.corrcoef(data[:, :-1], y=data[:, -1], rowvar=False))[:-1, -1]
            best_i = np.nanargmax(correlations)
            # Determine split_val
            split_val = np.median(data[:, best_i])
            # Sanity check for edge case
            max_val = max(data[:, best_i])
            if split_val == max_val:
                return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
            # Create root, left and right trees, and return up the stack.
            left_tree = self.build_tree(data[data[:, best_i] <= split_val])
            right_tree = self.build_tree(data[data[:, best_i] > split_val])
            root = np.array([[best_i, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def addEvidence(self, Xtrain, Ytrain):
		'''
		Trains learners on in-sample data set.  This learner is not online, so will construct a new tree.
		
		Parameters:
		Xtrain (nparray) - Observational data in feature columns
		Ytrain (nparray) - Ground-truth values for each observation
		'''
        Ytrain = np.reshape(Ytrain, (Ytrain.shape[0], -1))
        data = np.append(Xtrain, Ytrain, axis=1)
        self.tree = self.build_tree(data)

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
        while self.tree[row, 0] != "leaf":
            rf, rs, rl, rr = self.tree[row]
            if test[int(float(rf))] <= float(rs):
                row = row + int(float(self.tree[row, 2]))
            else:
                row = row + int(float(self.tree[row, 3]))
        return self.tree[row, 1]
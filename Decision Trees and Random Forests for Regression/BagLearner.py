import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
		'''
		Initializes BagLearner instance
		
		Parameters:
		learner (object) - name of learner from which to create ensemble
		kwargs (arguments) - kwargs to pass to learner instantiation
		bags (int) - number of learners in the ensemble
		
		'''
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.tree = None
        self.learners = []
        for _ in range(self.bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, Xtrain, Ytrain):
		'''
		Trains learners on in-sample data set.
		
		Parameters:
		Xtrain (nparray) - Observational data in feature columns
		Ytrain (nparray) - Ground-truth values for each observation
		'''
        n = Xtrain.shape[0]     # Calculate size of bag (= size of data set)
        for learner in self.learners:
            bag_indices = np.random.choice(n, n)  # sample from n rows n times (with replacement)
            learner.addEvidence(Xtrain[bag_indices], Ytrain[bag_indices])
        pass

    def query(self, Xtest):
		'''
		Queries learners with out-of-sample data set.
		
		Parameters:
		Xtest (nparray) - Observation data in feature columns
		
		Returns:
		Ytest (nparray) - Regression values for the query
		'''
        # Construct array of queries for each learner on the testing data.
        bag_queries = np.array([learner.query(Xtest) for learner in self.learners])
        # Return the mean since regression.
        return np.mean(bag_queries, axis=0)
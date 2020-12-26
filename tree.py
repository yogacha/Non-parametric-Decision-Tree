# %%
import numpy as np
from typing import List


class TauTree():
    """
    Attribute
    ---------

    tree:
        tree of decision rules, (ith dimension, criterion)
        go left if Xi < criterion.
                 (1, 2.2)
                  /    \ 
            (0, 8.1)  (3, 7.2)
        ...
    
    
    """

    def __init__(self, depth=4, min_split=2):
        self.sgn_matrix = None
        self.depth = depth
        self.n_split = 2**self.depth - 1  # number of non-leaf nodes in tree
        self.min_split = min_split

    def fit(self, X, y):
        """
        X, np.array:
            shape = (n, d)
        """
        self.n, self.dim = X.shape
        self.sgn_matrix, self.orders = self.precompute(X, y)
        self.y = y
        
        self.tree = []         # list(binary tree) of tuples of (dim to split, stump)
        self.leaf_members = [       # list(binary tree) of List[int(id)]
            list(range(self.n))   # all belongs to root
        ]

        # split every leaf
        for leaf_ind in range(self.n_split):  # binary tree leaf id

            if not self.can_split(leaf_ind):
                self.not_split()
                continue
            
            dim, left, right = self.best_split(
                sign_matrix=self.sgn_matrix,
                order_list=self.orders,
                members=self.leaf_members[leaf_ind]
            )
            
            # found best split for ith leaf...
            self.leaf_members.extend( [ left, right ] )
            boundary = [ left[-1], right[0] ]
            self.tree.append(
                ( dim, X[boundary, dim].mean() )  # middle point as stump
            )
    
    
    def predict(self, X, estimate_func=np.median, min_df=1):
        
        estimations = np.array([
            ( estimate_func( self.y[leaf_member] ) 
             if len(leaf_member) >= min_df 
             else 0 )
            for leaf_member in self.leaf_members
        ])

        group = np.zeros(X.shape[0], dtype=np.int)

        for leaf_ind, (dim, stump) in enumerate(self.tree):
            member = (group == leaf_ind)
            val = X[member, dim]

            new_group = np.where(
                val < stump, 2*leaf_ind+1, 2*leaf_ind+2
            )
            new_group[ np.isnan(val) ] = leaf_ind

            group[member] = new_group

        return estimations[group]


    @staticmethod
    def precompute(X, y):
        y = np.array(y).reshape(-1, 1)
        # equilavent to sgn_matrix = np.sign(y - y.T)
        sgn_matrix = (y > y.T).astype(np.int8)
        sgn_matrix[y < y.T] = -1

        order_list = [
            np.argsort(Xd)
            for Xd in X.T
        ]
        return sgn_matrix, order_list


    @staticmethod
    def best_split(sign_matrix, order_list, members: List[int]):
        maximum = -1
        # iterate all dimensions
        for d, order in enumerate(order_list):
            sub_order = order[ np.isin(order, members) ]  # subsequence
            abs_tau = np.abs(
                sign_matrix[np.ix_(sub_order, sub_order)]
                    .sum(axis=1)
                    .cumsum()
            )
            max_tau = abs_tau.max()
            if maximum < max_tau:
                maximum, dim = max_tau, d
                size_left = abs_tau.argmax() + 1
                left, right = np.split( sub_order, [size_left] )

        return dim, left, right


    def can_split(self, leaf_ind):
        size = len(self.leaf_members[leaf_ind])
        if (size == 0):
            return False
        elif (self.min_split != None) and (size < self.min_split):
            return False
        else:
            return True


    def not_split(self):
        self.leaf_members.extend( [ [], [] ] )
        self.tree.append( None )

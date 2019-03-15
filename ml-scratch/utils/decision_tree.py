import  argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--criterion', default='gini', type=str,
                    help='criterion')
parser.add_argument('--splitter', default='best', type=str,
                    help='The strategy used to choose the split at each node')
parser.add_argument('--max_depth', default=None, type=int,
                    help='depth')
parser.add_argument('--min_samples_split', default=2, type=int,
                    help='min_samples_split')
parser.add_argument('--min_samples_leaf', default=1, type=int,
                    help='added to the decision function')
parser.add_argument('--min_weight_fraction_leaf', default=0.0, type=float,
                    help='min_weight_fraction_leaf')
parser.add_argument('--max_features', default=None, type=int or float or str,
                    help='max_features')
parser.add_argument('--random_state', default=None, type=int,
                    help='random_state')
parser.add_argument('--max_leaf_nodes', default=None, type=int,
                    help='Grow a tree')
parser.add_argument('--min_impurity_decrease', default=0.0, type=float,
                    help='min_impurity_decrease')
parser.add_argument('--min_impurity_split', default=None, type=float,
                    help='min_impurity_split')
parser.add_argument('--class_weight', default=None, type=dict,
                    help='class_weight')
parser.add_argument('--presort', default=False, type=bool,
                    help='presort')
parser.add_argument('--file_path', default=None, type=str,
                    help='file_path')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='test_size')                    
                    
# function作成
def decision_tree_classifier(args):
    #print(args.file_path)
    df = pd.read_csv(args.file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)
    clf = DecisionTreeClassifier(args.criterion,
                                 args.splitter, 
                                 args.max_depth,
                                 args.min_samples_split,
                                 args.min_samples_leaf,
                                 args.min_weight_fraction_leaf,
                                 args.max_features,
                                 args.random_state,
                                 args.max_leaf_nodes,
                                 args.min_impurity_decrease,
                                 args.min_impurity_split,
                                 args.class_weight,
                                 args.presort,)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print(score)
    print(y_pred, y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    decision_tree_classifier(args)

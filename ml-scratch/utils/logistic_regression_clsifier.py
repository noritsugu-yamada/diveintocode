import  argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--penalty', default='l2', type=str,
                    help='penalty type')
parser.add_argument('--dual', default=False, type=bool,
                    help='Dual or primal formulation.')
parser.add_argument('--tol', default=0.0001, type=float,
                    help='tol')
parser.add_argument('--C', default=1.0, type=float,
                    help='Inverse of regularization strength')
parser.add_argument('--fit_intercept', default=True, type=bool,
                    help='added to the decision function')
parser.add_argument('--intercept_scaling', default=1, type=float,
                    help='intercept_scaling is appended to the instance vector')
parser.add_argument('--class_weight', default=None, type=dict,
                    help='Weights associated with classes')
parser.add_argument('--random_state', default=None, type=int,
                    help='random_state')
parser.add_argument('--solver', default='warn', type=str,
                    help='Algorithm to use in the optimization problem')
parser.add_argument('--max_iter', default=100, type=int,
                    help='Maximum number of iterations')
parser.add_argument('--multi_class', default='warn', type=str,
                    help='multi class')
parser.add_argument('--verbose', default=0, type=int,
                    help='set verbose')
parser.add_argument('--warm_start', default=False, type=bool,
                    help='reuse the solution of the previous call to fit as initialization')
parser.add_argument('--n_jobs', default=None, type=int,
                    help='Number of CPU cores')
parser.add_argument('--file_path', default=None, type=str,
                    help='file_path')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='test_size')                    
                    
# function作成
def logistic_regression(args):
    #print(args.file_path)
    df = pd.read_csv(args.file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)
    clf = LogisticRegression(args.penalty,
                             args.dual, 
                             args.tol,
                             args.C,
                             args.fit_intercept,
                             args.intercept_scaling,
                             args.class_weight,
                             args.random_state,
                             args.solver,
                             args.max_iter,
                             args.multi_class,
                             args.verbose,
                             args.warm_start,
                             args.n_jobs
                             )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print(score)
    print(y_pred, y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    logistic_regression(args)
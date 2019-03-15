import  argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--C', default=1.0, type=float,
                    help='Penalty parameter C')
parser.add_argument('--kernel', default='rbf', type=str,
                    help='kernel')
parser.add_argument('--degree', default=3, type=int,
                    help='Degree of the polynomial kernel function')
parser.add_argument('--gamma', default='auto', type=str,
                    help='Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’')
parser.add_argument('--coef0', default=0.0, type=float,
                    help='Independent term in kernel function')
parser.add_argument('--shrinking', default=True, type=bool,
                    help='Whether to use the shrinking heuristic')
parser.add_argument('--probability', default=False, type=bool,
                    help='Whether to enable probability estimates')
parser.add_argument('--tol', default=0.001, type=float,
                    help='Tolerance for stopping criterion')
parser.add_argument('--cache_size', default=200, type=float,
                    help='Specify the size of the kernel cache')
parser.add_argument('--class_weight', default=None, type=bool,
                    help='Set the parameter C of class ')
parser.add_argument('--verbose', default=False, type=bool,
                    help='verbose')
parser.add_argument('--max_iter', default=-1, type=int,
                    help='Hard limit on iterations')
parser.add_argument('--decision_function_shape', default='ovr', type=str,
                    help='decision_function_shape')
parser.add_argument('--random_state', default=None, type=int,
                    help='random_state')
parser.add_argument('--file_path', default=None, type=str,
                    help='file_path')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='test_size')                    


# function作成
def support_vector_machine(args):
    
    #print(args.file_path)
    df = pd.read_csv(args.file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)
    clf = SVC(args.C,
              args.kernel, 
              args.degree,
              args.gamma,
              args.coef0,
              args.shrinking,
              args.probability,
              args.tol,
              args.cache_size,
              args.class_weight,
              args.verbose,
              args.max_iter,
              args.decision_function_shape,
              args.random_state)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print(score)
    print(y_pred, y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    support_vector_machine(args)
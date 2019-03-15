import  argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--fit_intercept', default=True, type=bool,
                    help='fit_intercept')
parser.add_argument('--normalize', default=False, type=bool,
                    help='normalize')
parser.add_argument('--copy_X', default=True, type=bool,
                    help='copy_X')
parser.add_argument('--n_jobs', default=None, type=int,
                    help='The number of jobs to use for the computation')
parser.add_argument('--file_path', default=None, type=str,
                    help='file_path')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='test_size')               
                    
# function作成
def linear_regression(args):
    #print(args.file_path)
    df = pd.read_csv(args.file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)
    reg = LinearRegression(args.fit_intercept,
                           args.normalize, 
                           args.copy_X,
                           args.n_jobs)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2_score = reg.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('r2_score {}'.format(r2_score))
    print('mse {}'.format(mse))

if __name__ == '__main__':
    args = parser.parse_args()
    linear_regression(args)
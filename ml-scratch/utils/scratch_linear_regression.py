import  argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='argparse sample')
parser.add_argument('--num_iter', default=50, type=int,
                    help='itertnum')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning_rate')
parser.add_argument('--bias', default=True, type=bool,
                    help='bias')
parser.add_argument('--verbose', default=False, type=bool,
                    help='verbose')
parser.add_argument('--random_state', default=False, type=bool,
                    help='random_seed_fix')
parser.add_argument('--standardization', default=False, type=bool,
                    help='standardization')
parser.add_argument('--logarithm', default=False, type=bool,
                    help='logarithm')
parser.add_argument('--val_true', default=False, type=bool,
                    help='val_true')                  
parser.add_argument('--file_path', default=None, type=str,
                    help='file_path')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='test_size')               


class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    
    num_iter : int
    lr : float
    no_bias : bool
    verbose :bool
    学習過程を出力する場合はTrue
     Attributes
    ----------
    self.coef : theta 次の形のndarray, shape (n_features,)
      parameter
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    
    """
    
    def __init__(self, num_iter, lr, bias, verbose, random_state):
         # ハイパーパラメータを属性として記録
            self.num_iter = num_iter
            self.lr = lr
            self.bias = bias
            self.verbose = verbose
            self.random_state = random_state
            self.loss = np.zeros(self.num_iter)
            self.val_loss = np.zeros(self.num_iter)
            
            
    def _linear_hyposis(self,X):
        
        """線形の仮定関数を計算する
        Parameters
        ----------
        self.X : 次の形のndarray, shape (n_samples, n_features)
        学習データ fitから持ってくる
        Returns
        -------
        次の形のndarray, shape (n_samples, 1)
        線形の仮定関数による推定結果
        """
        # randint(0,100,(5,5)) 0～99までの整数を5*5の行列に
        # self.coef_ = np.random.randint(1,10,(self.X.shape[1]))
        y_pred = np.dot(X, self.coef_)
        #print('_linear_hyposis,y_pred,{}'.format(y_pred))
        
        return y_pred
    
    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算
        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
        推定した値
        y : 次の形のndarray, shape (n_samples,)
        正解値
        Returns
        ----------
        mse : numpy.float
        平均二乗誤差
        """
        #sampleを算出
        ids = len(y)
        # Σ(y_pred - y_true)^2 / 2*m
        mse = np.sum((y_pred - y)**2)/(2*ids)
        
        return mse
    
    def _compute_cost(self, X, y):
        """
        平均二乗誤差を計算する。MSEは共通の関数を作っておき呼び出す
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
        学習データ
        y : 次の形のndarray, shape (n_samples, 1)
        正解値
        Returns
        -------
        次の形のndarray, shape (1,)
        平均二乗誤差
        """
        y_pred = np.dot(X, self.coef_)
        mse = self.MSE(y_pred, y)
        
        return mse
        
    def _gradient_descent(self, X, y):
        """
        説明を記述
        """
        # gradient と mseを算出
        y_pred = self._linear_hyposis(X)
        mse = self.MSE(y_pred, y)
        ids = len(y)
        # MSEの微分が勾配 1/mΣ(y_pred - y_true)X
        grad = np.dot(y_pred - y, X) / ids
        
        # 更新 -lr(/ids) * grad
        lr = self.lr
        self.coef_ = self.coef_ - lr * grad

        return mse
    
    def fit(self,X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        # biasがTrueならX0として要素が1の 次の形のndarray, shape (n_features,)
        if self.bias:
            b = np.ones(X.shape[0])
            # biasをinsertしたXを上書きしインスタンス変数にして格納
            X = np.insert(X, 0, b, axis=1)
        
        # Xの形に合わせてself.coef_ をインスタンス変数に
        # random_state=Trueでseed固定
        if self.random_state:
            rand_seed = 0
            np.random.seed(rand_seed)
        self.coef_ = np.random.randn(X.shape[1])
        # bias = Trueならval_dataにbiasを追加する 
        if (X_val is not None) and (y_val is not None):
            if self.bias:
                b = np.ones(X_val.shape[0])
                X_val = np.insert(X_val, 0, b, axis=1)
            
        # lossの格納とthetaのnum_iter回の更新
        for i in range(self.num_iter):
            mse = self._gradient_descent(X, y)
            self.loss[i] = mse
            
            if self.verbose:
                print('{}回目mse {}'.format(i+1, mse))
            
            # val_lossの格納
            if (X_val is not None) and (y_val is not None):
                y_val_pred = np.dot(X_val, self.coef_)
                val_mse = self.MSE(y_val_pred, y_val)
                self.val_loss[i] = val_mse
                
                if self.verbose:
                    print('{}回目val_loss {}'.format(i+1, val_mse))
                
    def predict(self, X_test):
        # 学習と異なるXで予測
        if self.bias:
            b = np.ones(X_test.shape[0])
        # biasをinsertしたXを上書きしインスタンス変数にして格納
            X_test = np.insert(X_test, 0, b, axis=1)
        # bias=Falseならそのまま
        
        # y_predを算出
        y_pred = np.dot(X_test, self.coef_)
        
        return y_pred

    


# function作成
def linear_regression(args):
    #print(args.file_path)
    df = pd.read_csv(args.file_path)
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values

    # 標準化
    if args.standardization:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # 対数化(正規分布に近づける)
    if args.logarithm:
        y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size)
    reg = ScratchLinearRegression(args.num_iter,
                                  args.lr,  
                                  args.bias,
                                  args.verbose,
                                  args.random_state)
    reg.fit(X_train, y_train, X_test, y_test)
    y_pred = reg.predict(X_test)
    print(reg.MSE(y_pred, y_test))

    plt.title('loss')
    plt.plot(reg.loss, 'o',label='loss')
    if args.val_true:
        plt.plot(reg.val_loss, '--', linewidth=2, label='val_loss')
    plt.xlabel('num_iter')
    plt.ylabel('loss and val_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    linear_regression(args)
class GetMiniBatch:
    """
    ミニバッチを取得するイテレータ

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, 1)
      正解値
    batch_size : int
      バッチサイズ
    seed : int
      NumPyの乱数のシード
    """
    def __init__(self, X, y, batch_size = 10, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)

    def __len__(self):
        return self._stop

    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self.X[p0:p1], self.y[p0:p1]        

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]




class ThreeLayerNetwork():
    """
    シンプルな三層ニューラルネットワーク分類器

    Parameters
    ----------
    lr:学習率
    

    Attributes
    ----------
    """
    def __init__(self, epoch, batch_size, lr, n_features, n_nodes1, n_nodes2, n_output, sigma = 0.01):
        
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_features = n_features
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.n_output = n_output
        self.sigma = sigma     # ガウス分布の標準偏差
        # W, bの初期化
        self.W1 = self.sigma * np.random.randn(self.n_features, self.n_nodes1)
        self.b1 = self.sigma * np.random.randn(self.n_nodes1)[np.newaxis, :]
        self.W2 = self.sigma * np.random.randn(self.n_nodes1, self.n_nodes2)
        self.b2 = self.sigma * np.random.randn(self.n_nodes2)[np.newaxis, :]
        self.W3 = self.sigma * np.random.randn(self.n_nodes2, self.n_output)
        self.b3 = self.sigma * np.random.randn(self.n_output)[np.newaxis, :]
        
        self.losses = []
        self.val_losses = []
        
        
    def forward(self, X):
        
        # 1層目の実装
        A1 = np.dot(X, self.W1) + self.b1
        Z1 = self.tanh(A1)

        # 2層目の実装
        A2 = np.dot(Z1, self.W2) + self.b2
        Z2 = self.tanh(A2)

        # 3層目の実装
        A3 = np.dot(Z2, self.W3) + self.b3
        Z3 = self.softmax(A3)

        self.Z1 = Z1
        self.Z2 = Z2
        
        return Z3
    
    def sigmoid(self, X):
        
        return 1/1+np.exp(-X)
    
    def tanh(self, X):
        
        return (np.exp(X)-np.exp(-X)) / (np.exp(X)+np.exp(-X))
    
    #def softmax(self, X):
        #C = np.max(X)
       # z = X - C
       # Z = np.exp(z)/np.sum(np.exp(z), axis=1)[:,np.newaxis]
        #return Z
    def softmax(self, X):
        if X.ndim == 2:
            X = X.T
            X = X - np.max(X, axis=0)
            y = np.exp(X) / np.sum(np.exp(X), axis=0)
            return y.T
        X = X - np.max(X)
        return np.exp(X) / np.sum(np.exp(X), axis=1)[:,np.newaxis]

    def relu(self, X):
        return np.max(0, X)
    
    def cross_entropy_error(self, y_pred, y):
        cross_entropy_error = np.sum(-y*np.log(y_pred),axis=1)
        
        return cross_entropy_error
    
    
    def derivative_closs_entropy_error_with_softmax(self, X, y):
        Z = self.forward(X)
        return Z - y
    
    def derivative_tanh(self, Z):
        
        return 1 - Z**2
    
    def derivative_sigmoid(self, Z):
        return Z*(1-Z)
    
    
    def back_propagation(self, X, y):
        grad = {}
        # backpropagation
        #1層目
        dout1 = self.derivative_closs_entropy_error_with_softmax(X, y) # Z3 - y
        grad_b3 = np.average(dout1)
        grad['b3'] = grad_b3
        grad_W3 = np.dot(self.Z2.T, dout1)
        grad['W3'] = grad_W3
        
        #2層目
        dout2 = (1-self.Z2**2)*np.dot(dout1, self.W3.T) # tanhの導関数
        grad_b2 = np.average(dout2)
        grad['b2'] = grad_b2
        grad_W2 = np.dot(self.Z1.T, dout2)
        grad['W2'] = grad_W2
        
        #3層目
        dout3 = (1-self.Z1**2)*np.dot(dout2, self.W2.T)
        grad_b1 = np.average(dout3)
        grad['b1'] = grad_b1
        grad_W1 = np.dot(X.T, dout3)
        grad['W1'] = grad_W1
        return grad

    def fit(self,X, y, X_val=None, y_val=None):
        """
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
        
        #batch_num = len(X_train)/self.batch_size
        #get_mini_batch = GetMiniBatch(X, y, self.batch_size, seed=10)
        
        # 学習
        for i in range(self.epoch):
            print('-----------------')
            print('epoch{}回目の学習'.format(i+1))
            get_mini_batch = GetMiniBatch(X, y, self.batch_size, seed=10)
            #if X_val is not None and y_val is not None:
                #mini_val_loss = 0
            
            # minibatchのイテレーション
            for mini_X_train, mini_y_train in get_mini_batch:
                #y_hat = self.forward(mini_X_train)
                #mini_loss += self.cross_entropy_error(y_hat, mini_y_train)
                
                #if X_val is not None and y_val is not None:
                    #y_val_hat = self.forward(X_val)
                    #Emini_val_loss += self.cross_entropy_error(y_val_hat, y_val)
                    
                # 更新
                grad = self.back_propagation(mini_X_train, mini_y_train)
                self.W1 -= self.lr * grad['W1']/self.batch_size
                self.b1 -= self.lr * grad['b1']/self.batch_size
                self.W2 -= self.lr * grad['W2']/self.batch_size
                self.b2 -= self.lr * grad['b2']/self.batch_size
                self.W3 -= self.lr * grad['W3']/self.batch_size
                self.b3 -= self.lr * grad['b3']/self.batch_size
                
            
            #1epoch毎のtrain_loss 重みは最後のミニバッチ
            y_hat = self.forward(X)
            epoch_loss = np.sum(self.cross_entropy_error(y_hat, y))/len(X)
            
            self.losses.append(epoch_loss)
            print('epoch_loss{}'.format(epoch_loss))
                
            if X_val is not None and y_val is not None:
                y_val_hat = self.forward(X_val)
                #1epoch毎のval_loss 重みは最後のミニバッチ
                epoch_val_loss = np.sum(self.cross_entropy_error(y_val_hat, y_val))/len(X_val) 
                #epoch_val_loss = np.sum(mini_val_loss)/batch_num/len(X_val) # 1epoch毎のval_loss
                self.val_losses.append(epoch_val_loss)
                print('epoch_val_loss{}'.format(epoch_val_loss))
            
    def predict(self, X_test):
        y_pred = self.forward(X_test)
        y_pred = np.argmax(y_pred, axis=1) # 確率の高いものをチョイス
        
        return y_pred
    
    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred==y_test)/len(y_test)
        
        return accuracy
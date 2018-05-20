def find_best_split(feature_vector, target_vector):
    df = pd.DataFrame(columns=["feature", "target"])
    df['feature'] = feature_vector
    df['target'] = target_vector
    setf = np.sort(df['feature'].unique())
    setf = setf[:len(setf)-1]
    size = df.shape[0]
    an = pd.DataFrame(columns=["thresholds", "ginis"])
    for i in setf:
        left = df[df['feature'] <= i]
        left_size = left.shape[0]
        right = df[df['feature'] > i]
        right_size = right.shape[0]
        col_left = 1 - (left[left['target'] == 0].shape[0]/left_size)**2 - (left[left['target'] == 1].shape[0]/left_size)**2
        col_right = 1 - (right[right['target'] == 0].shape[0]/right_size)**2 - (right[right['target'] == 1].shape[0]/right_size)**2
        gini = -(left_size/size) * col_left - (right_size/size) * col_right
        el = pd.DataFrame([[i,gini]], columns=["thresholds", "ginis"])
        an = an.append(el, ignore_index = True)
    an_best = an[an['ginis'] == an['ginis'].max()]

    return an

def find_best_split_w(feature_vector, target_vector, w):
    df = pd.DataFrame(columns=["feature", "target", 'w'])
    df['feature'] = feature_vector
    df['target'] = target_vector
    df['w'] = w
    setf = np.sort(df['feature'].unique())
    setf = setf[:len(setf)-1]
    size = df.shape[0]
    an = pd.DataFrame(columns=["thresholds", "ginis"])
    for i in setf:
        left = df[df['feature'] <= i]
        left_size = left.shape[0]
        right = df[df['feature'] > i]
        right_size = right.shape[0]
        col_left = (left['w']*(left['target']-left['target'].mean())**2).mean()
        col_right = (right['w']*(right['target']-right['target'].mean())**2).mean()
        gini = -(left_size/size) * col_left - (right_size/size) * col_right
        el = pd.DataFrame([[i,gini]], columns=["thresholds", "ginis"])
        an = an.append(el, ignore_index = True)
    an_best = an[an['ginis'] == an['ginis'].max()]

    return an

class DecisionTree(BaseEstimator):
    def __init__(self, md=100000, mss=1, msl=1):
        self._tree = []
        self.max_depth = md
        self.min_samples_split = mss
        self.min_samples_leaf = msl
        self.depth = 0
        self.counter = 0

    def fit_node(self, subX, suby, node):
        subx = subX.copy()
        subY = suby.copy()
        if (self.depth == self.max_depth+1) or (subx.shape[0]<self.min_samples_split) or (len(subY.unique()) == 1) or (self.min_samples_leaf*2 >subx.shape[0]):
            d = subY.value_counts(normalize=True).to_dict()
            print('leaf')
            try:
                node.append(d[1])
            except:
                node.append(0)
            return node

        all_thresholds = pd.DataFrame(columns=["thresholds", "ginis", "f"])
        for k,i in enumerate(subx.columns):
            sp = find_best_split(subx[i], subY)
            sp['f'] = k
            all_thresholds = all_thresholds.append(sp)
            del sp
        all_thresholds = all_thresholds.astype('float').reset_index()

        if all_thresholds.shape[0] == 0:
            print('leaf')
            d = subY.value_counts(normalize=True).to_dict()
            try:
                node.append(d[1])
            except:
                node.append(0)
            return node

        while(True):
            best = all_thresholds[all_thresholds.ginis == all_thresholds.ginis.max()]
            set_l = subx[subx[subx.columns[int(best.f.values[0])]] <= best.thresholds.values[0]]
            set_r = subx[subx[subx.columns[int(best.f.values[0])]] > best.thresholds.values[0]]
            if (set_l.shape[0] < self.min_samples_leaf) or (set_r.shape[0] < self.min_samples_leaf):
                all_thresholds = all_thresholds.drop(best.index[0])
                continue
            break

        y_l = subY.loc[set_l.index]
        y_r = subY.loc[set_r.index]
        self.depth += 1
        node.append(int(best.f.values[0]))
        node.append(best.thresholds.values[0])
        node_left = self.fit_node(set_l, y_l, w_l, [])
        self.depth = d
        node_right = self.fit_node(set_r, y_r, w_r, [])
        self.depth = d
        node.append(node_left)
        node.append(node_right)
        return node

    def fit(self, X, y):
        self.depth += 1
        self.fit_node(X, y, self._tree)

    def predict(self, X):
        def predict_node(x, node, col):
            if len(node) == 1:
                return node[0]
            if x[col[node[0]]] <= node[1]:
                return predict_node(x, node[2], col)
            if x[col[node[0]]] > node[1]:
                return predict_node(x, node[3], col)
        predicted = []
        for x in X.index:
            predicted.append(predict_node(X.loc[x,:], self._tree, X.columns))
        return np.array(predicted)

    def to_dot(self, col):
        def node_f(node, col,A):
            self.counter += 1
            if len(node)==1:
                A.add_node(self.counter, label='var:'+str(round(node[0],4)))
                return None
            A.add_node(self.counter,label=col[node[0]]+' <= '+str(node[1]))
            num = self.counter
            A.add_edge(num,self.counter+1)
            node_f(node[2],col,A)
            A.add_edge(num,self.counter+1)
            node_f(node[3],col,A)
            return None
        A=pgv.AGraph()
        A.add_node(self.counter, label=col[self._tree[0]] +' <= '+str(self._tree[1]))
        num = self.counter
        A.add_edge(num,self.counter+1)
        node_f(self._tree[2],col,A)
        A.add_edge(num,self.counter+1)
        node_f(self._tree[3],col,A)
        A.node_attr['shape']='box'
        return A
class DecisionTreeReg(BaseEstimator):
    def __init__(self, md=100000, mss=1, msl=1):
        self._tree = []
        self.max_depth = md
        self.min_samples_split = mss
        self.min_samples_leaf = msl
        self.depth = 0
        self.counter = 0

    def fit_node(self, subX, suby, w, node):
        subx = subX.copy()
        subY = suby.copy()
        w = w.copy()
        if (self.depth == self.max_depth+1) or (subx.shape[0]<self.min_samples_split) or (self.min_samples_leaf*2 >subx.shape[0]):
            node.append(subY.mean())
            return node

        all_thresholds = pd.DataFrame(columns=["thresholds", "ginis", "f"])
        for k,i in enumerate(subx.columns):
            sp = find_best_split_w(subx[i], subY, w)
            sp['f'] = k
            all_thresholds = all_thresholds.append(sp)
            del sp
        all_thresholds = all_thresholds.astype('float').reset_index()

        if all_thresholds.shape[0] == 0:
            node.append(subY.mean())
            return node

        while(True):
            best = all_thresholds[all_thresholds.ginis == all_thresholds.ginis.max()]
            set_l = subx[subx[subx.columns[int(best.f.values[0])]] <= best.thresholds.values[0]]
            set_r = subx[subx[subx.columns[int(best.f.values[0])]] > best.thresholds.values[0]]
            if (set_l.shape[0] < self.min_samples_leaf) or (set_r.shape[0] < self.min_samples_leaf):
                all_thresholds = all_thresholds.drop(best.index[0])
                continue
            break

        y_l = subY.loc[set_l.index]
        y_r = subY.loc[set_r.index]
        w_l = w.loc[set_l.index]
        w_r = w.loc[set_r.index]
        self.depth += 1
        d = self.depth
        node.append(int(best.f.values[0]))
        node.append(best.thresholds.values[0])
        node_left = self.fit_node(set_l, y_l, w_l, [])
        self.depth = d
        node_right = self.fit_node(set_r, y_r, w_r, [])
        self.depth = d
        node.append(node_left)
        node.append(node_right)
        return node

    def fit(self, X, y, w):
        self.depth += 1
        self.fit_node(X, y, w, self._tree)

    def predict(self, X):
        def predict_node(x, node, col):
            if len(node) == 1:
                return node[0]
            if x[col[node[0]]] <= node[1]:
                return predict_node(x, node[2], col)
            if x[col[node[0]]] > node[1]:
                return predict_node(x, node[3], col)
        predicted = []
        for x in X.index:
            predicted.append(predict_node(X.loc[x,:], self._tree, X.columns))
        return np.array(predicted)

    def to_dot(self, col, var):
        def node_f(node, col,A, var):
            self.counter += 1
            if len(node)==1:
                r = np.where(var >= round(node[0],4))[0][0]+1
                A.add_node(self.counter, label='var:'+str(r))
                return None

            A.add_node(self.counter,label=col[node[0]]+' <= '+str(node[1]))
            num = self.counter
            A.add_edge(num,self.counter+1)
            node_f(node[2],col,A,var)
            A.add_edge(num,self.counter+1)
            node_f(node[3],col,A,var)
            return None
        A=pgv.AGraph()
        A.add_node(self.counter, label=col[self._tree[0]] +' <= '+str(self._tree[1]))
        num = self.counter
        A.add_edge(num,self.counter+1)
        node_f(self._tree[2],col,A,var)
        A.add_edge(num,self.counter+1)
        node_f(self._tree[3],col,A,var)
        A.node_attr['shape']='box'
        return A

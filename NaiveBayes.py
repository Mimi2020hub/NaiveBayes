import numpy as np
import sys

input_train_data = sys.argv[1]
input_test_data = sys.argv[2]

df_c = 0.000000001

# Store labels and attributes 
def key_val_split(data_to_split, idx_max): 
    keys = []
    vals = []    
    with open(data_to_split) as f:
        lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        keys.append(int(words[0]))
        arr = np.zeros(idx_max, dtype=int)
        for word in words[1:]:
            _ws = word.split(':')
            idx = int(_ws[0])
            val = int(_ws[1])
            arr[idx-1] = val
        vals.append(arr)
    return keys, vals
    


# Read train data, test data into labels and attributes lists
def data_get(train_data, test_data): 
    
    train_keys = []
    train_vals = []
    test_keys = []
    test_vals = []
    idx_max = -1
    
    with open(train_data) as f:
        lines = f.readlines()
    for line in lines:
        words = line.strip().split()
        words = words[-1].split(':')
        words.insert(0,'0')
        idx = int(words[-2])
        if idx_max < idx:
            idx_max = idx
        
    train_keys, train_vals = key_val_split(train_data, idx_max)   
    test_keys, test_vals = key_val_split(test_data, idx_max)   
    
    return train_keys, train_vals, test_keys, test_vals

# Compute probability of x given y 
def train_model_px(training_vals):
    N,D = np.shape(training_vals)  # row column
    training_vals = np.mat(training_vals) # list of array to matrix
    model = {} # level 3, for attributes
    for d in range(D):
        data = training_vals[:,d].tolist() # column to list
        data = list(np.ravel(data)) # list of matrix to list of integer
        keys = set(data) # values
        model[d] ={}  # level 4, for values
        for key in keys: # each value
            model[d][key] = float(data.count(key)/N) # probability for each value
    return model

# Naive Bayes training model to get the probabilites needed for prediction
def train_model(vals,labs):
    model = {} # level 1, for labels
    keys = set(labs)
    n_y = len(labs)
    for key in keys:
        model[key]={} # level 2, for Pro(y), Pro(x/y)
        
        model_py = labs.count(key)/n_y # Pro(y)
        
        index = np.where(np.array(labs)==key)[0].tolist()
        feats = np.array(vals)[index]
        model_px_y = train_model_px(feats)
        
        model[key]["PY"] = model_py
        model[key]["PX"] = model_px_y
        
    return model

# Predict labels with probability of y, and probability of x given y    
def llk_y(feat, keys, model, df_c):
    results = {}
    for key in keys:
        model_py = model.get(key).get("PY")
        model_px = model.get(key).get("PX")
        list_px = []
        for d in range(len(feat)):
            get_px = model_px.get(d,df_c).get(feat[d],df_c)+df_c
            list_px.append(get_px)
        # Use a log function to avoid underflow
        result = np.log(model_py) + np.sum(np.log(list_px))
        results[key] = result
    llk_df = -10000000
    for k in results:
        if results[k]>llk_df:
            llk_df=results[k]
            best_y=k
    return best_y

train_keys, train_feats, test_keys, test_feats = data_get(input_train_data, input_test_data)
keys = set(train_keys)
model = train_model(train_feats,train_keys)



for i in [[train_feats, train_keys], [test_feats, test_keys]]:
    pred_acc = np.zeros(4, dtype=int)
    pred_truth_clsf = [[1,1],[-1,1],[1,-1],[-1,-1]] # [prediction, true]
    for k in range(len(i[1])): # train_keys or test_keys
        res = llk_y(i[0][k], keys, model, df_c) # dataset[train_feats/test_feats][row]
        pred_acc[pred_truth_clsf.index([res,i[1][k]])]= pred_acc[pred_truth_clsf.index([res,i[1][k]])]+1
    print(*pred_acc)
    



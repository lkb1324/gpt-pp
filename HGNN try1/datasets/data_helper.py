import scipy.io as sio
import numpy as np


def load_ft(data_dir, feature_name):
    data = sio.loadmat(data_dir)  # 读取mat文件，是字典形式存储{key:value}
    lbls = data['Y'].astype(np.long)

    #     idx = data['indices'].item()
    #     print('idx',idx)
    if feature_name == 'CGNN':
        fts = data['X'].astype(np.float32)
    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError

    idx_train = np.zeros(853)
    idx_test = np.zeros(213)
    for i in range(853):
        idx_train[i] = i
    for i in range(213):
        idx_test[i] = i + 853

    return fts, lbls, idx_train, idx_test


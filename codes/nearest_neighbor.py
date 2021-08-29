import numpy as np
import shutil
import os


__all__ = ['search_NN']


def search_NN(test_emb, train_emb_flat, NN=1, method='kdt'):
    '''
    nearest neighbor search以下分成兩種 ngt / kdt:
    
    1. KDT:
        KD Tree (K維樹)是一種可以對K維資料進行劃分的資料結構
        可以看成是BST的延伸，透過不斷對於空間中的維度做劃分，利用prune search的特性縮短time complexity 
        主要可以用在多維空間搜尋 ex. 底下要用的neighbor search
    '''
    if method == 'ngt':
        return search_NN_ngt(test_emb, train_emb_flat, NN=NN)

    from sklearn.neighbors import KDTree
    kdt = KDTree(train_emb_flat) 

    Ntest, I, J, D = test_emb.shape  # (83, 13, 13, 64)
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    for n in range(Ntest): #83
        for i in range(I): #13
            '''
            dists: NN個最接近的neighbor的距離, 也會用他的值來畫anomaly map
            inds: NN個最接近的neighbor
            '''
            dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
            closest_inds[n, i, :, :] = inds[:, :]
            l2_maps[n, i, :, :] = dists[:, :]   # 等於用他的anomaly score當成 map

    return l2_maps, closest_inds


def search_NN_ngt(test_emb, train_emb_flat, NN=1):
    import ngtpy

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    # os.makedirs('tmp', exist_ok=True)
    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds

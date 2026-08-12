import numpy as np
from sklearn.preprocessing import normalize

def si_vq(X,Y,metric):
    n, M = X.shape
    L = Y.shape[1]
    distances = np.empty((n,M-L+1))
    argmins = np.empty_like(distances,dtype=int)
    
    nY = normalize(Y,axis=1).astype('f')
    Ynorm = np.sqrt(np.sum(Y**2,axis=1,keepdims=True))
    for shift in range(M-L+1):
        Xshift = X[:,shift:shift+L].astype('f')
        if metric == 'cosine':
            nX = normalize(Xshift,axis=1)
            XY = nX@nY.T
            argmins[:,shift] = np.argmax(XY,axis=1)
            distances[:,shift] = 1 - XY[np.arange(n),argmins[:,shift]] # distance_cosine(x,y) = 1- sim(x,y)
        else:
            Xnorm = np.sqrt(np.sum(Xshift**2,axis=1,keepdims=True))
            XY = Xshift@Y.T
            all_distances = Xnorm + XY + Ynorm.T
            argmins[:,shift] = np.argmin(all_distances,axis=1)
            distances[:,shift] = all_distances[np.arange(n),argmins[:,shift]] # distance_cosine(x,y) = 1- sim(x,y)
    
    best_shifts = np.argmin(distances,axis=1)
    best_distances = distances[np.arange(n),best_shifts]
    best_argmins = argmins[np.arange(n),best_shifts]
    return best_argmins, best_shifts, best_distances


# Changed sign_invarient code -------------
def si2_vq(X,Y,metric):
    n, M = X.shape
    L = Y.shape[1]
    best_labels_b, best_shifts, best_distances = si_vq(X,np.vstack((Y,-Y)), metric)
    best_signs = np.where(best_labels_b//2 == 0, np.ones_like(best_shifts), np.full_like(best_shifts, -1))
    n_centroids = Y.shape[0]
    best_labels = best_labels_b % n_centroids
    return best_labels, best_shifts, best_distances, best_signs

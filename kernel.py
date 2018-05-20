def kernel_base(x,z,cat_n):
    return (1 - np.sum(np.absolute(z-x), axis=1)/np.sum(cat_n-1))
def kernel_max(x,z,cat_n):
    return (1 - np.max(np.absolute(z-x),axis=1)/np.max(cat_n-1))
def kernel_dist(x,z,cat_n):
    return(1 - np.linalg.norm(z-x,axis=1)/np.linalg.norm(cat_n-1))

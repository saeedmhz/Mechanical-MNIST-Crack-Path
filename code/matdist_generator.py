import numpy as np
import os


def bitmap(res, Xcenters, Ycenters):
    """calculates rigidity ratio at (x,y) in UFL

    Parameters
    ----------
    res: int
        resolution of the uniform grid over the domain
    Xcenters : numpy vector
        inclusions center point locatoin on X axis
    Ycenters : numpy vector
        inclusions center point locatoin on Y axis

    Returns
    -------
    (res,res) numpy array respresenting material distribution of a sample
    """
    X = np.ones((res,1))
    Y = np.ones((1,res))
    pos = np.linspace(1/(res*2), 1 -1/(res*2), res) - 0.5
    X = X*pos
    Y = Y*pos.reshape(res,1)

    dist = (Xcenters-X)**2 + (Y - Ycenters)**2
    
    d_min = (0.25/28)**2
    alpha = -1000
    beta = 0.9
    
    num = np.sum(dist*np.exp(dist*alpha), axis=0)
    denum = np.sum(np.exp(dist*alpha), axis=0)
    d = num/denum
    
    return 1 * np.less(d, d_min) + np.less_equal(d_min, d) * d_min/(beta*d_min + (1-beta)*d)


def main(case_num):
    res = 1024

    with open('./inclusions/input' + str(case_num) + '.npy', 'rb') as file:
        centers = np.loadtxt(file)

    # shrinking offset
    eta = 0.7

    Xcenters = eta*(centers[:,0]/28-0.5).reshape(-1,1,1)
    Ycenters = eta*((28-centers[:,1])/28-0.5).reshape(-1,1,1)

    # saving material distribution
    if not os.path.exists('./material/'):
        os.mkdir('./material/')
    np.savetxt('./material/mat' + str(case_num) + '.npy', bitmap(res, Xcenters, Ycenters))


if __name__ == '__main__':
    case_num = 0
    main(case_num)

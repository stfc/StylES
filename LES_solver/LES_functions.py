import cupy as cp

def load_fields():
    data = cp.load('restart.npz')
    ctotTime = data['t']
    totTime = cp.asnumpy(ctotTime)
    U = data['U']
    V = data['V']
    P = data['P']
    C = data['C']
    B = data['B']

    return U, V, P, C, B, totTime


def save_fields(totTime, U, V, P, C, B):

    cp.savez('restart.npz', t=totTime, U=U, V=V, P=P, C=C, B=B)

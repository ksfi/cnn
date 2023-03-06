import math

def H(H_in, pad, ker, stride): # (.., 1, 3, 1) for conv (.., 1, 2, 2) for pool
    return math.floor(((H_in+2*pad-(ker-1)-1)/stride)+1)

def outH(lay, size):
    out=size
    for l in lay:
        if l == 1:
            out = H(out, 1, 3, 1)
        if l == 0:
            out = H(out, 1, 2, 2)
    return out

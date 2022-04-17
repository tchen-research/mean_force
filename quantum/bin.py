import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output

class get_hamiltonian():

    def __init__(self,Jx,Jy,Jz,s):
        self.N = len(Jz)
        self.s = s
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.dtype = np.float128

        self.M = int(2*self.s+1)
        self.Sx = np.zeros((self.M,self.M),dtype='complex')
        self.Sy = np.zeros((self.M,self.M),dtype='complex')
        self.Sz = np.zeros((self.M,self.M),dtype='complex')
        for i in range(self.M):
            for j in range(self.M):
                self.Sx[i,j] = ((i==j+1)+(i+1==j))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2
                self.Sy[i,j] = ((i+1==j)-(i==j+1))*np.sqrt(s*(s+1)-(s-i)*(s-j))/2j
                self.Sz[i,j] = (i==j)*(s-i)

    def __matmul__(self,v):
                
        if v.ndim == 2:
            m,n = v.shape
        else:
            m = len(v)
            n = 1 
    
        out = np.zeros((m,n),dtype='complex')

        for j in range(self.N):
            if  np.count_nonzero(self.Jx[:,j]) != 0 or np.count_nonzero(self.Jy[:,j]) != 0:
                I1 = self.M**j
                I2 = self.M**(self.N-j-1)
                Sxj_v = ((self.Sx@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                Syj_v = ((self.Sy@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                Szj_v = ((self.Sz@v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T

                # symmetry
                for i in range(j):
                    if self.Jx[i,j] != 0 or self.Jy[i,j] != 0:
                        I1 = self.M**i
                        I2 = self.M**(self.N-i-1)
                        Sxi_Sxj_v = ((self.Sx@Sxj_v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T
                        Syi_Syj_v = ((self.Sy@Syj_v.T.reshape(n,I1,-1,I2))).reshape(n,-1).T

                        out += (2-(i==j))*self.Jx[i,j] * Sxi_Sxj_v
                        out += (2-(i==j))*self.Jy[i,j] * Syi_Syj_v 

                out += self.Jz[j] * Szj_v
            
        return out.flatten() if n==1 else out
    
    def tosparse(self):
                
        out = sp.sparse.coo_matrix((self.M**self.N,self.M**self.N),dtype='complex')

        for j in range(self.N):
            if  np.count_nonzero(self.Jx[:,j]) != 0 or np.count_nonzero(self.Jy[:,j]) != 0:
                I1 = sp.sparse.eye(self.M**j,dtype='complex')
                I2 = sp.sparse.eye(self.M**(self.N-j-1),dtype='complex')
                Sxj = sp.sparse.kron(sp.sparse.kron(I1,self.Sx),I2)
                Syj = sp.sparse.kron(sp.sparse.kron(I1,self.Sy),I2)
                Szj = sp.sparse.kron(sp.sparse.kron(I1,self.Sz),I2)

                for i in range(j):
                    if self.Jx[i,j] != 0 or self.Jy[i,j] != 0:
                        I1 = sp.sparse.eye(self.M**i,dtype='complex')
                        I2 = sp.sparse.eye(self.M**(self.N-i-1),dtype='complex')
                        Sxi_Sxj = sp.sparse.kron(sp.sparse.kron(I1,self.Sx),I2)@Sxj
                        Syi_Syj = sp.sparse.kron(sp.sparse.kron(I1,self.Sy),I2)@Syj
                        #Szi_Szj = sp.sparse.kron(sp.sparse.kron(I1,self.Sz),I2)@Szj

                        out += (2-(i==j))*self.Jx[i,j] * Sxi_Sxj
                        out += (2-(i==j))*self.Jy[i,j] * Syi_Syj

                out += self.Jz[j] * Szj
            
        return out.tocsr()
    
def block_lanczos(H,V,k,reorth=False,returnQ=False):
    
    assert np.sum(np.abs(np.diag(V.T@V) - 1)>1e-8) == 0, 'V must be orthonormal'
    
    n,s = V.shape
    Q = np.copy(V) #+ orth
    
    A = [ np.zeros((s,s)) ]*k
    B = [ np.zeros((s,s)) ]*k
    
    if reorth:
        returnQ = True
    
    if returnQ:
        Q_full = np.zeros((n,s*k),dtype=H.dtype)
        Q_full[:,:s] = Q
    else:
        Q_full = None
        
    for i in range(0,k):
        Q__ = np.copy(Q)
        Z = H@Q - Q_@(B[i-1].conj().T) if i>0 else H@Q
        Q_ = Q__

        A[i] = Q_.conj().T@Z
        Z -= Q_@A[i]
        
        if reorth:
            Z -= Q_full@(Q_full.conj().T@Z)

        Q,B[i] = np.linalg.qr(Z)
        
        if i < k-1 and returnQ:
            Q_full[:,(i+1)*s:(i+2)*s] = Q
    
    return Q_full,Q,A,B

def par_block_lanczos(H,Vs,k):
    
    for V in Vs:
        assert np.sum(np.abs(np.diag(V.T@V) - 1)>1e-8) == 0, 'V must be orthonormal'
    
    r = len(Vs)
    n,s = Vs[0].shape
    
    Q = np.zeros((n,s,r),dtype='complex')
    Q_ = np.zeros((n,s,r),dtype='complex')
    Q__ = np.zeros((n,s,r),dtype='complex')
    for j in range(r):
        Q[:,:,j] = Vs[j]
    
    A = [[ np.zeros((s,s)) ]*k for j in range(r)]
    B = [[ np.zeros((s,s)) ]*k for j in range(r)]
        
    for i in range(0,k):
        
        HQ = (H@Q[:,:].reshape(n,-1)).reshape(n,s,r)

        for j in range(r):
            Q__[:,:,j] = np.copy(Q[:,:,j])

            Z = HQ[:,:,j] - Q_[:,:,j]@(B[j][i-1].conj().T) if i>0 else HQ[:,:,j]
            Q_[:,:,j] = Q__[:,:,j]

            A[j][i] = Q_[:,:,j].conj().T@Z
            Z -= Q_[:,:,j]@A[j][i]

            Q[:,:,j],B[j][i] = np.linalg.qr(Z)
        
    return None,Q,A,B


def get_block_tridiag(M,R):

    k = len(M)
    s = len(M[0])
    
    T = np.zeros((k*s,k*s),dtype=M[0].dtype)

    for i in range(k):
        T[i*s:(i+1)*s,i*s:(i+1)*s] = M[i]

    for i in range(k-1):
        T[(i+1)*s:(i+2)*s,i*s:(i+1)*s] = R[i]
        T[i*s:(i+1)*s,(i+1)*s:(i+2)*s] = R[i].conj().T
        
    return T

def get_partial_traces(H_T,H_I,H_B_iso,βs,k,n_ave,M,N,N_S,N_B,E0=0,s_print=0):
    

    trB_expH_T = [np.zeros((M**N_S,M**N_S),dtype='complex') for l in enumerate(βs)]
    tr_expH_B = np.zeros(len(βs),dtype='complex')
    trB_H_I = np.zeros((M**N_S,M**N_S),dtype='complex')

    # average over samples
    for j in range(n_ave):
        
        if s_print>0:
            if j%s_print==0:
                print(f'nave: {j}', end="\r")

        # sample test vector
        v = np.random.randn(M**N_B)
        v /= np.linalg.norm(v)
        V = np.kron(np.eye(M**N_S),v.reshape(-1,1))

        # this estimator is incredibly slow because H_I has trace zero I think..
        trB_H_I += V.conj().T@(H_I@V)

        Q,q,MM,RR = block_lanczos(H_T,V,k)        
        T = get_block_tridiag(MM,RR)
        θ,S = sp.linalg.eigh(T)
        S0 = S[:M**N_S]
        for l,β in enumerate(βs):
            trB_expH_T[l] += (S0*np.exp(-β*(θ-E0)))@S0.conj().T

        # need this for H*
        Q,q,MM,RR = block_lanczos(H_B_iso,v[:,None],k)        
        T = get_block_tridiag(MM,RR)
        θ,S = sp.linalg.eigh(T)

        for l,β in enumerate(βs):
            tr_expH_B[l] += (S[0]*np.exp(-β*(θ-E0)))@S.conj().T[:,0]

    trB_H_I *= M**N_B/n_ave
    for l,β in enumerate(βs):

        trB_expH_T[l] = (trB_expH_T[l] + trB_expH_T[l].conj().T)/2 #probably not necessary, but it's not clear whether the ij or ji entries would be more accurate
        trB_expH_T[l] *= M**N_B/n_ave

        tr_expH_B[l] *= M**N_B/n_ave
        
    return trB_expH_T,tr_expH_B,trB_H_I

def get_connection_matrix(J_T,Jz_T,N_S):
    
    N = len(Jz_T)
    
    J_S = np.zeros((N,N))
    J_I = np.zeros((N,N))
    J_B = np.zeros((N,N))

    J_B[N_S:,N_S:] = J_T[N_S:,N_S:]
    J_S[:N_S,:N_S] = J_T[:N_S,:N_S]
    J_I = J_T-(J_S+J_B)

    Jz_S = np.zeros(N)
    Jz_I = np.zeros(N)
    Jz_B = np.zeros(N)

    Jz_B[N_S:] = Jz_T[N_S:]
    Jz_S[:N_S] = Jz_T[:N_S]
    Jz_I = Jz_T-(Jz_S+Jz_B)

    return J_B,J_S,J_I,Jz_B,Jz_S,Jz_I

def get_hamiltonians(J_T,Jz_T,N_S,s,method='explicit'):
    
    # explicit is faster but requires more storage and more upfront cost
    N = len(Jz_T)
    N_B = N - N_S
    
    J_B,J_S,J_I,Jz_B,Jz_S,Jz_I = get_connection_matrix(J_T,Jz_T,N_S)
    
    if method=='implicit':
        H_T = get_hamiltonian(J_T,Jz_T,s)
        H_S = get_hamiltonian(J_S,Jz_S,s)
        H_B = get_hamiltonian(J_B,Jz_B,s)
        H_I = get_hamiltonian(J_I,Jz_I,s)

        H_S_iso = get_hamiltonian(J_S[:N_S,:N_S],J_S[:N_S,:N_S],Jz_S[:N_S],s)
        H_B_iso = get_hamiltonian(J_B[N_S:,N_S:],J_B[N_S:,N_S:],Jz_B[N_S:],s)

    elif method=='explicit':
        H_S_iso = get_hamiltonian(J_S[:N_S,:N_S],J_S[:N_S,:N_S],Jz_S[:N_S],s).tosparse()
        H_B_iso = get_hamiltonian(J_B[N_S:,N_S:],J_B[N_S:,N_S:],Jz_B[N_S:],s).tosparse()
        H_I = get_hamiltonian(J_I,J_I,Jz_I,s).tosparse()

        H_S = sp.sparse.kron(H_S_iso,sp.sparse.eye((2*s+1)**N_B))
        H_B = sp.sparse.kron(sp.sparse.eye((2*s+1)**N_S),H_B_iso)
        H_T = H_S + H_B + H_I
        
    return H_T,H_S,H_B,H_I,H_S_iso,H_B_iso

def get_partial_trace(H,M,N,N_S,N_B):
    T = np.zeros((M**N_S,M**N_S))
    for m in range(M**N_S):
        for n in range(M**N_S):
            T[m,n] = np.trace(H[m*M**N_B:(m+1)*M**N_B,n*M**N_B:(n+1)*M**N_B])
    return T
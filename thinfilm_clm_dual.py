import numpy as np
from scipy.sparse import lil_matrix, csc_matrix,coo_matrix,csr_matrix,hstack,vstack
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

SL = 0.0
SR = 0.0
g1 = 1.0
g2 = 2.0

mobexp = 2.0 # mobility exponent n in m(h)=h^n
L = 2.0 #initial domain size (0,L)

ppower =    2.0 #power 2/alpha in dynamic contact angle law
mu0    =    1.0 #prefactor in dynamic contact angle law
nt     =  5000 #number of time steps
npoint =    512 #number of vertices
ndof   = npoint
#create element decomposition for FE method

nelement        = npoint-1
local_mass_p1   = np.array([[1/3,1/6],[1/6,1/3]]); 


dt=1e-3

xii=(np.array([(-1/3)*np.sqrt(5+2*np.sqrt(10/7)),(-1/3)*np.sqrt(5-2*np.sqrt(10/7)),0, (1/3)*np.sqrt(5-2*np.sqrt(10/7)),(1/3)*np.sqrt(5+2*np.sqrt(10/7))])+1)/2

wii=np.array([(322-13*np.sqrt(70))/900,(322+13*np.sqrt(70))/900,128/225,(322+13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900])/2

nd=[]
for i in range(nelement):
    nd.append([i,i+1])
nd=np.array(nd)

def integration_weights_dual():
    ii=np.zeros((nelement,4),dtype=int)
    jj=np.zeros((nelement,4),dtype=int)
    for i in range(nelement):
        ii[i,0]=nd[i,0]
        ii[i,1]=nd[i,1]
        ii[i,2]=nd[i,0]
        ii[i,3]=nd[i,1]
        
        jj[i,0]=nd[i,0]
        jj[i,1]=nd[i,0]
        jj[i,2]=nd[i,1]
        jj[i,3]=nd[i,1]
    return ii,jj
        
        

# Function to build FEM matrices
def build_FE_matrices_dual(x, h,ii,jj):

    S = lil_matrix((npoint, npoint))
    Sw = lil_matrix((npoint, npoint))
    M = lil_matrix((npoint, npoint))
    Dx = lil_matrix((npoint, npoint))
    Z = lil_matrix((npoint, npoint))
    
    edet = x[nd[:, 1]] - x[nd[:, 0]]

    
    mobi = np.zeros(nelement)
    
    for i in range(len(xii)):
        hval=(1-xii[i])*h[nd[:,0]]+xii[i]*h[nd[:,1]]
        mobi+=wii[i]*(np.abs(hval))**mobexp
    
    dphiref=np.array([-1,1])
    
    S_=np.zeros((nelement,2,2))
    Ms_=np.zeros((nelement,2,2))
    Dx_=np.zeros((nelement,2,2))
    Sw_=np.zeros((nelement,2,2))
    
    for i in range(2):
        for j in range(2):
            S_[:,i,j]+=dphiref[i]*dphiref[j]/edet
            Sw_[:,i,j]+=dphiref[i]*dphiref[j]*mobi/edet
            Ms_[:,i,j]+=local_mass_p1[i,j]*edet
            Dx_[:,i,j]+=dphiref[j]/2
    
    
    ii=ii.ravel(order="F")
    jj=jj.ravel(order="F")
    
    S=coo_matrix((S_.ravel(order="F"),(ii,jj)))
    Sw=coo_matrix((Sw_.ravel(order="F"),(ii,jj)))
    M=coo_matrix((Ms_.ravel(order="F"),(ii,jj)))
    Dx=coo_matrix((Dx_.ravel(order="F"),(ii,jj)))
    
    
    
    A1=np.hstack([M,Sw])
    A2=np.hstack([-dt*S,M])
    A=np.vstack([A1,A2])
    
    
    return csc_matrix(S),csc_matrix(M),csc_matrix(Sw),csc_matrix(Dx)
        

    


# Function to build ALE dual matrices
def build_ALE_matrix_dual(x, h, M, Dx):
    
    P = lil_matrix((ndof, ndof))
    I = lil_matrix((ndof, ndof))
    X = lil_matrix((ndof, ndof))
    
    dh=spsolve(M, Dx @ h)
    xi=(x-x[0])/(x[-1]-x[0])
    
    for i in range(ndof):
        P[i,0]=(1-xi[i])*dh[i]
        X[i,0]=(1-xi[i])
    for i in range(ndof):
        P[i,npoint-1]=xi[i]*dh[i]
        X[i,npoint-1]=xi[i]
    for i in range(1,ndof-1):
        I[i,i]=1
    
    return csc_matrix(P), csc_matrix(I), csc_matrix(X),dh


def solve_thinfilm():
    x = np.linspace(0,L,npoint)
    h=(1-(x-1)**2)
    t=0
    ii,jj=integration_weights_dual()
    for i in range(nt):
        S,M,Sw,Dx=build_FE_matrices_dual(x,h,ii,jj)
        P,I,X,dh=build_ALE_matrix_dual(x,h,M,Dx)
        
        Mbd=lil_matrix((2,ndof))
        Mbd[0,0]=1
        Mbd[1,ndof-1]=1
        Mbd=csc_matrix(Mbd)
        
        ZZ=lil_matrix((2,ndof))
        ZZ1=lil_matrix((2,2))
        Sbd=mu0*np.array([[abs(dh[0])**ppower,0],[0,abs(dh[-1])**ppower]])
        
        rhs=np.append((S @ h + M @ (2*g2*h-g1*x)),np.zeros(ndof+2))
        rhs[0]+= (SL+(dh[0]**2)/2)/(dh[0])
        rhs[ndof-1]-=(SR+(dh[-1]**2)/2)/(dh[-1])
        
        row1 = hstack([(-dt * S), M, Mbd.T])
        row2 = hstack([M, Sw, ZZ.T])
        row3 = hstack([Mbd, ZZ, Sbd])
        
        A = vstack([row1, row2, row3])
        A=csc_matrix(A)
        
        u=spsolve(A,rhs)
        
        U=spsolve((I-P),u[:ndof])
        h+=dt * (I @ U)
        x+=dt * (X @ U)
        t+=dt
        
        if (i%100==1):
            plt.plot(x,h)
    plt.xlim(-20,20)
    plt.ylim(0,2)
    plt.savefig("b.png")
    
if __name__=="__main__":
    solve_thinfilm()

import time
import numpy as np
from scipy import optimize
np.set_printoptions(10,suppress=True,sign="+",floatmode="fixed")

NSPIN = 2
DIM = 3
spinError = ValueError('spin must be 1 or -1')

eta1 = -1*np.array([1,np.sqrt(3)])/2
eta2 =  np.array([1,0])
eta3 =  np.array([-1,np.sqrt(3)])/2

J=1
D=0.125
B=0
T=1
res = 480
resX = res+1

initLamda = +3.4252088421
initChiUp = -0.6091298075
initChiDn = -0.5930922894

resY = resX # previously squarer: int( resX * 2/np.sqrt(3) )
Nk = resX*resY
KX= np.linspace( -np.pi/3.         , 2*np.pi/3       , resX ) # the hexagonal BZ is periodic
KY= np.linspace( -np.pi/np.sqrt(3) , np.pi/np.sqrt(3) , resY ) # so one can use a rectangle.
unitVolume = np.pi * 2*np.pi/np.sqrt(3) / Nk


def hopping(sigma,chiSig,chiOpp):
    return J*(chiSig+chiOpp) - 1j*sigma*D*chiOpp

def hamiltonian(kx,ky,sigma,lamda,hop): # lambda with "b" is protected
    k = np.array([kx,ky])
    cosk1 = np.cos(np.dot(k,eta1))
    cosk2 = np.cos(np.dot(k,eta2))
    cosk3 = np.cos(np.dot(k,eta3))
    h = np.matrix([
        [ (lamda-sigma*B)/2 ,     hop*cosk1     ,        0          ],
        [         0         , (lamda-sigma*B)/2 ,    hop*cosk2      ],
        [     hop*cosk3     ,         0         , (lamda-sigma*B)/2 ]
    ])
    return h + h.H  # conjuguate transpose hence /2 on the diagonal

def eighOnMesh(sigma,lamda,hop):
    tensorShape = (len(KX),len(KY),DIM,DIM)
    hamiltonianOnMesh = np.empty(tensorShape,dtype=complex)
    for i,kx in enumerate(KX):
        for j,ky in enumerate(KY):
            hamiltonianOnMesh[i,j,:,:] = hamiltonian(kx,ky,sigma,lamda,hop)
    # print(hamiltonianOnMesh[10,10,:,:])
    return np.linalg.eigh(hamiltonianOnMesh)

def bose(energ,beta=1/T):
    return 1.0/(np.exp(beta*energ)-1)

def dummyIntegral(qtyOnMesh):
    return np.sum(qtyOnMesh) / Nk




solLamda = initLamda
solChiUp = initChiUp
solChiDn = initChiDn


tUp = hopping( 1,solChiUp,solChiDn)
tDn = hopping(-1,solChiDn,solChiUp)
bandsUp,eigVecsUp = eighOnMesh( 1,solLamda,tUp)
bandsDn,eigVecsDn = eighOnMesh(-1,solLamda,tDn)
fnkUp = (bandsUp-(solLamda-B))
fnkDn = (bandsDn-(solLamda+B))


t0 = time.clock()
S = dummyIntegral( bose(bandsUp) + bose(bandsDn) )
resChiUp = 2*dummyIntegral( fnkUp * bose(bandsUp) )/(np.abs(tUp))
print(time.clock()-t0)
print(S)
print(resChiUp)



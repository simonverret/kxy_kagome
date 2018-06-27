import numpy as np
from scipy import optimize
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NSPIN = 2
DIM = 3
spinError = ValueError('spin must be 1 or -1')

eta1 = -1*np.array([1,np.sqrt(3)])/2
eta2 =  np.array([1,0])
eta3 =  np.array([-1,np.sqrt(3)])/2

J=1
D=0.2
B=0.1
T=0.2
resX = 33

initLamda =  4.014814
initChiUp = -1.804222
initChiDn = -0.153052

resY = resX # previously squarer: int( resX * 2/np.sqrt(3) )
Nk = resX*resY
KX= np.linspace( -np.pi/3.         , 2*np.pi/3.       , resX ) # the hexagonal BZ is periodic
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
        [ (lamda-sigma*B)/2 ,     hop*cosk1    ,        0         ],
        [        0         , (lamda-sigma*B)/2 ,    hop*cosk2     ],
        [    hop*cosk3     ,         0        , (lamda-sigma*B)/2 ]
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

def selfConsistCond(varSet):
    selfConsistCond.counter += 1
    lamda = varSet[0]
    chiUp = varSet[1]
    chiDn = varSet[2]

    tUp = hopping( 1,chiUp,chiDn)
    tDn = hopping(-1,chiDn,chiUp)
    bandsUp = eighOnMesh( 1,lamda,tUp)[0]
    bandsDn = eighOnMesh(-1,lamda,tDn)[0]

    # Doki et al., Universal..., 2018, sup. mat. Eq.(S6a)
    S = dummyIntegral( bose(bandsUp) + bose(bandsDn) )
    # Doki et al., Universal..., 2018, sup. mat. Eq.(S6b)
    fnkUp = (bandsUp-(lamda-B)) # =the eigenvalues of the hopping matrix
    fnkDn = (bandsDn-(lamda+B))
    resChiUp = 2*dummyIntegral( fnkUp * bose(bandsUp) )/(np.abs(tUp))
    resChiDn = 2*dummyIntegral( fnkDn * bose(bandsDn) )/(np.abs(tDn))

    conditionlamda = 2*S-1
    conditionChiUp = resChiUp-chiUp
    conditionChiDn = resChiDn-chiDn

    print("call ",selfConsistCond.counter)
    print("vars in  :",np.array([lamda,chiUp,chiDn]))
    print("vars out :",np.array([S,resChiUp,resChiDn]))
    print("residual :",np.array([conditionlamda,conditionChiUp,conditionChiDn]))

    return np.array([conditionlamda,conditionChiUp,conditionChiDn])
selfConsistCond.counter =0

#### UNCOMMENT FOR SINGLE INIT RUN and exit
# solLamda = initLamda
# solChiUp = initChiUp
# solChiDn = initChiDn
# selfConsistCond([initLamda,initChiUp,initChiDn])
# exit(1)

sol_object = optimize.root(selfConsistCond, np.array([initLamda,initChiUp,initChiDn]))
print("\nSolution reached in "+str(selfConsistCond.counter)+" call:")
print(sol_object.x)
solLamda = sol_object.x[0]
solChiUp = sol_object.x[1]
solChiDn = sol_object.x[2]




#######################
#### VARIOUS PLOTS ####
#######################

tUp = hopping( 1,solChiUp,solChiDn)
tDn = hopping(-1,solChiDn,solChiUp)
bandsUp,eigVecsUp = eighOnMesh( 1,solLamda,tUp)
bandsDn,eigVecsDn = eighOnMesh(-1,solLamda,tDn)

#### BANDS FOR SPIN UP AND DOWN
X,Y = np.meshgrid(KY,KX)
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(DIM):
    ax.plot_surface(X, Y, bandsUp[:,:,i])
    ax.plot_surface(X, Y, bandsDn[:,:,i])
plt.show()

# #### PLOT FOR LAMBDA
# res1d = 51
# X=np.linspace(solLamda-2.5,solLamda+2.5, res1d)
# Y=np.zeros(res1d)
# for i,x in enumerate(X):
#     Y[i] = selfConsistCond([x,1,1])[0]
# fig = plt.figure()
# plt.ylim(-3,3)
# plt.plot(X,Y)
# plt.show()

# #### PLOT FOR RESIDUALS vs CHIup and CHIdn
# res = 15
# X = np.linspace(solChiUp-0.5,solChiUp+0.5, res)
# Y = np.linspace(solChiDn-0.5,solChiDn+0.5, res)
# Z=[np.zeros((res,res)),np.zeros((res,res)),np.zeros((res,res))]
# for i,x in enumerate(X):
#     for j,y in enumerate(Y):
#         Z[0][i,j],Z[1][i,j],Z[2][i,j] = selfConsistCond([solLamda,x,y])
# X,Y = np.meshgrid(X, Y)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(2):
#     ax.plot_surface(X, Y, Z[i])
# plt.show()



# go see : https://stackoverflow.com/questions/41443444/numpy-element-wise-dot-product
# for element wise tensor protduct
# https://sites.google.com/a/ucsc.edu/krumholz/teaching-and-courses/ast119_w15/class-7
# for multivariate optimize
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# to know how to add bounds for some and None for others
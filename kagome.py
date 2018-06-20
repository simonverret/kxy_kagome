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
B=0.01
T=1
resX = 33
initLamda =  2.59
initChiUp = -0.57
initChiDn = -0.58

resY = resX # previously squarer: int( resX * 2/np.sqrt(3) )
Nk = resX*resY 
KX= np.linspace( -np.pi/3.         , 2*np.pi/3.       , resX ) # the hexagonal BZ is periodic
KY= np.linspace( -np.pi/np.sqrt(3) , np.pi/np.sqrt(3) , resY ) # so one can use a rectangle.
unitVolume = np.pi * 2*np.pi/np.sqrt(3) / Nk
print("dk volume is : "+str(unitVolume))

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
    print(hamiltonianOnMesh[10,10,:,:])
    return np.linalg.eigh(hamiltonianOnMesh)

def bose(energ,beta=1/T):
    return 1.0/(np.exp(beta*energ)-1)

def dummyIntegral(qtyOnMesh):
    return np.sum(qtyOnMesh) / Nk

def selfConsistCond(varSet):
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
    resChiUp = dummyIntegral( fnkUp * bose(bandsUp) )/(2*np.abs(tUp))
    resChiDn = dummyIntegral( fnkDn * bose(bandsDn) )/(2*np.abs(tDn))

    conditionlamda = 2*S-1
    conditionChiUp = resChiUp-chiUp
    conditionChiDn = resChiDn-chiDn


    print("hopping up   : "+str(tUp))
    print("hopping down : "+str(tDn))
    print("bands :\n",bandsUp[10,10])
    print("fnkup :\n",fnkUp[10,10])
    print("fnkdn :\n",fnkDn[10,10])


    print("lambda = "+"{:<8}".format(str(lamda))+" -->      S = "+str(S))
    print("chi_up = "+"{:<8}".format(str(chiUp))+" --> chi_up = "+str(resChiUp))
    print("chi_dn = "+"{:<8}".format(str(chiDn))+" --> chi_dn = "+str(resChiDn))
    print("condition  : ",np.array([conditionlamda,conditionChiUp,conditionChiDn],dtype=complex))
    
    return np.array([conditionlamda,conditionChiUp,conditionChiDn])

# sol_object = optimize.root(selfConsistCond, np.array([initLamda,initChiUp,initChiDn]), method = 'anderson')
# print("\n The solution is")
# print(sol_object.x)
# sollamda = sol_object.x[0]
# solChiUp = sol_object.x[1]
# solChiDn = sol_object.x[2]

sollamda = initLamda
solChiUp = initChiUp
solChiDn = initChiDn

selfConsistCond([initLamda,initChiUp,initChiDn])

tUp = hopping( 1,solChiUp,solChiDn)
tDn = hopping(-1,solChiDn,solChiUp)
bandsUp,eigVecsUp = eighOnMesh( 1,sollamda,tUp)
bandsDn,eigVecsDn = eighOnMesh(-1,sollamda,tDn)



X,Y = np.meshgrid(KY,KX)
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(DIM):
    ax.plot_surface(X, Y, bandsUp[:,:,i])
    ax.plot_surface(X, Y, bandsDn[:,:,i])
plt.show()



# res1d = 31
# X=np.linspace(-minE+0.001,maxE, res1d)
# Y=np.zeros(res1d)
# for i,x in enumerate(X):
#     Y[i] = selfConsistCond([x,1,1])[0]
# fig = plt.figure()
# plt.plot(X,Y)
# plt.show()

# res = 21
# X = np.linspace(0.001,1.1, res)
# Y = np.linspace(0.001,1.1, res)
# Z=[np.zeros((res,res)),np.zeros((res,res)),np.zeros((res,res))]
# for i,x in enumerate(X):
#     for j,y in enumerate(Y):
#         Z[0][i,j],Z[1][i,j],Z[2][i,j] = selfConsistCond([simple_lamdada_root,x,y])
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

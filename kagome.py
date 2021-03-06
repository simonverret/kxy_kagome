import numpy as np
from scipy import optimize
np.set_printoptions(4,suppress=True,sign="+",floatmode="fixed")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import spence as dilog

NSPIN = 2
DIM = 3
spinError = ValueError('spin must be 1 or -1')
verbose = 1

eta1 = -1*np.array([1,np.sqrt(3)])/2
eta2 =  np.array([1,0])
eta3 =  np.array([-1,np.sqrt(3)])/2

J=1
D=0.2
Bfield=0.01
T=1
res = 41
resX = res+1


initLamda = +3.1996
initChiUp = -0.7083
initChiDn = -0.6816

resY = resX # previously squarer: int( resX * 2/np.sqrt(3) )
Nk = resX*resY
KX= np.linspace( -np.pi/3.         , 2*np.pi/3       , resX ) # the hexagonal BZ is periodic
KY= np.linspace( -np.pi/np.sqrt(3) , np.pi/np.sqrt(3) , resY ) # so one can use a rectangle.
unitVolume = np.pi * 2*np.pi/np.sqrt(3) / Nk


def hopping(sigma,chiSig,chiOpp):
    return J*(chiSig+chiOpp) - 1j*sigma*D*chiOpp


def hamiltonian(kx,ky,sigma,lamda,hop,B=Bfield): # lambda with "b" is protected
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


def eighOnMesh(sigma,lamda,hop,B=Bfield):
    tensorShape = (len(KX),len(KY),DIM,DIM)
    hamiltonianOnMesh = np.empty(tensorShape,dtype=complex)
    for i,kx in enumerate(KX):
        for j,ky in enumerate(KY):
            hamiltonianOnMesh[i,j,:,:] = hamiltonian(kx,ky,sigma,lamda,hop,B)
    # print(hamiltonianOnMesh[10,10,:,:])
    return np.linalg.eigh(hamiltonianOnMesh)


def bose(energ,beta=1/T):
    return 1.0/(np.exp(beta*energ)-1)


def dummyIntegral(qtyOnMesh):
    return np.sum(qtyOnMesh) / Nk


def selfConsistCond(varSet,beta=1/T, B=Bfield):
    selfConsistCond.counter += 1
    lamda = varSet[0]
    chiUp = varSet[1]
    chiDn = varSet[2]
 
    tUp = hopping( 1,chiUp,chiDn)
    tDn = hopping(-1,chiDn,chiUp)
    bandsUp = eighOnMesh( 1,lamda,tUp,B)[0]
    bandsDn = eighOnMesh(-1,lamda,tDn,B)[0]

    # Doki et al., Universal..., 2018, sup. mat. Eq.(S6a)
    S = dummyIntegral( bose(bandsUp,beta) + bose(bandsDn,beta) )/2.
    # Doki et al., Universal..., 2018, sup. mat. Eq.(S6b)
    fnkUp = (bandsUp-(lamda-B)) # =the eigenvalues of the hopping matrix
    fnkDn = (bandsDn-(lamda+B))
    resChiUp = dummyIntegral( fnkUp * bose(bandsUp,beta) )/(np.abs(tUp))
    resChiDn = dummyIntegral( fnkDn * bose(bandsDn,beta) )/(np.abs(tDn))

    conditionlamda = 2*S-1
    conditionChiUp = resChiUp-chiUp
    conditionChiDn = resChiDn-chiDn

    if (verbose>2):
        print("call ",selfConsistCond.counter)
        print("vars in  :",np.array([lamda,chiUp,chiDn]))
        print("vars out :",np.array([S,resChiUp,resChiDn]))
        print("residual :",np.array([conditionlamda,conditionChiUp,conditionChiDn]))

    return np.array([conditionlamda,conditionChiUp,conditionChiDn])
selfConsistCond.counter =0


## Berry phase

def dhdk(XorY,kx,ky,sigma,lamda,hop):
    # XorY=0 for x, and 1 for y.
    k = np.array([kx,ky])
    etaSink1 = -eta1[XorY]*np.sin(np.dot(k,eta1))
    etaSink2 = -eta2[XorY]*np.sin(np.dot(k,eta2))
    etaSink3 = -eta3[XorY]*np.sin(np.dot(k,eta3))
    dhdk = np.matrix([
        [       0         ,  hop*etaSink1   ,       0         ],
        [       0         ,        0         ,  hop*etaSink2  ],
        [  hop*etaSink3  ,        0         ,       0         ]
    ])
    return dhdk + dhdk.H

def dhdkOnMesh(sigma,lamda,hop):
    tensorShape = (len(KX),len(KY),DIM,DIM)
    dhdkxOnMesh = np.empty(tensorShape,dtype=complex)
    dhdkyOnMesh = np.empty(tensorShape,dtype=complex)
    for i,kx in enumerate(KX):
        for j,ky in enumerate(KY):
            dhdkxOnMesh[i,j,:,:] = dhdk(0,kx,ky,sigma,lamda,hop)
            dhdkyOnMesh[i,j,:,:] = dhdk(1,kx,ky,sigma,lamda,hop)
    return dhdkxOnMesh, dhdkyOnMesh

def berryPhaseOnMesh(bands,eigVecs,dhdkx,dhdky):
    tensorShape = (len(KX),len(KY),DIM)
    berryPhase = np.zeros(tensorShape,dtype=complex)
    for i in range(len(KX)):
        for j in range(len(KX)):
            for n in range(DIM):
                rightx = np.dot(dhdkx[i,j,:,:],eigVecs[i,j,:,n])
                righty = np.dot(dhdky[i,j,:,:],eigVecs[i,j,:,n])
                dvdkx = np.zeros(DIM,dtype=complex)
                dvdky = np.zeros(DIM,dtype=complex)
                for m in range(DIM):
                    if (m != n):
                        dvdkx += (np.dot(np.conj(eigVecs[i,j,:,m]),rightx[:])/(bands[i,j,n]-bands[i,j,m])) *eigVecs[i,j,:,m]
                        dvdky += (np.dot(np.conj(eigVecs[i,j,:,m]),righty[:])/(bands[i,j,n]-bands[i,j,m])) *eigVecs[i,j,:,m]
                berryPhase[i,j,n] =  2*np.imag(np.dot(np.conj(dvdkx),dvdky))
    return berryPhase

def c2(x):
    # dilog from scipy.special.spence, defined for z = 1 - x
    c2 = (1+x)*(np.log((1+x)/x))**2 - (np.log(x))**2 - 2*dilog(1-(-x))
    return c2

def kappaxy(bands,omega,beta=1/T):
    return -dummyIntegral( (c2(bose(bands,beta))-np.pi*np.pi/3)*omega )/beta


def evaluateKappaXY(initialValues,temp,Bfield):
    print("\ntemp T = "+str(round(temp,3))+"  field B = "+str(round(Bfield,3)))

    selfConsistCond.counter =0
    sol_object = optimize.root(selfConsistCond, np.array(initialValues), args=(1/temp,Bfield))
    print("Solution reached in "+str(selfConsistCond.counter)+" calls:")
    
    print("lamba, chiUp, chiDn = ",sol_object.x)
    solLamda = sol_object.x[0]
    solChiUp = sol_object.x[1]
    solChiDn = sol_object.x[2]
    nextValues = sol_object.x

    tUp = hopping( 1,solChiUp,solChiDn)
    tDn = hopping(-1,solChiDn,solChiUp)
    print("tUp, tDn = ",np.array([tUp,tDn]))
    bandsUp,eigVecsUp = eighOnMesh( 1,solLamda,tUp,B=Bfield)
    bandsDn,eigVecsDn = eighOnMesh(-1,solLamda,tDn,B=Bfield)
    dhdkxUp,dhdkyUp   = dhdkOnMesh( 1,solLamda,tUp)
    dhdkxDn,dhdkyDn   = dhdkOnMesh(-1,solLamda,tDn)
    omegaUp = berryPhaseOnMesh(bandsUp,eigVecsUp,dhdkxUp,dhdkyUp)
    omegaDn = berryPhaseOnMesh(bandsDn,eigVecsDn,dhdkxDn,dhdkyDn)
    
    kxy = np.real(kappaxy(bandsUp,omegaUp,beta=1/temp) + kappaxy(bandsDn,omegaDn,beta=1/temp))
    print("kxy = ",np.array([kxy]))
    return kxy, nextValues
    




solLamda = initLamda
solChiUp = initChiUp
solChiDn = initChiDn
tempList = np.concatenate((np.arange(1.2,0.5,-0.05),np.arange(0.5,0.2,-0.01)))
kappaXyList = np.zeros(len(tempList))
initialValues = [solLamda,solChiUp,solChiDn]
for ii,temp in enumerate(tempList):
    kappaXyList[ii], initialValues = evaluateKappaXY(initialValues,temp,Bfield)

solLamda = +3.2005
solChiUp = -0.7091
solChiDn = -0.6824
temp = 1.0
BfieldList1 = np.arange(0.01,0.0,-0.002)
BfieldList2 = np.arange(0.01,0.03,0.002)
kappaXyListvsB = np.zeros(len(BfieldList1) + len(BfieldList2))
initialValues = [solLamda,solChiUp,solChiDn]
for ii,Bfield in enumerate(BfieldList1):
    kappaXyListvsB[ii], initialValues = evaluateKappaXY(initialValues,temp,Bfield)
initialValues = [solLamda,solChiUp,solChiDn]
for ii,Bfield in enumerate(BfieldList2):
    initialValues = [solLamda,solChiUp,solChiDn]
    kappaXyListvsB[ii + len(BfieldList1)], initialValues = evaluateKappaXY(initialValues,temp,Bfield)


fig,ax = plt.subplots(1)
Xplt = tempList
Yplt = np.array(kappaXyList)/tempList
ax.plot(Xplt,Yplt)
plt.show()

fig,ax = plt.subplots(1)
Xplt = np.concatenate((BfieldList1,BfieldList2))
Yplt = np.array(kappaXyListvsB)
ax.plot(Xplt,Yplt)
plt.show()


#######################
#### VARIOUS PLOTS ####
#######################

# solLamda = 2.2
# tUp = (1.1246 + 0.34668j)*2
# tDn = (1.137 + 0.35666j)*2

# print(tUp)
# print(tDn)
# print(solLamda)

# # #### BANDS FOR SPIN UP AND DOWN in 3D
# X,Y = np.meshgrid(KY,KX)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(DIM):
#     ax.plot_surface(X, Y, bandsUp[:,:,i])
#     ax.plot_surface(X, Y, bandsDn[:,:,i])
# plt.show()

# # #### BERRY CURVATURE in 3D
# X,Y = np.meshgrid(KY,KX)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(DIM):
#     ax.plot_surface(X, Y, omegaUp[:,:,i])
#     ax.plot_surface(X, Y, omegaDn[:,:,i])
# plt.show()

# ### BANDS AND BERRY ALONG KX
# zerox = int(res/3)+1
# zeroy = int(res/2)
# overx = int(res/6)+1
# Xplt = np.concatenate( ((KX[zerox:],(KX[:overx]+np.pi))) )/(np.pi)
# fig,ax = plt.subplots(2, sharex=True)
# for i in range(DIM):
#     ZpltUp = np.concatenate(( bandsUp[zerox:,zeroy,i],bandsUp[:overx,0,i] ))
#     ZpltDn = np.concatenate(( bandsDn[zerox:,zeroy,i],bandsDn[:overx,0,i] ))
#     ax[0].plot(Xplt, ZpltUp)
#     ax[0].plot(Xplt, ZpltDn)
# for i in range(DIM):
#     ZpltUp = np.concatenate(( omegaUp[zerox:,zeroy,i],omegaUp[:overx,0,i] ))
#     #ZpltDn = np.concatenate(( omegaDn[zerox:,zeroy,i],omegaDn[:overx,0,i] ))
#     ax[1].plot(Xplt, ZpltUp)
#     #ax[1].plot(Xplt, ZpltDn)
# plt.show()

# ### C2 FUNCTION
# fig,ax = plt.subplots(1)
# Xplt = np.arange(1,100)/10.
# Yplt = c2(Xplt)
# ax.plot(Xplt, Yplt)
# plt.show()





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
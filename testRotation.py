# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:10:12 2024

@author: jcugnoni
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button, Slider
import numpy as np

import matplotlib
from CompositeDesigner import *

matplotlib.use("QtAGG")


def rotmatrixLocal2Global(alpha,beta,gamma):
    # Rotation matrix defined by yaw = alpha, pitch = beta, roll = gamma in degrees wrt global coord system
    # transformation of Local coordinate System to Global coordinate system
    # NOTE the correct transformation is achieve by the reverse sequence X Y Z in matrix operations 
    
    # Rotation matrices
    alpha=alpha/180.0*np.pi
    
    beta = - beta/180.0*np.pi
    
    gamma=gamma/180.0*np.pi
    
    # Rotation matrices
    R_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    # Combined rotation matrix
    R =  R_z @ R_y @ R_x

    return R

def rotmatrixGlobal2Local(alpha,beta,gamma):
    # Rotation matrix defined by yaw = alpha, pitch = beta, roll = gamma in degrees wrt global coord system
    # transformation of Local coordinate System to Global coordinate system
    # note the transformation is achieve by using the Z Y X sequence with negative angles to achieve the correct results
    
    # Rotation matrices
    alpha= -alpha/180.0*np.pi
    
    beta =  beta/180.0*np.pi
    
    gamma= -gamma/180.0*np.pi
    
    # Rotation matrices
    R_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

    # Combined rotation matrix
    R =  R_x @ R_y @ R_z

    return R


# compute local (material) coordinate system support vectors expressed in global coordinates
def vectF2(a,b,c):
    vx=np.array([1.0,0.0,0.0])
    vy=np.array([0.0,1.0,0.0])
    vz=np.array([0.0,0.0,1.0])
    R=rotmatrixLocal2Global(a, b, c)
    vx=R @ vx
    vy=R @ vy
    vz=R @ vz
    return vx, vy, vz

# compute the global coordinate system trihedron expressed in local coordinates system
def vectF2INV(a,b,c):
    vx=np.array([1.0,0.0,0.0])
    vy=np.array([0.0,1.0,0.0])
    vz=np.array([0.0,0.0,1.0])
    R=rotmatrixGlobal2Local(a, b, c)
    vx=R @ vx
    vy=R @ vy
    vz=R @ vz
    return vx, vy, vz

    
# compute local (material) coordinate system support vectors expressed in global coordinates
def vectF(a,b,c):
    vx=[1.0,0.0,0.0]
    vy=[0.0,1.0,0.0]
    vz=[0.0,0.0,1.0]
    # nautical angles (see Code-Aster doc for AFFE_CARA_ELEM for example) 
    # order : Z-Y-X for global to local transform and X-Y-Z for local to global transform
    M=RotX(c)
    vx=prodMatVect(M,vx)
    vy=prodMatVect(M,vy)
    vz=prodMatVect(M,vz)
    
    M=RotY(-b)
    vx=prodMatVect(M,vx)
    vy=prodMatVect(M,vy)
    vz=prodMatVect(M,vz)

    M=RotZ(a)
    vx=prodMatVect(M,vx)
    vy=prodMatVect(M,vy)
    vz=prodMatVect(M,vz)

    return vx,vy,vz

# input angles
a=-45.0
b=-30.0
c=-20.0
orient=OrientationItem("defaut")

# compute transorm, using either vectF (python code in GUI) or vectF2 (numpy code in Aster)
vx,vy,vz=vectF(a,b,c)
vx2,vy2,vz2=vectF2(a,b,c)


print([vx,vy,vz])
print([vx2,vy2,vz2])

print(rotmatrixGlobal2Local(a,b,c))

print(rotmatrixLocal2Global(a,b,c))

print(rotmatrixLocal2Global(a,b,c) @ rotmatrixGlobal2Local(a,b,c))

vx3 = rotmatrixLocal2Global(a,b,c) @ np.array([1,1,1])
print(vx3)

vx4 = rotmatrixGlobal2Local(a,b,c) @ vx3
print(vx4)



# check vector to angle

alpha=atan2(vx[1],vx[0])
beta=atan2(vx[2], sqrt( vy[2]**2 + vz[2]**2) )
gamma=atan2(vy[2] , vz[2])
alpha=alpha*180/pi
beta=beta*180/pi
gamma=gamma*180/pi


print(" VECT 2  ANGLE\n")
print(a,alpha)
print(b,beta)
print(c,gamma)

def plotVect(v,axis,color):
    x=[0,v[0]]
    y=[0,v[1]]
    z=[0,v[2]]
    axis.plot(x,y,z,color)

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True

fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')



ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

plotVect([1,0,0],ax,'k-')
plotVect([0,1,0],ax,'k-')
plotVect([0,0,1],ax,'k-')


plotVect(vx,ax,'r-')
plotVect(vy,ax,'g-')
plotVect(vz,ax,'b-')

#plt.show()


# check local vector build
vref=[1,1,0]
vz=[1,1,1]
vx=multVect( 1.0 / norm(vref) , vref)
vz=multVect( 1.0 / norm(vz) , vz)
vz=diffVect(vz, multVect( dot(vx,vz) , vx) )
vz=multVect( 1.0 / norm(vz) , vz)
vy=cross(vz, vx)
print("---------------")
print(vx); print(vy) ; print(vz)



# stress tensor transform 2D
# input angles
a=0.0
b=0.0
c=0.0

aa=a/180*pi

R1=rotmatrixGlobal2Local(a,b,c)

Sxx=1000; Sxy=0; Syy=0; Szz=-1000; Sxz=0; Syz=0;

Sglob=np.array([[Sxx,Sxy,Sxz],[Sxy,Syy,Syz],[Sxz,Syz,Szz]])


print(Sglob)

Slocal= R1 @ Sglob @ R1.T

print(Slocal)

S11=Sxx*cos(aa)**2 + Syy*sin(aa)**2 + 2*Sxy*sin(aa)*cos(aa) 
S22=Sxx*sin(aa)**2 + Syy*cos(aa)**2 - 2*Sxy*sin(aa)*cos(aa) 
S12=(Syy-Sxx)*sin(aa)*cos(aa) + Sxy*(cos(aa)**2 - sin(aa)**2)

print(S11, S12, S22)



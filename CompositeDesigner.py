# COMPOSITE DESIGNER GUI, J. Cugnoni / HEIG-VD, 2022
#  Licence ??


import sys
import os
import json, pickle
from math import *
from string import Template
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QTreeView, QVBoxLayout
from PyQt5.QtWidgets import QMenu, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from PyQt5.uic import loadUi

compdesignerbasepath=os.path.dirname(os.path.realpath(__file__))

# HELPERS

def norm(a):
    return sqrt(a[0]**2+a[1]**2+a[2]**2)

def sumVect(a,b):
    return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]

def multVect(x,a):
    return [x*a[0],x*a[1],x*a[2]]

def diffVect(a,b):
    return sumVect(a,multVect(-1.0,b))

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

def dot(a,b):
    c= a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    return c

def prodMatVect(M,v):
    c=[0,0,0]
    for i in range(3):
        for j in range(3):
            c[i]=c[i] + M[i][j] * v[j]
    return c

def RotX(alpha): 
    # note input angle is in degrees
    c=cos(alpha / 180.0 * pi)
    s=sin(alpha / 180.0 * pi)
    M=[[1,0,0] , [0, c, -s], [0, s, c]]
    return M

def RotY(alpha): 
    # note input angle is in degrees
    c=cos(alpha / 180.0 * pi)
    s=sin(alpha / 180.0 * pi)
    M=[[c,0,s] , [0, 1, 0], [-s, 0, c]]
    return M

def RotZ(alpha): 
    # note input angle is in degrees
    c=cos(alpha / 180.0 * pi)
    s=sin(alpha / 180.0 * pi)
    M=[[c,-s,0] , [s, c, 0], [0, 0, 1]]
    return M


# --- DATA CLASSES ---

# base class to store the model tree

class CompositeModel(QStandardItemModel):
    def __init__(self):
        super().__init__()  
        self.addBaseItems()
    
    def addBaseItems(self):
        item=QStandardItem('Materials')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootMaterials=item
        self.appendRow(item)
        item=QStandardItem('Orientations')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootOrientations=item
        self.appendRow(item)
        item=QStandardItem('Solid Plies')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootSolidPlies=item
        self.appendRow(item)
        item=QStandardItem('Shell Layups')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootShellLayups=item
        self.appendRow(item)
        item=QStandardItem('Shell Regions')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootShellRegions=item
        self.appendRow(item)
        item=QStandardItem('Simulation')
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) #not editable
        self.rootSimulation=item
        self.appendRow(item)
        
        
       # def addItems(self, parent, elements):
       #     for text, children in elements:
       #         item = QStandardItem(text)
       #         parent.appendRow(item)
       #         if children:
       #             self.addItems(item, children)
              
    def addMaterial(self):
        matlist=getMaterialList(self)
        matname='Mat'+str(len(matlist))
        item=MaterialItem(matname)
        self.rootMaterials.appendRow(item)
 
    def addOrientation(self):
        orlist=getOrientationList(self)
        name='Orient'+str(len(orlist))
        item=OrientationItem(name)
        self.rootOrientations.appendRow(item)
       
    def addSolidPly(self):
        lst=getSolidPliesList(self)
        name='Ply'+str(len(lst))
        item=SolidPlyItem(name)
        self.rootSolidPlies.appendRow(item)
    
    def addShellLayup(self):
        lst=getShellLayupsList(self)
        name='Layup'+str(len(lst))
        item=ShellLayupItem(name)
        self.rootShellLayups.appendRow(item)

    def addShellRegion(self):
        lst=getShellRegionsList(self)
        name='Shell'+str(len(lst))
        item=ShellRegionItem(name)
        self.rootShellRegions.appendRow(item)    
    
    def deleteItem(self,item):
        parent=item.parent()
        row=item.row()
        if row>=0:
            parent.removeRow(row)
            
    def getInfoFromIndex(self,index):
        parentName = ""
        parentRow = -1
        selectName =""
        selectRow = -1
        level = 0
        selectedItem=self.itemFromIndex(index)
        selectName = str(index.data())
        selectRow = index.row()
        parentName = str( index.parent().data() )
        parentRow = index.parent().row()
        while index.parent().isValid():
            index = index.parent()
            level += 1
        return selectedItem, parentName, parentRow, selectName, selectRow, level
    
    def loadModelData(self):
        name , typ = QFileDialog.getOpenFileName(None,"Load File", "","composite design (*.json);;All files (*.*);;All files (*);;")
        file = open(name,'r')
        # list of "roots" in item tree for each data sets
        modelDataRoots={'Materials':self.rootMaterials, 
                        'Orientations':self.rootOrientations,
                        'SolidPlies':self.rootSolidPlies,
                        'ShellLayups':self.rootShellLayups,
                        'ShellRegions':self.rootShellRegions}
        modelDataClasses={'Materials':MaterialItem,
                          'Orientations':OrientationItem,
                          'SolidPlies':SolidPlyItem,
                          'ShellLayups':ShellLayupItem,
                          'ShellRegions':ShellRegionItem}
        modelData = json.load(file)
        file.close()
        for key in modelData.keys():
            if key in modelDataRoots.keys():  # ignore unsupported data 
                root=modelDataRoots[key]
                # clear objects
                root.removeRows(0,root.rowCount())
                dic=modelData[key]                
                for name in dic.keys():
                    data=dic[name]
                    modelClass=modelDataClasses[key]
                    item=modelClass(name)
                    item.__dict__=data
                    root.appendRow(item)
        
    def saveModelData(self):
        name,typ = QFileDialog.getSaveFileName(None, 'Save File' ,'',"composite design (*.json);;All files (*.*);;All files (*);;")
        file = open(name,'w')

        # list of dict and corresponding "roots" in the model tree
        modelDataRoots={'Materials':self.rootMaterials, 
                        'Orientations':self.rootOrientations,
                        'SolidPlies':self.rootSolidPlies,
                        'ShellLayups':self.rootShellLayups,
                        'ShellRegions':self.rootShellRegions}
        # build a dictionary containing each base types 
        modelData={}
        for key in modelDataRoots.keys():
            modelData[key]={}
            root=modelDataRoots[key]
            dic=modelData[key]
            for j in range(root.rowCount()):
                item=root.child(j)
                if hasattr(item,'ui'):
                    item.ui=None
                name=item.text()
                data=item.__dict__
                dic[name]=data
        json.dump(modelData, file)
        file.close()
        
    def exportCodeAsterModel(self):
        name,ftype = QFileDialog.getSaveFileName(None, 'Export Code-Aster commands' ,'',"CA commands (*.comm);")
        file = open(name,'w')
        
        # DEBUT & LIRE_MAILLAGE
        txt="DEBUT(LANG='EN')\n\nmesh = LIRE_MAILLAGE(UNITE=20)\n\n# ! CHECK Shell faces normal in POST ! \n\n"
        file.write(txt)
        
        # ASK user if the shell orientation shall be corrected
        qm = QMessageBox()
        ret = qm.question(None,'Modify shell normal', "Do you want to modify shell normals (top/down) based on normal vectors defined in the orientation? If No, the shell normals as defined in the mesh will be used (light blue = top, dark blue bottom)", qm.Yes | qm.No)
        if ret == qm.Yes:
            modifyNormals=True
        else:
            modifyNormals=False
        
        # orient shell normals
        # example
        #         mesh = MODI_MAILLAGE(reuse=mesh, MAILLAGE=mesh, ORIE_NORM_COQUE= ( 
        #    _F(GROUP_MA = ('testShell', '') , GROUP_NO = ('testShell', '') , VECT_NORM= (0.0, 0.0, 1.0) ),  
        # )
        # )
        shellRegionList=getShellRegionsList(self)
        if (len(shellRegionList)>0) & modifyNormals: 
            lst=''
            txtStart =' ( \n'
            txtLoop='   _F(GROUP_MA = $groupList , GROUP_NO = $groupNode , VECT_NORM= $Vnorm ),  \n'
            txtEnd =' )\n'
            
            lst= lst + txtStart
            for i in range(self.rootShellRegions.rowCount()):
                shellRegion=self.rootShellRegions.child(i)
                grpList=tuple([ grp for grp in shellRegion.groupList ])
                grpNode=( grpList[0] , )
                orient=getChildByName(self.rootOrientations,shellRegion.orientationName)
                vnorm=tuple(orient.vectors['Vnorm'])
                lst = lst + Template(txtLoop).substitute(groupList=str(grpList), groupNode = str(grpNode), Vnorm=str(vnorm)) 
            lst= lst + txtEnd
            
            txt='mesh = MODI_MAILLAGE(reuse=mesh, MAILLAGE=mesh, ORIE_NORM_COQUE= $listOrientShell ) \n'
            file.write(Template(txt).substitute(listOrientShell=lst))
            
        # FE modeling
        # Example:
        # model = AFFE_MODELE(AFFE= (
        #  _F(GROUP_MA= ('', 'testShell') , MODELISATION=('DKT', ), PHENOMENE='MECANIQUE'),
        #  _F(GROUP_MA= ('', 'testSolid') , MODELISATION=('3D', ), PHENOMENE='MECANIQUE'), 
        # )
        #  ,MAILLAGE=mesh)
        txt='model = AFFE_MODELE(AFFE= $listFEM , MAILLAGE=mesh)\n\n'
        ## build model assignement list
        lst='(\n'
        dowrite=False
        if len(getShellLayupsList(self))>0:
            dowrite=True
            grplist=[]
            for i in range(self.rootShellRegions.rowCount()):
                shellRegion=self.rootShellRegions.child(i)
                for grp in shellRegion.groupList:
                    grplist.append(grp)
            grplist=list(set(grplist)) # use a set to keep only unique values
            temp=Template(" _F(GROUP_MA= $groupList , MODELISATION=('DKT', ), PHENOMENE='MECANIQUE'),\n")
            lst=lst + temp.substitute(groupList=str(tuple(grplist)))   
        if len(getSolidPliesList(self))>0:
            dowrite=True
            grplist=[]
            for i in range(self.rootSolidPlies.rowCount()):
                solidPly=self.rootSolidPlies.child(i)
                for grp in solidPly.groupList:
                    grplist.append(grp)
            grplist=list(set(grplist)) # use a set to keep only unique values
            temp=Template(" _F(GROUP_MA= $groupList , MODELISATION=('3D', ), PHENOMENE='MECANIQUE'), \n\n")
            lst=lst + temp.substitute(groupList=str(tuple(grplist)))   
        lst=lst+')\n'
        if dowrite:
            file.write(Template(txt).substitute(listFEM=lst))  
        
        # Shell properties 
        # Example:
        # listShellProp = ( 
        #                   _F(ANGL_REP=(90.0, 0.0), COQUE_NCOU=3, EPAIS=2.0, GROUP_MA=('face', )) , 
        #                   _F(VECTEUR=(1.0, 0.0, 0.0), COQUE_NCOU=3, EPAIS=2.0, GROUP_MA=('face2', )) , 
        #                 )
        # elemprop = AFFE_CARA_ELEM(COQUE= listShellProp,
        #                           INFO=1,
        #                           MODELE=model)
        #
        txt="elemprop = AFFE_CARA_ELEM(  INFO=1, MODELE=model, $options )\n\n" 
        txtShell="COQUE= $listShellProp , \n "
        txtLoopShell='  _F(VECTEUR= $Vref , COQUE_NCOU= $nply , EPAIS= $thick , GROUP_MA= $groupList ) ,\n'
        txtSolid="MASSIF= $solidPropList , \n"
        txtLoopSolid="_F(ANGL_REP=( $alpha , $beta , $gamma ), GROUP_MA= $groupList ),\n"
        lst='  (\n'
        optionsTxt=''
        dowrite=False
        if len(getShellLayupsList(self))>0:
            dowrite=True
            for i in range(self.rootShellRegions.rowCount()):
                shellRegion = self.rootShellRegions.child(i)
                grplist = shellRegion.groupList
                layupName = shellRegion.layupName
                layup = getChildByName(self.rootShellLayups,layupName)
                nply = len(layup.plyList)
                thick = 0.0
                for ply in layup.plyList:
                    thick = thick + ply[1]   #ply=[matname,thickness,angle]
                if nply == 0:
                    print('Data error / warning, the composite layup ', layupName, ' does not contain plies !\n')
                    print(' nb layer (COQUE_NCOU) and thickness (EPAIS) has been set to 1 in AFFE_CARA_ELEM\n')
                    nply = 1
                    thick = 1.0
                orientName = shellRegion.orientationName
                orient = getChildByName(self.rootOrientations,orientName)
                Vref = orient.vectors['Vref']
                lst = lst + Template(txtLoopShell).substitute(Vref=str(tuple(Vref)), nply=nply, thick=thick,
                                                     groupList=str(tuple(grplist)) )
            lst=lst+'  ) \n'
            optionsTxt = optionsTxt + Template(txtShell).substitute(listShellProp=lst)
        if len(getSolidPliesList(self))>0:
            dowrite=True
            lst='('
            for i in range(self.rootSolidPlies.rowCount()):
                solidPly = self.rootSolidPlies.child(i)
                grplist = solidPly.groupList
                orientName = solidPly.orientationName
                orient = getChildByName(self.rootOrientations, orientName)
                alpha = orient.angles['alpha']
                beta = orient.angles['beta']
                gamma = orient.angles['gamma']
                lst = lst + Template(txtLoopSolid).substitute(alpha=str(alpha), beta=str(beta), gamma=str(gamma), \
                                                              groupList=str(tuple(grplist)) )
            lst = lst + ')'
            optionsTxt = optionsTxt + Template(txtSolid).substitute(solidPropList=lst)
        if dowrite:
            file.write(Template(txt).substitute(options = optionsTxt))
            
        # Material definitions, example
        #define orthotropic material for analysis
        # $matName = DEFI_MATERIAU(ELAS_ORTH=_F(E_L=$E1, E_N=$E3, E_T=$E2, G_LN=$G12, G_LT=$G13, G_TN=$G23, 
        #             NU_LN=$NU13, NU_LT=$NU12, NU_TN=$NU23, RHO=$RHO,
        #             S_LT=$S12, XC=$XC, XT=$XT, YC=$YC, YT=$YT))
        txt='''$matName = DEFI_MATERIAU(ELAS_ORTH=   
        _F(E_L=$E1, E_N=$E3, E_T=$E2, G_LN=$G12, G_LT=$G13, G_TN=$G23, 
           NU_LN=$NU13, NU_LT=$NU12, NU_TN=$NU23, RHO=$RHO,
           S_LT=$S12, XC=$XC, XT=$XT, YC=$YC, YT=$YT))\n\n'''
        for i in range(self.rootMaterials.rowCount()):
            mat=self.rootMaterials.child(i)
            el=mat.elasticProperties
            st=mat.strengthProperties
            matname=mat.text()
            file.write(Template(txt).substitute(E1=el['E1'],E2=el['E2'],E3=el['E3'],
                                                G12=el['G12'],G13=el['G13'],G23=el['G23'],
                                                NU12=el['NU12'],NU13=el['NU13'],NU23=el['NU23'],
                                                RHO=el['RHO'],S12=st['S12'],
                                                XC=st['XC'],XT=st['XT'],
                                                YT=st['YT'],YC=st['YC'], matName=matname) ) 
          
        # shell layup section, example:
        # listPlies = (
        #              _F(EPAIS=0.666, MATER=UD1, ORIENTATION=-45.0),
        # )
        # $layupName = DEFI_COMPOSITE( COUCHE=listPlies, )
        txt="$layupName = DEFI_COMPOSITE( COUCHE= $listPlies, )\n\n" 
        txtLoop='  _F(EPAIS= $thick , MATER= $matName, ORIENTATION= $angle) ,\n'
        
        if self.rootShellLayups.rowCount()>0:
            for i in range(self.rootShellLayups.rowCount()):
                layup=self.rootShellLayups.child(i)
                layupName=layup.text()
                nply=len(layup.plyList)
                plies=layup.plyList
                if nply==0:
                    print('Data error / warning, the composite layup ', layupName, ' does not contain plies !\n')
                    print(' to avoid import problem a dummy layup was defined !!\n')
                    plies=[ ['dummyMat', 1.0, 0.0] , ] 
                lst='  (\n'
                for ply in plies:    
                    matName=ply[0]
                    thick=ply[1]
                    angle=ply[2]
                    lst=lst+Template(txtLoop).substitute(matName=matName,angle=angle,thick=thick)
                lst=lst+'  ) \n'
                file.write(Template(txt).substitute(listPlies=lst,layupName=layupName))
            
        # material regions assignement
        # example
        # fieldmat = AFFE_MATERIAU(AFFE=_F(GROUP_MA=('face', ),
        #                                  MATER=(lam1, )),
        #                          MODELE=model)
        txt="fieldmat = AFFE_MATERIAU(AFFE= $affeList , MODELE=model)\n\n"
        txtLoop="_F(GROUP_MA= $grpList , MATER=( $matName , ) ),\n"
        lst='('
        dowrite=False
        if len(getShellLayupsList(self))>0:
            dowrite=True
            for i in range(self.rootShellRegions.rowCount()):
                grplist=[]
                shellRegion=self.rootShellRegions.child(i)
                for grp in shellRegion.groupList:
                    grplist.append(grp)
                layupName=shellRegion.layupName
                lst=lst + Template(txtLoop).substitute(grpList=str(tuple(grplist)),matName=layupName)   
        if len(getSolidPliesList(self))>0:
            dowrite=True
            for i in range(self.rootSolidPlies.rowCount()):
                grplist=[]
                solidPly=self.rootSolidPlies.child(i)
                for grp in solidPly.groupList:
                    grplist.append(grp)
                matname=solidPly.materialName
                lst=lst + Template(txtLoop).substitute(grpList=str(tuple(grplist)),matName=matname)      
        lst=lst+')'
        if dowrite:
            file.write(Template(txt).substitute(affeList=lst))  
            
        # basic load case: gravity
        # load = AFFE_CHAR_MECA(
        #   MODELE=model, 
        #   PESANTEUR=_F(
        #     DIRECTION=(0.0, 0.0, -1.0), 
        #     GRAVITE=9810.0
        #   )
        # ) 
        txt="# !!TO COMPLETE by USER!!\nload = AFFE_CHAR_MECA( MODELE=model, PESANTEUR=_F( DIRECTION=(0.0, 0.0, -1.0),GRAVITE=9810.0) )\n\n"
        file.write(txt)
        
        # write concepts to output, for model verification
        #IMPR_CONCEPT(
        #   CONCEPT=(
        #         _F(CHAM_MATER=fieldmat ), 
        #         _F(CARA_ELEM=elemprop, MODELE=model, REPERE_LOCAL='ELEM'), 
        #         _F(CHARGE=load)
        #         ), 
        #   UNITE=2
        # ) 
        txt="IMPR_CONCEPT(CONCEPT=(_F(CHAM_MATER=fieldmat ), _F(CARA_ELEM=elemprop, MODELE=model, REPERE_LOCAL='ELEM'),_F(CHARGE=load)), UNITE=81)\n\n"
        file.write(txt)
        # linear solver
        #reslin = MECA_STATIQUE(
        #   CARA_ELEM=elemprop, 
        #   CHAM_MATER=fieldmat, 
        #   EXCIT=_F(
        #     CHARGE=load
        #   ), 
        #   MODELE=model
        # ) 
        
        txt="reslin = MECA_STATIQUE( CARA_ELEM=elemprop, CHAM_MATER=fieldmat, EXCIT=_F(CHARGE=load), MODELE=model) \n\n"
        file.write(txt)
        
        # basic output
        # IMPR_RESU(
        #   RESU=_F(
        #     RESULTAT=reslin
        #   ), 
        #   UNITE=80
        # ) 
        txt="IMPR_RESU(RESU=_F(RESULTAT=reslin), UNITE=80)\n\n"
        file.write(txt)
        
        file.close()
        
    def exportCodeAsterPostPro(self):
        # Compute failure criteria on composite structures.
        # Definitions of user defined fields:
        # Ply level stresses :
        #  UT01_ELGA
        #   X1 = S11 = stress along material dir 1 (= fibers usually)
        #   X2 = S22 = stress along mat. Dir 2 (= transverse in plane usually)
        #   X3 = S12 (in plane shear stress, note :TODO, add interlaminar shear !!!)
        # Failure criterion
        #  UT02_ELGA.
        #   X1= sqrt(CritFT) = linearized Hashin fiber tension criterion (similar to von Mises criterion)
        #   X2= Sqrt(CritFC) = linearized Hashin fiber compression criterion
        #   X3=Sqrt(CritMT) = linearized Hashin matrix tensile criterion
        #   X4=Sqrt(CritMC)= linearized Hashin matrix compression criterion
        #
        #  Note: the original Hashin criteria are quadratic with stress and defined as:
        #   Fiber tension failure: CritFt = ( 0.5* (S11+abs(S11) ) / XT)**2 + (S12 / SL)**2
        #     where 0.5*(S11+abs(S11)) = S11 for positive values of S11, else =0
        #   Fiber compression failure : CritFc= (0.5 * (S11-abs(S11)) / XC )**2
        #     where 0.5*(S11-abs(S11)) = S11 for negative values of S11, else =0
        #   Matrix tension failure: CritMt = (0.5* (S22+abs(S22)) /YT)**2 + (S12/SL)**2
        #   Matrix compression failure: CritMc = ( 0.5*(S22-abs(S22) ) / YC )**2 + (S12/SL)**2
        #
        #   Here the criteria are linearized to facilitate calculation of the safety factor.
        #   with the linearized criterion formulation, the safety factor is simply 1/LinCriterion
        
        # in progress: implement post pro for solid regions (use formula for stress rotation for each solid region  => failure criteria)

        name,ftype = QFileDialog.getSaveFileName(None, 'Export Code-Aster commands' ,'',"CA commands (*.comm);")
        file = open(name,'w')
        
        # POURSUITE (follow up on previous job)
        txt="POURSUITE()\n\n"
        file.write(txt)

        SolidPostProPythonDefs = """
import numpy as np


# PYTHON FUNCTIONS
def rotmatrix(alpha,beta,gamma):
    # Rotation matrix defined by yaw = alpha, pitch = beta, roll = gamma in degrees wrt global coord system
    
    # Rotation matrices
    alpha=alpha/180.0*np.pi
    
    beta = beta/180.0*np.pi
    
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
    R = R_x @ R_y @ R_z

    return R


def transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma):
    # transform stress components SIXX,.. to local coord system defined by yaw,pitch, roll angles (in degree) 
    sigma_global = np.array([
        [SIXX, SIXY, SIXZ],
        [SIXY, SIYY, SIYZ],
        [SIXZ, SIYZ, SIZZ]
    ])
    
    R=rotmatrix(alpha,beta,gamma)

    # Transform the stress tensor
    sigma_local = R @ sigma_global @ R.T

    # Extract the components
    SI11 = sigma_local[0, 0]
    SI22 = sigma_local[1, 1]
    SI33 = sigma_local[2, 2]
    SI12 = sigma_local[0, 1]
    SI13 = sigma_local[0, 2]
    SI23 = sigma_local[1, 2]

    return SI11, SI22, SI33, SI12, SI13, SI23

def LocalStress(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ,alpha=0,beta=0,gamma=0):
    SI11, SI22, SI33, SI12, SI13, SI23 = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SI11, SI22, SI33, SI12, SI13, SI23

def SI11(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIXX

def SI22(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIYY

def SI33(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIZZ

def SI12(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIXY

def SI13(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIXZ

def SI23(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0):
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    return SIYZ

def FT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0,XT=1,S12=1,S13=1):
    #compute local failure criterion FT in solid region 
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    FT=(SIXX+abs(SIXX))/(2*abs(SIXX)) * sqrt( (SIXX / XT )**2 + ( SIXY / S12 )**2 + ( SIXZ / S13 )**2 ) 
    return FT
   
def FC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0, XC=1):
    #compute local failure criterion FC in solid region 
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    FC=(abs(SIXX)-SIXX)/(2*abs(SIXX)) * sqrt( (SIXX / XC )**2  )  
    return FC

def MT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0, YT=1, ZT=1, S12=1, S13=1, S23=1 ):
    #compute local failure criterion MT in solid region 
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    MT=(abs(SIYY + SIZZ)+(SIYY+SIZZ))/(2*abs(SIYY+SIZZ)) * sqrt( ( (SIYY+SIZZ) / (0.5*(YT+ZT)) )**2 + (SIYZ**2-SIYY*SIZZ)/( ( S23 )**2 ) + (SIXY**2 + SIXZ**2)/ ( (0.5 * (S12+S13) )**2 )   )  
    return MT

def MC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=0,beta=0,gamma=0, YC=1, ZC=1, S12=1, S13=1, S23=1):
    #compute local failure criterion MC in solid region 
    SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ = transform_stress(SIXX,SIYY,SIZZ,SIXY,SIXZ,SIYZ, alpha, beta, gamma)
    try:
        ST=SIYY + SIZZ
        negValue=(abs(ST)-(ST))/(2.0*abs(ST))
        YZC=(0.5*(YC+ZC))
        F1=abs(1.0 / YZC  * ( ( YZC / (2.0 * S23 ) )**2 - 1.0 ))
        S1X= 0.5 * (S12+S13)
        Crit=F1 * ST + ( ST / (2.0 * S23 ) )**2 + (SIYZ**2 - SIYY*SIZZ)/( ( S23 )**2 ) + (SIXY**2 + SIXZ**2)/( ( S1X )**2 )
        MC=sqrt( abs(negValue * Crit) )
    except:
        MC=-1.0
    return MC

"""

        #write python functions 
        file.write(SolidPostProPythonDefs)

        # loop over each shell region
        shellFCresults=[]
        shellPlyresults=[]
        for idRegion in range(self.rootShellRegions.rowCount()):
            shellRegion = self.rootShellRegions.child(idRegion)
            shellRegionName=shellRegion.text()
            layup=getChildByName(self.rootShellLayups,shellRegion.layupName)
            groupList=shellRegion.groupList
            nplies=len(layup.plyList)
            plyList=layup.plyList
            # loop over each ply of the shell region   
            resnames=[]
            for j in range(nplies):    
                resultName='R'+str(idRegion)+'_'+str(j)   
                resnames.append(resultName)
                shellPlyresults.append(resultName)
                plyData=plyList[j]
                matname=plyData[0]
                material=getChildByName(self.rootMaterials,matname)
                el=material.elasticProperties
                st=material.strengthProperties
                angle=plyData[2] #ply angle in deg
                theta=angle/180.0*pi  # play angle in rad
                XT=abs(st['XT']) # ensure that XT is positive
                XC=-abs(st['XC']) # ensure that XC is negative !
                YT=abs(st['YT'])
                YC=-abs(st['YC'])
                S=abs(st['S12'])
                
                # extract ply level stresses in shell coordinate system (defined by Vref=0° direction)
                # ply1 = POST_CHAMP(EXTR_COQUE=_F(NIVE_COUCHE='MOY',
                #                         NOM_CHAM=('SIEF_ELGA', ),
                #                         NUME_COUCHE=1),
                #           GROUP_MA=('face', ),
                #           RESULTAT=reslin)
                txt= "$resuname = POST_CHAMP(EXTR_COQUE=_F(NIVE_COUCHE='MOY',NOM_CHAM=('SIEF_ELGA', ),NUME_COUCHE= $nply)," + \
                    "GROUP_MA= $grpList ,RESULTAT=reslin)\n\n"
                file.write( Template(txt).substitute(resuname=resultName, nply=str(j+1), grpList = str(tuple(groupList)) ) )
                
                # compute stresses in the ply reference frame
                # apply rotation formula (from shell reference frame (=0°) to ply ref frame)
                # # rotation of local stress state to material ref frame                                    
                # S11 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'),
                #               VALE='SIXX*cos(t)**2 + SIYY*sin(t)**2  + SIXY*2*cos(t)*sin(t)',
                #               t=theta)
                # S22 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'),
                #               VALE='SIXX*sin(t)**2 + SIYY*cos(t)**2  - SIXY*2*cos(t)*sin(t)',
                #               t=theta)
                # S12 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'),
                #               VALE='SIXX*-1*sin(t)*cos(t) + SIYY*sin(t)*cos(t) +SIXY*(cos(t)**2 - sin(t)**2)', 
                #               t=theta) 
                #
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'), VALE='SIXX*cos(${t})**2 + SIYY*sin(${t})**2  + SIXY*2*cos(${t})*sin(${t})')\n\n"
                formulaName=resultName+'_S11'
                file.write(Template(txt).substitute(name=formulaName,t=theta))
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'),VALE='SIXX*sin(${t})**2 + SIYY*cos(${t})**2  - SIXY*2*cos(${t})*sin(${t})')\n\n"
                formulaName=resultName+'_S22'
                file.write(Template(txt).substitute(name=formulaName,t=theta))
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIXY'),VALE='SIXX*-1*sin(${t})*cos(${t}) + SIYY*sin(${t})*cos(${t}) +SIXY*(cos(${t})**2 - sin(${t})**2)')\n\n"
                formulaName=resultName+'_S12'
                file.write(Template(txt).substitute(name=formulaName,t=theta))
                # compute local stresses
                # shell1_1 = CALC_CHAMP(reuse=shell1_1,
                #                  CHAM_UTIL=_F(FORMULE=(shell1_1_S11, shell1_1_S22, shell1_1_S12),
                #                  NOM_CHAM='SIEF_ELGA',
                #                  NUME_CHAM_RESU=1),
                #                  RESULTAT=shell1_1)
                txt="${name} = CALC_CHAMP(reuse=${name},CHAM_UTIL=_F(FORMULE=(${f11}, ${f22}, ${f12}), NOM_CHAM='SIEF_ELGA', NUME_CHAM_RESU=1), RESULTAT=${name})\n\n"
                file.write(Template(txt).substitute(name=resultName,f11=resultName+'_S11',f22=resultName+'_S22',f12=resultName+'_S12'))
                
                # evaluate failure criteria, based on Hashin 1980                                            
                # # Hashin failure criteria 
                # failFT = FORMULE(NOM_PARA=('X1', 'X3'),
                #                  VALE='( 0.5* (X1+abs(X1) ) / XT)**2 + (X3 / S)**2', XT=XT,S=S)
                
                # failFC = FORMULE(NOM_PARA=('X1', ),
                #                  VALE='( 0.5* (X1-abs(X1) ) / XC)**2 ',XC=XC)
                
                # failMT = FORMULE(NOM_PARA=('X2', 'X3'),
                #                  VALE='(0.5* (X2+abs(X2)) / YT)**2 + (X3/S)**2',YT=YT,S=S)
                
                # failMC = FORMULE(NOM_PARA=('X2', 'X3'),
                #                  VALE='(0.5* (X2-abs(X2)) / YC)**2 + (X3/S)**2',YC=YC,S=S)
                txt="${name} = FORMULE(NOM_PARA=('X1', 'X3'),VALE='sqrt( ( 0.5* (X1+abs(X1) ) / ${XT} )**2 + (X3 / ${S} )**2 ) ' )\n\n"
                file.write(Template(txt).substitute(name=resultName+'_FT',XT=str(XT),S=str(S)))
                txt="${name} = FORMULE(NOM_PARA=('X1', ),  VALE='sqrt( (0.5* (X1-abs(X1) ) / ${XC})**2 )')\n\n"
                file.write(Template(txt).substitute(name=resultName+'_FC',XC=str(XC)))
                txt="${name} = FORMULE(NOM_PARA=('X2', 'X3'), VALE='sqrt( (0.5* (X2+abs(X2)) / ${YT} )**2 + (X3/ ${S} )**2 ) ' )\n\n"
                file.write(Template(txt).substitute(name=resultName+'_MT', YT=str(YT), S=str(S) ) )
                txt="${name} = FORMULE(NOM_PARA=('X2', 'X3'), VALE='sqrt( (0.5* (X2-abs(X2)) / ${YC} )**2 + (X3/ ${S} )**2 ) ' )\n\n"
                file.write(Template(txt).substitute(name=resultName+'_MC', YC=str(YC), S=str(S) ) )
                
                # evaluate criteria
                # ply1 = CALC_CHAMP(reuse=ply1,
                #                   CHAM_UTIL=_F(FORMULE=(failFT, failFC, failMT, failMC),
                #                                NOM_CHAM='UT01_ELGA',
                #                                NUME_CHAM_RESU=2),
                #                   RESULTAT=ply1)
                
                # chCrit1 = CREA_CHAMP(NOM_CHAM='UT02_ELGA',
                #                      OPERATION='EXTR',
                #                      PROL_ZERO='OUI',
                #                      RESULTAT=ply1,
                #                      TYPE_CHAM='ELGA_NEUT_R')
                txt="${name} = CALC_CHAMP(reuse=${name}, CHAM_UTIL=_F(FORMULE=( $FT , $FC , $MT , $MC ), NOM_CHAM='UT01_ELGA', NUME_CHAM_RESU=2), RESULTAT=${name})\n\n"
                file.write(Template(txt).substitute(name=resultName,FT=resultName+'_FT',FC=resultName+'_FC',MT=resultName+'_MT',MC=resultName+'_MC'))
                txt="${name} = CREA_CHAMP(NOM_CHAM='UT02_ELGA', NUME_ORDRE=1, OPERATION='EXTR', PROL_ZERO='OUI', RESULTAT=${resu}, TYPE_CHAM='ELGA_NEUT_R')\n\n"
                file.write(Template(txt).substitute(name='FC_'+resultName,resu=resultName))
                
                # end of loop on plies
            # combine results from plies of the current shell region in a single "result"
            # result time codes = 1 + 0.001 * ply_id   
            
            # critRes = CREA_RESU(AFFE=(_F(CARA_ELEM=elemprop,
            #                  CHAM_GD=chCrit1,
            #                  CHAM_MATER=fieldmat,
            #                  INST=(1.0, ),
            #                  MODELE=model),
            #               _F(CARA_ELEM=elemprop,
            #                  CHAM_GD=chCrit2,
            #                  CHAM_MATER=fieldmat,
            #                  INST=(2.0, ),
            #                  MODELE=model),
            #               _F(CARA_ELEM=elemprop,
            #                  CHAM_GD=chCrit3,
            #                  CHAM_MATER=fieldmat,
            #                  INST=(3.0, ),
            #                  MODELE=model)),
            #         NOM_CHAM='UT02_ELGA',
            #         OPERATION='AFFE',
            #         TYPE_RESU='EVOL_ELAS')
            txt="$name = CREA_RESU(AFFE= $affeList , NOM_CHAM='UT02_ELGA',OPERATION='AFFE',TYPE_RESU='EVOL_ELAS')\n\n"
            txtLoop="_F(CARA_ELEM=elemprop, CHAM_GD= $critName , CHAM_MATER=fieldmat, INST=( $time , ), MODELE=model),\n"
            lst='('
            for j in range(nplies):
                t= 1.0 +0.001*j 
                lst=lst + Template(txtLoop).substitute( critName = 'FC_'+resnames[j] , time = str(t) )
            lst=lst+')'
            
            file.write(Template(txt).substitute(name='FC_'+str(idRegion), affeList=lst))
            shellFCresults.append('FC_'+str(idRegion))
        # end loop on shell regions
        
        # write shell failure criteria to output MED file
        if self.rootShellRegions.rowCount()>0:
            txt="IMPR_RESU( RESU= $affeList,  UNITE=2 ) \n\n"
            txtLoop="_F( RESULTAT= $resu ),\n"
            lst='('
            for resu in shellFCresults:
                lst=lst+Template(txtLoop).substitute(resu=resu)
            lst=lst+')'
            file.write(Template(txt).substitute(affeList=lst))
        
        # ASK user if detailed shell outputs must be saved
        qm = QMessageBox()
        ret = qm.question(None,'Detailed shell output', "Do you want to save detailed ply-by-ply shell normals stress & failure criteria?", qm.Yes | qm.No)
        if ret == qm.Yes:
            # save each ply data set to shell results file
            # IMPR_RESU(RESU=(_F(RESULTAT=R0_0),
            #                 ...
            #                 _F(RESULTAT=R1_4)),
            #           UNITE=2)
            txt="IMPR_RESU( RESU= $affeList,  UNITE=2 ) \n\n"
            txtLoop="_F( RESULTAT= $resu ),\n"
            lst='('
            for resu in shellPlyresults:
                lst=lst+Template(txtLoop).substitute(resu=resu)
            lst=lst+')'
            file.write(Template(txt).substitute(affeList=lst))
        
        # combine all shell failure criteria into a single "max criterion" field
        #  first compute the max of each failure criteria in each shell region
        # critR1 = CREA_CHAMP(AFFE_SP=_F(CARA_ELEM=elemprop),
        #                     NOM_CHAM='UT02_ELGA',
        #                     OPERATION='EXTR',
        #                     PROL_ZERO='OUI',
        #                     RESULTAT=FC_R1,
        #                     TYPE_CHAM='ELGA_NEUT_R',
        #                     TYPE_MAXI='MAXI_ABS')
        txt="$name = CREA_CHAMP(AFFE_SP=_F(CARA_ELEM=elemprop), NOM_CHAM='UT02_ELGA', OPERATION='EXTR', PROL_ZERO='OUI'," + \
                     "RESULTAT=  $FCResults ,TYPE_CHAM='ELGA_NEUT_R',TYPE_MAXI='MAXI_ABS')\n\n"
        for i in range(len(shellFCresults)):
            file.write(Template(txt).substitute(name='Crit'+str(i), FCResults=shellFCresults[i]))
        
        #  finally combine all FC data:
        # CritALL = CREA_CHAMP(ASSE=(_F(CHAM_GD=critR0,
        #                               GROUP_MA=('top', 'inside', 'sides', 'middle')),
        #                            _F(CHAM_GD=critR1,
        #                               GROUP_MA=('end', 'front'))),
        #                      MODELE=model,
        #                      OPERATION='ASSE',
        #                      PROL_ZERO='OUI',
        #                      TYPE_CHAM='ELGA_NEUT_R')
        if self.rootShellRegions.rowCount()>0:
            txt = "CritALL = CREA_CHAMP(ASSE= $affeList , MODELE=model, OPERATION='ASSE', PROL_ZERO='OUI', TYPE_CHAM='ELGA_NEUT_R')\n\n"
            txtLoop = "_F(CHAM_GD= $critXX , GROUP_MA= $groupList ),\n"     
            lst='('
            # loop on shell regions, note shellFCresults has same order as shellRegions in the model tree
            for i in range(len(shellFCresults)):   
                shellRegion = self.rootShellRegions.child(i)
                groupList=shellRegion.groupList
                lst=lst+Template(txtLoop).substitute(critXX='Crit'+str(i), groupList=str(tuple(groupList)))
            lst=lst+')'
            file.write(Template(txt).substitute(affeList=lst))
            # finally save this to MED file, only on shell elements to avoid display issues
            # IMPR_RESU(RESU=_F(CHAM_GD=CritALL,
            #                   GROUP_MA=('top', 'end', 'front', 'inside', 'sides', 'middle')),   
            #           UNITE=3)
            txt="IMPR_RESU(RESU=_F(CHAM_GD=CritALL, GROUP_MA= $groupList ),UNITE=3)\n\n"
            # extract list of shell groups
            grplist=[]
            for i in range(self.rootShellRegions.rowCount()):
                shellRegion=self.rootShellRegions.child(i)
                for grp in shellRegion.groupList:
                    grplist.append(grp)
            grplist=list(set(grplist)) # use a set to keep only unique values
            file.write(Template(txt).substitute(groupList=str(tuple(grplist))))   
        
        # post processing for solid regions
        
        #OLD SOLUTION => does not work when combined solid / shell model... 
        # resMod = MODI_REPERE(AFFE=(_F(ANGL_NAUT=(45.0, 0.0, 0.0),
        #                       GROUP_MA=('ply1', )),
        #                    _F(ANGL_NAUT=(90.0, 0.0, 0.0),
        #                       GROUP_MA=('ply2', )),
        #                    _F(ANGL_NAUT=(-45.0, 0.0, 0.0),
        #                       GROUP_MA=('ply3', ))),
        #              CARA_ELEM=elemprop,
        #              MODI_CHAM=_F(NOM_CHAM='SIEF_ELGA',
        #                           NOM_CMP=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
        #                           TYPE_CHAM='TENS_3D'),
        #              REPERE='UTILISATEUR',
        #              RESULTAT=result)
        # NEW solution: use custom formula for stress transformation and criteria evaluation
        
        #txt= "resMod = MODI_REPERE(AFFE= $affeList , CARA_ELEM=elemprop, " + \
        #            "MODI_CHAM=_F(NOM_CHAM='SIEF_ELGA',NOM_CMP=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),TYPE_CHAM='TENS_3D'),\n" + \
        #            "REPERE='UTILISATEUR', RESULTAT= reslin )\n\n"
        #txtLoop="_F(ANGL_NAUT=( $alpha ,  $beta ,  $gamma ), GROUP_MA= $grpList ),\n"
        #lst="("
        #if self.rootSolidPlies.rowCount()>0:
        #    for i in range(self.rootSolidPlies.rowCount()):
        #        solidPly=self.rootSolidPlies.child(i)
        #        grpList=solidPly.groupList
        #        orientName=solidPly.orientationName
        #        orient=getChildByName(self.rootOrientations, orientName)
        #        alpha=orient.angles["alpha"]
        #        beta=orient.angles["beta"]
        #        gamma=orient.angles["gamma"]
        #        lst=lst + Template(txtLoop).substitute(alpha=str(alpha), beta=str(beta), gamma=str(gamma), grpList=str(tuple(grpList)))
        #    lst=lst+")"
        #    file.write(Template(txt).substitute( affeList=lst ))
        
        # failure criteria for solid orthotropic or  isotropic materials (Von Mises, principal stresses, Hashin 3D)

        qm = QMessageBox()
        ret = qm.question(None,'Detailed Solid output', "Do you want to compute failure criteria for each solid regions?", qm.Yes | qm.No)
        if ret == qm.Yes:
            doWriteSolids=True
        else:
            doWriteSolids=False
            
        if (self.rootSolidPlies.rowCount()>0) & doWriteSolids:
         
            for i in range(self.rootSolidPlies.rowCount()):
                resultName='S'+str(i)
                solidPly = self.rootSolidPlies.child(i)
                grpList=solidPly.groupList
                materialName = solidPly.materialName
                orientName=solidPly.orientationName
                orient=getChildByName(self.rootOrientations,orientName)
                material = getChildByName(self.rootMaterials, materialName)
                st = material.strengthProperties
                angl=orient.angles
                
                # local stress component S11
                #S0_SI11 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI11(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI11=SI11,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI11(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI11=SI11,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI11'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                # local stress component S22
                #S0_SI22 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI22(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI22=SI22,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI22(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI22=SI22,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI22'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                # local stress component S33
                #S0_SI33 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI33(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI33=SI33,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI33(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI33=SI33,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI33'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                # local stress component S12
                #S0_SI12 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI12(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI12=SI12,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI12(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI12=SI12,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI12'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                # local stress component S13
                #S0_SI13 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI13(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI13=SI13,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI13(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI13=SI13,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI13'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                # local stress component S23
                #S0_SI23 = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='SI23(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI23=SI23,alpha=0.0,beta=0.0,gamma=0.0)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), \n VALE='SI23(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, alpha=alpha, beta=beta,gamma=gamma)', SI23=SI23,alpha=$alpha,beta=$beta,gamma=$gamma)\n\n"
                formulaName=resultName+'_SI23'
                file.write(Template(txt).substitute(name=formulaName, alpha=angl['alpha'], beta=angl['beta'] , gamma=angl['gamma']))
                

                # first criterion: fiber tensile , Hashin model 
                #S0_FT = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),
                #VALE='FT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, XT=XT, S12=S12, S13=S13)', FT=FT, XT=1000, S12=100, S13=70)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'),\n VALE='FT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, XT=XT, S12=S12, S13=S13)', FT=FT, XT=$XT, S12=$S12, S13=$S13)\n\n"
                formulaName=resultName+'_FT'
                file.write(Template(txt).substitute(name=formulaName, XT=st['XT'], S12=st['S12'] , S13=st['S13']))
                # 2nd criterion: fiber compression, Hashin model 
                #S0_FC = FORMULE(NOM_PARA=('SIXX', 'SIXY', 'SIXZ'), VALE='FC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, XC=XC)',FC=FC,XC=-600)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ','SIXY', 'SIXZ', 'SIYZ'), VALE='FC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, XC=XC)',FC=FC,XC=$XC)\n\n"
                formulaName=resultName+'_FC'
                file.write(Template(txt).substitute(name=formulaName, XC=st['XC'] ))
                # 3rd criterion: matrix tensile , Hashin model 
                # S0_MT = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ','SIXY', 'SIXZ', 'SIYZ'),VALE='MT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, YT=YT,ZT=ZT,S12=S12,S13=S13,S23=S23)',MT=MT,YT=$YT,ZT=$ZT,S12=$S12,S13=$S13,S23=$S23)
                txt="${name} = FORMULE(NOM_PARA=('SIXX', 'SIYY', 'SIZZ','SIXY', 'SIXZ', 'SIYZ'),VALE='MT(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, YT=YT,ZT=ZT,S12=S12,S13=S13,S23=S23)',MT=MT,YT=$YT,ZT=$ZT,S12=$S12,S13=$S13,S23=$S23) \n\n"
                formulaName=resultName+'_MT'
                file.write(Template(txt).substitute(name=formulaName,YT=st['YT'],ZT=st['ZT'], S12=st['S12'], S13=st['S13'], S23=st['S23']))
                # 4th criterion: matrix compression, Hashin model 
                # S0_MC = FORMULE(NOM_PARA=('SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), VALE='MC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, YC=YC,ZC=ZC,S12=S12,S13=S13,S23=S23)',MC=MC,YC=-120,ZC=-120,S12=45,S13=45,S23=35)

                txt="${name} = FORMULE(NOM_PARA=('SIYY', 'SIZZ', 'SIXY', 'SIXZ', 'SIYZ'), VALE='MC(SIXX, SIYY, SIZZ, SIXY, SIXZ, SIYZ, YC=YC,ZC=ZC,S12=S12,S13=S13,S23=S23)',MC=MC,YC=$YC,ZC=$ZC,S12=$S12,S13=$S13,S23=$S23) \n\n"
                formulaName=resultName+'_MC'
                file.write(Template(txt).substitute(name=formulaName, YC=abs(st['YC']),ZC=abs(st['ZC']), S23=st['S23'], S12=st['S12'], S13=st['S13']))
                # compute fields
                txt="#COMPUTE POST PRO DATA FOR SOLID REGION S0, USER DEFINED FIELD UT_01, COMPONENTS: X1..X6=local stress SI11..SI23, X7..X10=FT,FC,MT,MC criteria \n" 
                txt+="${name} = CALC_CHAMP(CHAM_UTIL=_F(FORMULE=($SI11,$SI22, $SI33, $SI12, $SI13, $SI23, $FT, $FC, $MT, $MC),\n"
                txt+="        NOM_CHAM='SIEF_ELGA', \n NUME_CHAM_RESU=1), CONTRAINTE=('SIGM_NOEU', 'SIGM_ELNO'), CRITERES=('SIEQ_ELGA', ), \n "
                txt+="        GROUP_MA=$grpList, RESULTAT=$inpname)\n \n" 
                #${name} = CALC_CHAMP( GROUP_MA= $grpList , CONTRAINTE=( 'SIGM_NOEU','SIGM_ELNO', ), CRITERES=('SIEQ_ELGA', ), CHAM_UTIL=_F(FORMULE=(${ft}, ${fc}, ${mt}, ${mc}), NOM_CHAM='SIEF_ELGA', NUME_CHAM_RESU=1), RESULTAT=${inpname})\n\n"
                file.write( Template(txt).substitute(SI11=resultName+'_SI11', SI22=resultName+'_SI22', SI33=resultName+'_SI33' , \
                                                     SI12=resultName+'_SI12', SI13=resultName+'_SI13', SI23=resultName+'_SI23' , \
                                                     name=resultName, FT=resultName+'_FT', FC=resultName+'_FC', \
                                                    MT=resultName+'_MT', MC=resultName+'_MC' , inpname='reslin', \
                                                    grpList=str(tuple(grpList))  ) )
            # end loop on aolid regions
            # now save results
            txt="IMPR_RESU(RESU=  $affeList  ,UNITE=84)\n"
            txtLoop= "    _F(RESULTAT= $name , GROUP_MA= $groupList ), \n"
            lst="("
            for i in range(self.rootSolidPlies.rowCount()):
                solidPly = self.rootSolidPlies.child(i)
                grpList=solidPly.groupList
                lst=lst+Template(txtLoop).substitute(name="S"+str(i), groupList=grpList)
            lst = lst + ' _F(RESULTAT=reslin,), \n'
            lst=lst+")"
            file.write(Template(txt).substitute(affeList=lst))
                
        # end post pro
        file.close()
        
            
                
## some helper functions
    
def getMaterialList(model):
    return getChildrenList(model,'Materials')
    
def getOrientationList(model):
    return getChildrenList(model,'Orientations')
            
def getSolidPliesList(model):
    return getChildrenList(model,'Solid Plies')

def getShellLayupsList(model):
    return getChildrenList(model,'Shell Layups')

def getShellRegionsList(model):
    return getChildrenList(model,'Shell Regions')

def getChildrenList(model,baseDescriptor=''):
    mylist=[]
    myroot=model.findItems(baseDescriptor,flags=Qt.MatchExactly,column=0)
    if len(myroot)==1:
        myroot=myroot[0]
    else:
        return ['',]
    for i in range(myroot.rowCount()):
        item=myroot.child(i)
        mylist.append(str(item.text()))
    return mylist

def getChildByName(item,childname):
    for i in range(item.rowCount()):
        if item.child(i).text()==childname:
            return item.child(i)
 

# classes for each type of object in the model tree, 
# each class provides its own GUI dialog (in self.ui) 
# that is activated via a call to the edit()


class MaterialItem(QStandardItem):
    def __init__(self,name):
        self.ui=None
        QStandardItem.__init__(self,name)
        self.elasticProperties={'E1':120000.,'E2':7000.,'E3':7000.,
                                'G12':3000.,'G13':3000.,'G23':3000.,
                                'NU12':0.35,'NU13':0.35,'NU23':0.4,
                                'RHO':1500e-12}
        self.strengthProperties={'XT':1500.,'XC':-800.,
                                 'YT':80.,'YC':-140.,
                                 'ZT':50.,'ZC':-140.,
                                 'S12':80., 'S13':50., 'S23':50.}
    def edit(self):
        uiFilePath=os.path.join(compdesignerbasepath,'materialForm.ui')
        self.ui=loadUi(uiFilePath)
        self.ui.setWindowTitle('Material properties: ' + self.text())
        self.updateUiFromData()
        self.ui.accepted.connect(self.updateDataFromUi)
        self.ui.show()

    
    def updateUiFromData(self):
        print('Update Material Ui:')
        if not(self.ui==None):
            u=self.ui
            s=self.strengthProperties
            e=self.elasticProperties
            u.lineE1.setText(str(e['E1']))
            u.lineE2.setText(str(e['E2']))
            u.lineE3.setText(str(e['E3']))
            u.lineG12.setText(str(e['G12']))
            u.lineG13.setText(str(e['G13']))
            u.lineG23.setText(str(e['G23']))
            u.lineNU12.setText(str(e['NU12']))
            u.lineNU13.setText(str(e['NU13']))
            u.lineNU23.setText(str(e['NU23']))
            u.lineRHO.setText(str(e['RHO']))
            u.lineXT.setText(str(s['XT']))
            u.lineXC.setText(str(s['XC']))
            u.lineYT.setText(str(s['YT']))
            u.lineYC.setText(str(s['YC']))
            u.lineZT.setText(str(s['ZT']))
            u.lineZC.setText(str(s['ZC']))
            u.lineS12.setText(str(s['S12']))
            u.lineS13.setText(str(s['S13']))
            u.lineS23.setText(str(s['S23']))
            
    def updateDataFromUi(self):
        print('Update Material data:')
        if not(self.ui==None):
            u=self.ui
            #try:
            E1=float(u.lineE1.text())
            E2=float(u.lineE2.text())
            E3=float(u.lineE3.text())
            G12=float(u.lineG12.text())
            G13=float(u.lineG13.text())
            G23=float(u.lineG23.text())
            NU12=float(u.lineNU12.text())
            NU13=float(u.lineNU13.text())
            NU23=float(u.lineNU23.text())
            RHO=float(u.lineRHO.text())
            #except:
            #    print("ERROR: Invalid Elastic property")
            #try:
            XT=float(u.lineXT.text())
            XC=float(u.lineXC.text())
            YT=float(u.lineYT.text())
            YC=float(u.lineYC.text())
            ZT=float(u.lineZT.text())
            ZC=float(u.lineZC.text())
            S12=float(u.lineS12.text())
            S13=float(u.lineS13.text())
            S23=float(u.lineS23.text())
            #except:
            #    print("ERROR: Invalid Strength property")
            self.elasticProperties={'E1':E1,'E2':E2,'E3':E3,
                                    'G12':G12,'G13':G13,'G23':G23,
                                    'NU12':NU12,'NU13':NU13,'NU23':NU23,
                                    'RHO':RHO}
            self.strengthProperties={'XT':XT,'XC':XC,
                                     'YT':YT,'YC':YC,
                                     'ZT':ZT,'ZC':ZC,
                                     'S12':S12, 'S13':S13, 'S23':S23}
            print('Material: ',self.text())
            print(self.elasticProperties)
            print(self.strengthProperties)
            
              
class OrientationItem(QStandardItem):
    def __init__(self,name):
        QStandardItem.__init__(self,name)
        self.angles={'alpha':0.,'beta':0.,'gamma':0.}
        self.vectors={'Vref':[1.,0.,0.],'Vnorm':[0.,0.,1.]}
        self.byAngles=False
        self.byVectors=True
        
    def edit(self):
        uiFilePath=os.path.join(compdesignerbasepath,'orientationForm.ui')
        self.ui=loadUi(uiFilePath)
        self.ui.setWindowTitle('Orientation properties: ' + self.text())
        self.updateUiFromData()
        self.ui.accepted.connect(self.updateDataFromUi)
        self.ui.show()

        
    def updateDataFromUi(self):
        print('Update Orientation data:')
        if not(self.ui==None):
            u=self.ui
            if self.ui.radioVectors.isChecked():
                self.byAngles=False
                self.byVectors=True
                V1x=float(u.V1x.text())
                V1y=float(u.V1y.text())
                V1z=float(u.V1z.text())
                # V2x=float(u.V2x.text())
                # V2y=float(u.V2y.text())
                # V2z=float(u.V2z.text())
                V3x=float(u.V3x.text())
                V3y=float(u.V3y.text())
                V3z=float(u.V3z.text())
                self.vectors={'Vref':[V1x,V1y,V1z],
                              'Vnorm':[V3x,V3y,V3z]}
                self.vectorsToAngles()
            if self.ui.radioAngles.isChecked():
                self.byAngles=True
                self.byVectors=False
                alpha=float(u.alphaLineEdit.text())
                beta=float(u.betaLineEdit.text())
                gamma=float(u.gammaLineEdit.text())
                self.angles={'alpha':alpha,'beta':beta,'gamma':gamma}
                self.anglesToVectors()
            print('Orientation: ', self.text())
            print(self.angles)
            print(self.vectors)
            
    def updateUiFromData(self):
        print('Update Orientation Ui:')
        if not(self.ui==None):
            u=self.ui
            
            u.radioAngles.setChecked(self.byAngles)
            u.radioVectors.setChecked(self.byVectors)
            u.alphaLineEdit.setText(str(self.angles['alpha']))
            u.betaLineEdit.setText(str(self.angles['beta']))
            u.gammaLineEdit.setText(str(self.angles['gamma']))

            u.V1x.setText(str(round(self.vectors['Vref'][0],4)))
            u.V1y.setText(str(round(self.vectors['Vref'][1],4)))
            u.V1z.setText(str(round(self.vectors['Vref'][2],4)))
            # u.V2x.setText(str(self.vectors['V2'][0]))
            # u.V2y.setText(str(self.vectors['V2'][1]))
            # u.V2z.setText(str(self.vectors['V2'][2]))
            u.V3x.setText(str(round(self.vectors['Vnorm'][0],4)))
            u.V3y.setText(str(round(self.vectors['Vnorm'][1],4)))
            u.V3z.setText(str(round(self.vectors['Vnorm'][2],4)))
        
    def anglesToVectors(self):
        alpha=self.angles['alpha']
        beta=self.angles['beta']
        gamma=self.angles['gamma']
        vx=[1.0,0.0,0.0]
        vy=[0.0,1.0,0.0]
        vz=[0.0,0.0,1.0]
        # nautical angles (see Code-Aster doc for AFFE_CARA_ELEM for example) X-Y-Z
        M=RotX(gamma)
        vx=prodMatVect(M,vx)
        vy=prodMatVect(M,vy)
        vz=prodMatVect(M,vz)
        M=RotY(-beta)
        vx=prodMatVect(M,vx)
        vy=prodMatVect(M,vy)
        vz=prodMatVect(M,vz)
        M=RotZ(alpha)
        vx=prodMatVect(M,vx)
        vy=prodMatVect(M,vy)
        vz=prodMatVect(M,vz)

        self.vectors['Vref']=vx
        self.vectors['Vnorm']=vz
    
    def vectorsToAngles(self):
        vref=self.vectors['Vref']
        vz=self.vectors['Vnorm']
        alpha=0.0
        beta=0.0
        gamma=0.0
        # first make sure Vx is perpendicular to Vnorm and unit length
        vx=diffVect(vref, multVect( dot(vref,vz) , vz) )
        vx=multVect( 1.0 / norm(vx) , vx)
        vz=multVect( 1.0 / norm(vz) , vz)
        vy=cross(vz, vx)
        # eqs from https://www.code-aster.org/V2/doc/v14/fr/man_u/u4/u4.74.01.pdf
        if abs(vx[0])<1e-14:
            alpha=0.0
        else:
            alpha=atan(vx[1]/vx[0])
        normxy=sqrt(vx[0]**2+vx[1]**2)
        if normxy<1e-14:
            beta=0.0
        else:
            beta = -atan(vx[2]/normxy)
        if abs(vy[1])<1e-14:
            gamma=0.0
        else:
            gamma=atan(vy[2]/vy[1])
        self.angles['alpha']=alpha*180/pi
        self.angles['beta']=beta*180/pi
        self.angles['gamma']=gamma*180/pi
        

     
class SolidPlyItem(QStandardItem):
    def __init__(self,name):
        QStandardItem.__init__(self,name)
        self.materialName=''
        self.orientationName=''
        self.groupList=[]
        
    def updateDataFromUi(self):
        print('Update Solid Ply GUI data')
        if not(self.ui==None):
            u=self.ui
            self.materialName=u.materialComboBox.currentText()
            self.orientationName=u.orientationComboBox.currentText()
            #try:
            txt=u.groupsTextEdit.toPlainText()
            #print('groups:',txt)
            txt.strip(' []()')
            lst=[]
            for item in txt.split(','):
                lst.append(item.strip(' \'\"()[]\n'))
            self.groupList=lst
            #except:
            #    print('error in group list formatting: names should be in \' \' separated by , or newline')
            #    print('current entry:', u.groupsTextEdit.toPlainText())
            print('Solid Ply: ', self.text())
            print(self.materialName)
            print(self.orientationName)
            print(self.groupList)
                
    def updateUiFromData(self):
        print('Update Solid Ply Ui:')
        if not(self.ui==None):
            u=self.ui
            txt=''
            for item in self.groupList:
                txt=txt + item + ','
            u.groupsTextEdit.setText(txt.strip(','))
            matlist=getMaterialList(self.model())
            u.materialComboBox.addItems(matlist)
            u.materialComboBox.setCurrentText(self.materialName)
            orientlist=getOrientationList(self.model())
            u.orientationComboBox.addItems(orientlist)
            u.orientationComboBox.setCurrentText(self.orientationName)
            
        
    def edit(self):
        uiFilePath=os.path.join(compdesignerbasepath,'solidPlyForm.ui')
        self.ui=loadUi(uiFilePath)
        self.ui.setWindowTitle('Solid ply properties: ' + self.text())
        self.updateUiFromData()
        self.ui.accepted.connect(self.updateDataFromUi)
        self.ui.show()

        
class ShellLayupItem(QStandardItem):
    def __init__(self,name):
        QStandardItem.__init__(self,name)
        self.plyList=[] # each ply is a list of materialName, thickness, angle
        
    def updateDataFromUi(self):
        print('Update Shell Layup data')
        self.plyList=[] # each ply is a list of materialName, thickness, angle
       
        if not(self.ui==None):
            table=self.ui.layupTableWidget
            for i in range(table.rowCount()):
                data=self.getRowData(i)
                self.plyList.append(data)
            print('Shell Layup: ', self.text())
            print(self.plyList)
                
    def updateUiFromData(self):
        print('Update Shell Layup Ui')
        if not(self.ui==None):
            for i in range(len(self.plyList)):
                data=self.plyList[i]
                self.setupRow(i,data)
            
        
    def edit(self):
        uiFilePath=os.path.join(compdesignerbasepath,'shellLayupForm.ui')
        self.ui=loadUi(uiFilePath)
        self.ui.setWindowTitle('Shell Layup editor: ' + self.text())
        self.ui.addButton.clicked.connect(self.addPly) # setting up UI interactions
        self.ui.delButton.clicked.connect(self.delPly)
        self.ui.repeatButton.clicked.connect(self.repeatBlock)
        self.ui.symButton.clicked.connect(self.symBlock)
        self.updateUiFromData()
        self.ui.accepted.connect(self.updateDataFromUi)
        self.ui.show()

        
    def addPly(self):
        if not(self.ui==None):
            table=self.ui.layupTableWidget
            select=table.selectedRanges()
            if len(select)==0:
                bottom=table.rowCount()-1
            else:
                bottom=select[0].bottomRow()
            table.insertRow(bottom+1)
            self.setupRow(bottom+1)
            
    def setupRow(self,row,data=[]):
        if not(self.ui==None):
            if len(data)==0:
                data.append('') # no material
                data.append(0.0) # thickness
                data.append(0.0) # angle
            table=self.ui.layupTableWidget
            if (row+1)>table.rowCount():
                table.setRowCount(row+1)  #row index starts at 0 !
            # set item data
            table.setItem(row,0,QTableWidgetItem(str(data[0])))
            table.setItem(row,1,QTableWidgetItem(str(data[1])))
            table.setItem(row,2,QTableWidgetItem(str(data[2])))
            # add combobox
            comboMat=QComboBox()
            matlist=getMaterialList(self.model())
            comboMat.addItems(matlist)
            comboMat.setCurrentText(data[0])
            table.setCellWidget(row,0,comboMat)
            
    def getRowData(self,row):
        if not(self.ui==None):
            table=self.ui.layupTableWidget
            if row>=table.rowCount():
                return []
            data=[]
            matwidget=table.cellWidget(row,0)
            data.append(matwidget.currentText()) #material
            data.append(float(table.item(row,1).text())) #thickness
            data.append(float(table.item(row,2).text())) #angle
            return data
            
    def delPly(self):
        if not(self.ui==None):
            table=self.ui.layupTableWidget
            select=table.selectedRanges()
            if len(select)==0:
                bottom=table.rowCount()-1
            else:
                bottom=select[0].bottomRow()
            table.removeRow(bottom)
            
    def repeatBlock(self):
        pass
    
    def symBlock(self):
        pass
        
    
    
class ShellRegionItem(QStandardItem):
    def __init__(self,name):
        QStandardItem.__init__(self,name)
        self.layupName=''
        self.orientationName=''
        self.groupList=[]
        
    def updateDataFromUi(self):
        print('Update Shell region GUI data')
        if not(self.ui==None):
            u=self.ui
            self.layupName=u.layupComboBox.currentText()
            self.orientationName=u.orientationComboBox.currentText()
            #try:
            txt=u.groupsTextEdit.toPlainText()
            #print('groups:',txt)
            txt.strip(' []()')
            lst=[]
            for item in txt.split(','):
                lst.append(item.strip(' \'\"()[]\n'))
            self.groupList=lst
            #except:
            #    print('error in group list formatting: names should be in \' \' separated by , or newline')
            #    print('current entry:', u.groupsTextEdit.toPlainText())
            print('Shell region: ', self.text())
            print(self.layupName)
            print(self.orientationName)
            print(self.groupList)
                
    def updateUiFromData(self):
        print('Update Shell region Ui:')
        if not(self.ui==None):
            u=self.ui
            txt=''
            for item in self.groupList:
                txt=txt + item + ','
            u.groupsTextEdit.setText(txt.strip(','))
            layuplist=getShellLayupsList(self.model())
            u.layupComboBox.addItems(layuplist)
            u.layupComboBox.setCurrentText(self.layupName)
            orientlist=getOrientationList(self.model())
            u.orientationComboBox.addItems(orientlist)
            u.orientationComboBox.setCurrentText(self.orientationName)
            
        
    def edit(self):
        uiFilePath=os.path.join(compdesignerbasepath,'shellRegionForm.ui')
        self.ui=loadUi(uiFilePath)
        self.ui.setWindowTitle('Shell region properties: ' + self.text())
        self.updateUiFromData()
        self.ui.accepted.connect(self.updateDataFromUi)
        self.ui.show()

# --- MAIN GUI ---

class CompositeDesignerGUI(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('Composite Designer')
        self.resize(500,700)
        self.treeView = QTreeView()
        self.treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.openMenu)
        
        self.model = CompositeModel()
        self.treeView.setModel(self.model)
        
        self.model.setHorizontalHeaderLabels([self.tr("Objects")])
        
        layout = QVBoxLayout()
        layout.addWidget(self.treeView)
        self.setLayout(layout)
    
    def openMenu(self, position):
        # get selection and its attributes
        indexes = self.treeView.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            selectedItem, parentName, parentRow, selectName, \
                selectRow, level = self.model.getInfoFromIndex(index)
        # generate context menu based on selection            
        menu = QMenu()
        if level==0:
            if selectName=='Materials':
                action=menu.addAction(self.tr("Add Material"))
                action.triggered.connect(self.model.addMaterial)
            elif selectName=='Orientations':
                action=menu.addAction(self.tr("Add Orientation"))
                action.triggered.connect(self.model.addOrientation)
            elif selectName=='Solid Plies':
                action=menu.addAction(self.tr("Add Solid Ply"))
                action.triggered.connect(self.model.addSolidPly)
            elif selectName=='Shell Layups':
                action=menu.addAction(self.tr("Add Shell Layup"))
                action.triggered.connect(self.model.addShellLayup)
            elif selectName=='Shell Regions':
                action=menu.addAction(self.tr("Add Shell Region"))
                action.triggered.connect(self.model.addShellRegion)
            elif selectName=='Simulation':
                action=menu.addAction(self.tr("Load composite design"))
                action.triggered.connect(self.model.loadModelData)            
                action=menu.addAction(self.tr("Save composite design"))
                action.triggered.connect(self.model.saveModelData)
                action=menu.addAction(self.tr("Export Code-Aster Model"))
                action.triggered.connect(self.model.exportCodeAsterModel)
                action=menu.addAction(self.tr("Export Code-Aster PostPro"))
                action.triggered.connect(self.model.exportCodeAsterPostPro)
        elif level == 1:
            if parentName=='Materials':
                action=menu.addAction(self.tr("Edit Material"))
                action.triggered.connect(selectedItem.edit)
            elif parentName=='Orientations':
                action=menu.addAction(self.tr("Edit Orientation"))
                action.triggered.connect(selectedItem.edit)
            elif parentName=='Solid Plies':
                action=menu.addAction(self.tr("Edit Solid Ply"))
                action.triggered.connect(selectedItem.edit)
            elif parentName=='Shell Layups':
                action=menu.addAction(self.tr("Edit Shell Layup"))
                action.triggered.connect(selectedItem.edit)
            elif parentName=='Shell Regions':
                action=menu.addAction(self.tr("Edit Shell Region"))
                action.triggered.connect(selectedItem.edit)
            action=menu.addAction(self.tr("Delete"))
            action.triggered.connect(lambda: self.model.deleteItem(selectedItem))
        elif level == 2:
            #menu.addAction(self.tr("Edit object"))
            print("warning, tree level 2 should not exist...")
            pass
        menu.exec_(self.treeView.viewport().mapToGlobal(position))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = CompositeDesignerGUI()
    window.show()
    sys.exit(app.exec_())
# CompositeDesigner
Composite designer plugin for Salome-Meca / Code-Aster

Development: Joël Cugnoni, HEIG-VD

License: GPL v2
(contact author for other options)

# Installation
To install the code, place the files in your Salome-Meca plugin directory, which is typically located the /home/yourusername/.config/salome/Plugins. If the Plugins folder does not exist, just create it. For Windows, the same rule applies , put these files for example in c:\users\yourusername\.config\salome\Plugins.

Then edit or create the file "salome_plugins.py" to include the following line of codes:
```
import salome_pluginsmanager
import sys
import os

## composite designer plugin HEIG-VD 2022

basepath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,basepath)
print("Importing COMPOSITE DESIGNER PLUGIN")
import CompositeDesigner
compositeDesignerDialog = CompositeDesigner.CompositeDesignerGUI()

def compositeDesignerGUI(context):
    global compositeDesignerDialog 
    compositeDesignerDialog.show()

salome_pluginsmanager.AddFunction('Composite Designer','Composite designer app',compositeDesignerGUI)
```

# Usage of Composite Designer in Salome-Meca

## Features
* Composite Designer is an helper tool to build multilayer shell and/or 3D solid composite structure models.
* Based on a simple set of GUI input forms, the user can define orthotropic materials (elasticity and failure properties), define material orientations, composite shell layups and assign materials & orientations to groups of elements. * The FE mesh needs to be generated separatelly in Salome and must contain groups of 2D triangular/quadrangle elements (1st order/linear) for shell regions and groups of 3D elements for solid regions.
* Each regions corresponds to a given material and orientation definition. Thus the user needs to partition into groups corresponding to each region in advance.
* The plugin only generates a template of Code-Aster command file that defines a basic simulation. The generated command file can be imported in AsterStudy and edited for example to define proper load case definitions. For the moment the plugin only generate a linear elastic stress analysis simulation. The type of analysis can be edited manually in AsterStudy but please noted that the post-processing scripts may not work for other type of study.
* The plugin also generate a post processing command file (second stage) that is specific to the model. This script computes local stresses in each ply or solid regions and then computes Hashin failure criteria. Please note that the model and postprocessing command files must be regenerated from Composite Designer after each modification to the material/layup orientation modification.


## GUI

Composite Designer plugin is a simple GUI structured around a model tree (materials, orientations, layups, shell / solid regions, ..). Once the salome_plugins.py configuration (see above) has been done, the plugin should be available in Salome-Meca plugins menu. Note that at the present time Composite Designer can also be exexuted as a standalone Python application as it does not interact yet with Salome.

## Interactions
* The basic interaction is done by Rigth Click on the tree items to open their context menu.
* You can also double click an item (material for example) to rename it.
* Right click the Simulation item in the tree to load/save the Composite Designer data (.json file) or generate Code-Aster simulation & post-processing files.

## Workflow
1. To create a 3D model of the part. For shell only models, you can extract the shell and select faces to keep using the Explode tool for example. Please note that Salome always generate surface mesh elements even for solids, but those elements are not automatically assigned as shell elements in the simulation. Thus you can easily create a sandwich structure using a 3D solid model of the core and assign shell elements to the top and bottom surfaces.
2. Create groups (in Geometry or later in Mesh) for each shell or solid region. A region corresponds to a given combination of material/layup and orientation. Create groups of Faces for shells and of Volumes for solids. Note that group names must be at most 8 character long (case sensitive) and cannot start by a number. Also create groups in advance to model the load cases / supports.
3. Generate a mesh of the model and define/check the groups. The mesh of the surface elements must be 1st order. The 3D elements can be either 1st order (simpler but less accurate) or quadratic (more complex to combine with linear shells).
4. Start Composite Designer plugins from the plugin menu or as standalone app.
5. Define orthotropic materials (elasticity and failure) by right click on the Material tree item.
6. define local material orientations with right-click on Orientations tree item (vectors components or nautic angles, see Code-Aster AFFE_CARA_ELEM documentation).
7. continue by defining composite layups if you intend to model a shell.
8. assign layups (again right click...). Choose layup name, orientation and then enter a list of group names (comma separated) that define a shell region (same layup and orientation, but ca be disjoint). Double check the spelling of the group names tommake sure inlt matches those defined in the mesh (case sensitive).
9. same procedure for solid regions.
10. Right click on Simulation and save Composite Design to a json file (to be able to reload later). Right click again and save Code-Aster model and post pro command files (.comm).
11. in AsterStudy, create a new study, import the model .comm file as first stage (graphical) and import the post pro .comm file as second stage (import stage as text).
12. edit the load case (at least) in AFFE_CHAR_MECA. Optionnally you may neef to add surface elements groups for distributed loading on solid regions in AFFE_MODELE , under the '3D' modeling assignment list (surface regions of 3D elements need to be assigned a 3D modeling to define distributed loading).
13. Assign files to the following units (TODO, see example in zip file for the moment).
14. Save the study, and run the simulation
14. Post process the results by loading the resul med files in Paraviz. see post processing doc for the definition / naming of the fields. Also, many fields are computed at integration point of the elements (ELGA fields, at Gauss points). Those fields can be visualized using one of the following filters in Paraviz: Mechanics-> ELGA field to surface or to surface (avg by element) or ELGA to Point Gaussian (and choose Point Gaussian as visualization mode)
 

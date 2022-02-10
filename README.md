# CompositeDesigner
Composite designer plugin for Salome-Meca / Code-Aster

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

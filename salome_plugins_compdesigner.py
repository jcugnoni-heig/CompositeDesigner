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

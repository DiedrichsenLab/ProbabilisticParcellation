"""
Scripts to update models from GPU -> CPU 
old symmetric -> new symmetric
"""
from time import gmtime
from pathlib import Path
from ProbabilisticParcellation.util import *
import generativeMRF.full_model as fm
import glob

pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:\data\Cerebellum\ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

def move_to_cpu():
    fname = glob.glob(model_dir + '/Models/Models_04/*.pickle')
    for f in fname:
        print(f)
        with open(f, 'rb') as file:
            models = pickle.load(file)

        # Recursively tensors to device
        for m in models:
            m.move_to(device='cpu')

        with open(f, 'wb') as file:
            pickle.dump(models, file)

def update_symmetric():
    fname = glob.glob(model_dir + '/Models/Models_03/sym_MdPoNiIbWmDeSo*.pickle')
    for f in fname:
        print(f)
        with open(f, 'rb') as file:
            models = pickle.load(file)

        # Recursively update the models 
        new_models = []
        for m in models:
            nm = fm.update_symmetric_model(m)
            new_models.append(nm)

        with open(f, 'wb') as file:
            pickle.dump(new_models, file)
    pass

if __name__=='__main__':
    update_symmetric()
    pass
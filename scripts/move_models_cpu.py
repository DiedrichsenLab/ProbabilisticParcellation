# Evaluate cerebellar parcellations
from time import gmtime
from pathlib import Path
from ProbabilisticParcellation.util import *
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

# Ks = [34]
# for K in Ks:

    
# # Evaluate DCBC
# eval_all_dcbc(model_type=model_type,prefix=sym,K=K,space = 'MNISymC3', models=hcp_models, fname_suffix=fname_suffix)

# Concat DCBC
# concat_all_prederror(model_type=model_type,prefix=sym,K=Ks,outfile=fname_suffix)

if __name__=='__main__':
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

<<<<<<< Updated upstream
    # move_batch_to_device('Models_04/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-34','cpu')
=======
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
>>>>>>> Stashed changes
    pass
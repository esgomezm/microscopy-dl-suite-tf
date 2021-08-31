"""
Created on Tue Apr 7 2020

@author: E. Gómez de Mariscal
GitHub username: esgomezm

Example of config.json:
{
 "base_lr":	0.0001,
 "type":	"Adam",
 "momentum":	0.99,
 "momentum2":	0.999,
 "gamma":		0.1,
 "max_epochs":	1000,
 "save_freq":	"epoch", # save_freq: either 'epoch' or an integer number (frequency in number of samples seen) (look for callbacks.ModelCheckpoint info)
 "patience":	5, # patience: number of epochs to reduce the learning rate (look for callbacks.ReduceLROnPlateau info)
 "tb_update_freq": 10, # Tensorboard update frequency
 "patch_batch":	1,
 "batch_size":	1,
 "dim_size":	[256,256],
 "OUTPUTPATH":	"binary_dice_weighted",
 "TRAINPATH":	"../data/train/stack2im",
 "TESTPATH":	"../data/train/stack2im"
}
"""



import json

class Dict2Obj():
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self,path2config):
        with open(path2config) as json_file:
            config_dict = json.load(json_file)
        """Constructor"""
        for key in config_dict:
            setattr(self, key, config_dict[key])
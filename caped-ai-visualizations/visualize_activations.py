import os
from oneat.NEATModels.loss import volume_yolo_loss, static_yolo_loss, dynamic_yolo_loss
from vollseg import CARE, UNET, StarDist2D, StarDist3D, MASKUNET
import numpy as np
from oneat.NEATUtils.utils import load_json, normalizeFloatZeroOne
from keras import models 
from keras.models import load_model
from tifffile import imread


class visualize_activations(object):
    
    def __init__(self,config, catconfig, cordconfig, model_dir, model_name, imagename, oneat_vollnet = False,
                 oneat_lrnet = False, oneat_tresnet = False, oneat_resnet = False, voll_starnet_2D = False,
                 voll_starnet_3D = False, voll_unet = False, voll_care = False, layer_viz_start = None,
                 layer_viz_end = None, dtype = np.uint8, n_tiles = (1,1,1), normalize = True):
        
        self.config = config 
        self.model_dir = model_dir 
        self.model_name = model_name
        self.imagename = imagename 
        self.oneat_vollnet = oneat_vollnet 
        self.oneat_lrnet = oneat_lrnet 
        self.oneat_tresnet = oneat_tresnet 
        self.oneat_resnet = oneat_resnet
        self.voll_starnet_2D = voll_starnet_2D 
        self.voll_starnet_3D = voll_starnet_3D
        self.voll_net = voll_unet 
        self.voll_care = voll_care 
        self.catconfig = catconfig 
        self.cordconfig = cordconfig 
        self.layer_viz_start = layer_viz_start
        self.layer_viz_end  = layer_viz_end 
        self.dtype = dtype 
        self.n_tiles = n_tiles 
        self.normalize = normalize
        self.key_cord = self.cordconfig
        self.categories = len(self.catconfig)
        self.key_categories = self.catconfig
        if self.oneat_vollnet or self.oneat_lstmnet or self.oneat_cnnnet or self.oneat_staticnet: 
                self.config = load_json(os.path.join(self.model_dir, self.model_name) + '_Parameter.json')
                
                self.box_vector = self.config['box_vector']
                self.show = self.config['show']
                
                self.depth = self.config['depth']
                self.start_kernel = self.config['start_kernel']
                self.mid_kernel = self.config['mid_kernel']
                self.learning_rate = self.config['learning_rate']
                self.epochs = self.config['epochs']
                self.startfilter = self.config['startfilter']
                self.batch_size = self.config['batch_size']
                self.multievent = self.config['multievent']
                self.imagex = self.config['imagex']
                self.imagey = self.config['imagey']
                self.imagez = self.config['imagez']
                self.imaget = self.config['size_tminus'] + self.config['size_tplus'] + 1
                self.size_tminus = self.config['size_tminus']
                self.size_tplus = self.config['size_tplus']
                self.nboxes = self.config['nboxes']
                self.stage_number = self.config['stage_number']
                self.last_conv_factor = 2 ** (self.stage_number - 1)
                self.gridx = 1
                self.gridy = 1
                self.gridz = 1
                self.yolo_v0 = self.config['yolo_v0']
                self.yolo_v1 = self.config['yolo_v1']
                self.yolo_v2 = self.config['yolo_v2']
                self.stride = self.config['stride']
                if self.multievent == True:
                        self.entropy = 'binary'

                if self.multievent == False:
                    self.entropy = 'notbinary' 
        
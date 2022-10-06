import os
from oneat.NEATModels.loss import volume_yolo_loss, static_yolo_loss, dynamic_yolo_loss
from oneat.NEATModels.neat_vollnet import NEATVollNet
from vollseg import CARE, UNET, StarDist2D, StarDist3D, MASKUNET
import numpy as np
from oneat.NEATUtils.utils import load_json, normalizeFloatZeroOne
from keras import models 
from keras.models import load_model
from tifffile import imread
import napari 
from oneat.NEATModels.nets import Concat
import tensorflow as tf
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
        self.voll_unet = voll_unet 
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
        self.image = imread(imagename).astype(self.dtype)
        self.viewer = napari.Viewer()   
        self.all_max_activations = []
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
        
    
        
    def _load_model_losses(self):
        
        if self.normalize: 
            self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
        self.image = np.expand_dims(self.image, 0)    
            
        if self.oneat_vollnet: 
            
            self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1])  
            self.yololoss = volume_yolo_loss(self.categories, self.gridx, self.gridy, self.gridz, self.nboxes,
                                            self.box_vector, self.entropy)
        
        if self.oneat_tresnet:
            self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
            self.yololoss = static_yolo_loss(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector,
                                                        self.entropy)
        
        if self.oneat_lrnet:
            self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
            self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridt, self.nboxes,
                                          self.box_vector, self.entropy)

        if self.oneat_resnet:
            self.pad_width = (self.image.shape[-2], self.image.shape[-1]) 
            self.yololoss = static_yolo_loss(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector,
                                                        self.entropy)

        
        
         
        
        
        if self.oneat_vollnet:
             self.model = NEATVollNet(None, model_dir = self.model_dir, model_name = self.model_name)
             self.prediction_oneat = VollN
                    
        elif self.voll_starnet_2D:
                if len(self.image.shape) == 4:
                    self.image = self.image[0,0,:,:]
                if len(self.image.shape) == 3:
                    self.image = self.image[0,:,:]     
                self.pad_width = (self.image.shape[-2], self.image.shape[-1]) 
                self.model =  StarDist2D(None, name=self.model_name, basedir=self.model_dir)._build()
                self.prediction_star = self.model.predict(self.image)         
        elif self.voll_starnet_3D:
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                self.model =  StarDist3D(None, name=self.model_name, basedir=self.model_dir)._build()
                self.prediction_star = self.model.predict(self.image)     
        elif self.voll_unet:
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                if len(self.image.shape) >=3:
                     self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                else:
                     self.pad_width = (self.image.shape[-2], self.image.shape[-1])      
                self.model =  UNET(None, name=self.model_name, basedir=self.model_dir)._build()  
                self.prediction_unet = self.model.predict(self.image)
        elif self.voll_care:
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                if len(self.image.shape) >=3:
                     self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                else:
                     self.pad_width = (self.image.shape[-2], self.image.shape[-1])
                self.model =  CARE(None, name=self.model_name, basedir=self.model_dir)._build()
                self.prediction_care = self.model.predict(self.image)
                
                
    def _activations_predictions(self):
         
         
        if self.layer_viz_start is None:
            self.layer_viz_start = 0 
        if self.layer_viz_end is None:
            self.layer_viz_end = len(self.activations)
                
        if self.layer_viz_start < 0:
             self.layer_viz_start = len(self.activations) + self.layer_viz_start    
        if self.layer_viz_end < 0:
             self.layer_viz_end = len(self.activations) + self.layer_viz_end
         
            
        layer_outputs = [layer.output for layer in self.model.layers[self.layer_viz_start:self.layer_viz_end]]
        self.activation_model = models.Model(inputs= self.model.input, outputs=layer_outputs)   
         
        if self.oneat_vollnet:
             
            self.image = tf.reshape(self.image, (self.image.shape[0], self.image.shape[2], self.image.shape[3],self.image.shape[4], self.image.shape[1]))
                 
        self.activations = self.activation_model.predict(self.smallimage)
                 
import os
from oneat.NEATModels.loss import volume_yolo_loss, static_yolo_loss, dynamic_yolo_loss
from oneat.NEATModels.neat_vollnet import NEATVollNet, CreateVolume
from oneat.NEATModels.neat_lstm import NEATLRNet

from oneat.NEATModels.neat_dynamic_resnet import NEATTResNet
from oneat.NEATModels.neat_static_resnet import NEATResNet
from vollseg import CARE, UNET, StarDist2D, StarDist3D, MASKUNET
import numpy as np
from oneat.NEATUtils.utils import load_json, normalizeFloatZeroOne
from keras import models 
from keras.models import load_model
from tifffile import imread
from oneat.NEATModels.nets import Concat
import tensorflow as tf
import napari
from .visualize_action_volume_boxes import VisualizeBoxes
class visualize_activations(object):
    
    def __init__(self,  
                 catconfig: dict, 
                 cordconfig: dict, 
                 model_dir: str,  
                 image: np.ndarray, 
                 model_name: str = None,
                 segimage: np.ndarray = None, 
                 visualize_point: int = None, 
                 oneat_vollnet: bool = False, 
                 start_project_mid: int = 4, 
                 end_project_mid: int = 1,
                 oneat_lrnet: bool = False, 
                 oneat_tresnet: bool = False, 
                 oneat_resnet: bool = False, 
                 voll_starnet_2D: bool = False,
                 voll_starnet_3D: bool = False, 
                 voll_unet: bool = False, 
                 voll_care: bool = False, 
                 layer_viz_start: int = None,
                 event_threshold: float = 0.9, 
                 event_confidence: float = 0.9, 
                 nms_function: str = 'iou',
                 layer_viz_end: int = None, 
                 dtype: np.dtype = np.uint8, 
                 n_tiles: tuple = (1,1,1), 
                 normalize: bool = True):
        
        self.viewer = napari.Viewer()
        self.model_dir = model_dir 
        self.image = image
        self.model_name = model_name
        self.segimage = segimage
        self.start_project_mid = start_project_mid
        self.end_project_mid = end_project_mid
        self.event_threshold = event_threshold 
        self.event_confidence = event_confidence
        self.nms_function = nms_function  
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
        self.visualize_point = visualize_point
        self.all_max_activations = {}
        if self.oneat_vollnet or self.oneat_lrnet or self.oneat_tresnet or self.oneat_resnet: 
                self.config = load_json(os.path.join(self.model_dir, 'parameters.json'))
                
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
        
    def _load_model_loss(self):
            
            
            
        if self.oneat_vollnet: 
            
            self.yololoss = volume_yolo_loss(self.categories, self.gridx, self.gridy, self.gridz, self.nboxes,
                                            self.box_vector, self.entropy)
            self.model = NEATVollNet(None, self.model_dir, self.catconfig, self.cordconfig)
            
        
        if self.oneat_tresnet:
            self.yololoss = static_yolo_loss(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector,
                                                        self.entropy)
            self.model = NEATTResNet(None, self.model_dir, self.catconfig, self.cordconfig)
            
        
        if self.oneat_lrnet:
            self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, 1, self.nboxes,
                                          self.box_vector, self.entropy)
            self.model = NEATLRNet(None, self.model_dir,self.catconfig, self.cordconfig)
            

        if self.oneat_resnet:
            self.yololoss = static_yolo_loss(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector,
                                                        self.entropy)
            self.model = NEATResNet(None, self.model_dir ,  self.catconfig, self.cordconfig)
            
     
        elif self.voll_starnet_2D:
            
                
        
                if self.normalize: 
                    self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
                if len(self.image.shape) == 4:
                    self.image = self.image[0,0,:,:]
                if len(self.image.shape) == 3:
                    self.image = self.image[0,:,:]     
                self.pad_width = (self.image.shape[-2], self.image.shape[-1]) 
                self.model =  StarDist2D(None, name=self.model_name, basedir=self.model_dir)
        elif self.voll_starnet_3D:
            
        
                if self.normalize: 
                    self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                self.model =  StarDist3D(None, name=self.model_name, basedir=self.model_dir)
                     
        elif self.voll_unet:
                 
        
                if self.normalize: 
                    self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
                
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                if len(self.image.shape) >=3:
                     self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                else:
                     self.pad_width = (self.image.shape[-2], self.image.shape[-1])      
                self.model =  UNET(None, name=self.model_name, basedir=self.model_dir)  
                
        elif self.voll_care:
            
        
                if self.normalize: 
                    self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
            
                if len(self.image.shape) == 4:
                    self.image = self.image[0,:,:,:]
                if len(self.image.shape) >=3:
                     self.pad_width = (self.image.shape[-3], self.image.shape[-2], self.image.shape[-1]) 
                else:
                     self.pad_width = (self.image.shape[-2], self.image.shape[-1])
                self.model =  CARE(None, name=self.model_name, basedir=self.model_dir)
                
                
                
    def _model_prediction(self):
        
        if isinstance(self.model, NEATVollNet):
             #Load the prediction for VollNet
             marker_tree =  self.model.get_markers(self.segimage)
             self.model.predict(self.image,
                           n_tiles = self.n_tiles, 
                           event_threshold = self.event_threshold, 
                           event_confidence = self.event_confidence,
                           marker_tree = marker_tree, 
                           nms_function = self.nms_function,
                           normalize = self.normalize, 
                           activations = True)
        if isinstance(self.model, NEATLRNet):
             #Load the prediction for LRNet
             marker_tree =  self.model.get_markers(self.segimage,
                                                start_project_mid = self.start_project_mid,
                                                end_project_mid = self.end_project_mid,  
                                                ) 
             self.model.predict(self.image,
                               n_tiles = self.n_tiles,
                               event_threshold = self.event_threshold,
                               event_confidence = self.event_confidence,
                               marker_tree = marker_tree,
                               nms_function = self.nms_function,
                               start_project_mid = self.start_project_mid,
                               end_project_mid = self.end_project_mid,
                               normalize = self.normalize, 
                               activations = True)
        if isinstance(self.model, NEATTResNet):
             #Load the prediction for TResNet  
             marker_tree = self.model.get_markers( self.segimage, 
                                                  start_project_mid = self.start_project_mid,
                                                  end_project_mid = self.end_project_mid)
             self.model.predict(self.image,
                                n_tiles = self.n_tiles,
                                event_threshold = self.event_threshold,
                                event_confidence = self.event_confidence,
                                marker_tree = marker_tree,
                                nms_function = self.nms_function,
                                stert_project_mid = self.start_project_mid,
                                end_project_mid = self.end_project_mid,
                                normalze = self.normalize, 
                                activations = True)
        if isinstance(self.model, NEATResNet):
             #Load the prediction for ResNet
             self.model.predict(self.image,
                               event_threshold = self.event_threshold,
                               event_confidence = self.event_confidence,
                               n_tiles = self.n_tiles, 
                               activations = True
                               )
        if isinstance(self.model, StarDist3D or StarDist2D):
            #Load the prediction for StarDist3D
            self.prediction_star = self.model.predict(self.image)
        if isinstance(self.model, CARE):
            #Load the prediction for CARE
            self.prediction_care = self.model.predict(self.image)
        if isinstance(self.model, UNET):
            #Load the predicton for UNET
            self.prediction_unet = self.model.predict(self.image)                                
                        
                
    def _draw_boxes(self):    
        
        if isinstance(self.model, any(NEATVollNet, NEATTResNet, NEATLRNet, NEATResNet, NEATTResNet)):
          
           viz_box = VisualizeBoxes(viewer = self.viewer, key_categories = self.key_categories, event_threshold = self.event_threshold)
        
        if self.oneat_vollnet:
             
             viz_box.create_volume_boxes(iou_classedboxes = self.model.all_iou_classedboxes)
             
        if self.oneat_lrnet:
            
            viz_box.create_volume_boxes(iou_classedboxes = self.model.all_iou_classedboxes, volumetric = False, shape = self.model.image.shape)
            
        if self.oneat_tresnet:
            
            viz_box.create_volume_boxes(iou_classedboxes = self.model.all_iou_classedboxes, volumetric = False, shape = self.model.image.shape)
            
        if self.oneat_resnet:
            
            viz_box.create_area_boxes(iou_classedboxes = self.model.all_iou_classedboxes)           
              
                
                
    def _activations_predictions(self):
         
                
        self.model = self.model._build()  
        
        self.max_activation_layer = len(self.model.layers)
        if self.layer_viz_start is None:
            self.layer_viz_start = 0 
        if self.layer_viz_end is None:
            self.layer_viz_end = self.max_activation_layer
        
        if self.visualize_point is not None:
            if self.visualize_point < self.size_tminus:
                self.visualize_point = self.size_tminus + 1
                
            print(self.max_activation_layer)    
            smallimage = CreateVolume(self.image, self.size_tminus, self.size_tplus, self.visualize_point)
            smallimage = np.expand_dims(smallimage,0) 
            layer_outputs = [layer.output for layer in self.model.layers[self.layer_viz_start:self.layer_viz_end]]
            self.activation_model = models.Model(inputs = self.model.input, outputs=layer_outputs)   
             
            
            if self.oneat_vollnet:
                
                smallimage = np.reshape(smallimage, (smallimage.shape[0], smallimage.shape[2], smallimage.shape[3],smallimage.shape[4], smallimage.shape[1]))
            
            
            self.activations = self.activation_model.predict(smallimage)
            self.all_max_activations[self.visualize_point] = self.activations
           
    def VizualizePredictionsActivations(self):    
        print('loading model and losses')
        self._load_model_loss()
        print('running predictions')
        self._model_prediction()
        print('prediction done, creating boxes')
        self._draw_boxes()
        print('boxes done, creating activation maps')
        self._activations_predictions()
        
        
        self.viewer.add_image(self.image.astype('float32'), name= 'Image', blending= 'additive' )
        for (k,v) in self.all_max_activations.items():
            time = k
            activations = v
            for count, activation in enumerate(activations):
                max_activation = np.sum(activation, axis = -1)[0,:]
                max_activation = normalizeFloatZeroOne(max_activation)
                print(f'activation_function_shape: {max_activation.shape}')          
                self.viewer.add_image(max_activation.astype('float32'), name= 'Activation_count' + str(count) + 'time_' + str(time), blending= 'additive', colormap='inferno' )
        napari.run()
      
    def VizualizeActivations(self):    
        print('loading model and losses')
        self._load_model_loss()
        print('boxes done, creating activation maps')
        self._activations_predictions()
        
        if self.visualize_point is not None:
          self.viewer.add_image(self.image[self.visualize_point,:].astype('float32'), name= 'Image', blending= 'additive' )
        else:
          self.viewer.add_image(self.image.astype('float32'), name= 'Image', blending= 'additive' )   
        for (k,v) in self.all_max_activations.items():
            time = k
            activations = v
            for count, activation in enumerate(activations):
                max_activation = np.sum(activation, axis = -1)[0,:]
                max_activation = normalizeFloatZeroOne(max_activation)
                print(f'activation_function_shape: {max_activation.shape}')          
                self.viewer.add_image(max_activation.astype('float32'), name= 'Activation_count' + str(count) + 'time_' + str(time), blending= 'additive', colormap='inferno' )
        napari.run()    
      
    def VizualizePredictions(self):    
        print('loading model and losses')
        self._load_model_loss()
        print('running predictions')
        self._model_prediction()
        print('prediction done, creating boxes')
        self._draw_boxes()
        self.viewer.add_image(self.image.astype('float32'), name= 'Image', blending= 'additive' )
        napari.run()    
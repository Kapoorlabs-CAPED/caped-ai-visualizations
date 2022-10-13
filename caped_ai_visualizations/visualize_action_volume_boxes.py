import numpy as np
import napari
import json 
class VisualizeBoxes(object):
    
    def __init__(self, viewer : napari.Viewer, key_categories : dict,  event_threshold: float):
        
        
        self.viewer = viewer
        self.key_categories = key_categories
        self.event_threshold = event_threshold
        self.event_locations = []
        self.event_locations_dict = {}
        self.event_locations_size_dict = {}
        self.size_locations = []
        self.score_locations = []
        self.confidence_locations = []


    def create_area_boxes(self, iou_classedboxes):
        
        
        self.iou_classedboxes = iou_classedboxes
        for (event_name, event_label) in self.key_categories.items():
            
            if event_label > 0:

                    iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                    iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)
                    for iou_current_event_box in iou_current_event_boxes:
                        xcenter = iou_current_event_box['xcenter'] 
                        ycenter = iou_current_event_box['ycenter'] 
                        tcenter = iou_current_event_box['real_time_event']
                        confidence = iou_current_event_box['confidence']
                        score = iou_current_event_box[event_name]
                        
                        size = np.sqrt(
                            iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                                'width'] * iou_current_event_box['width'] +  iou_current_event_box['depth'] * iou_current_event_box['depth'] ) // 3
                        
                        if score > self.event_threshold:
                                self.event_locations.append([int(tcenter), int(ycenter), int(xcenter)])   
                        
                                if int(tcenter) in self.event_locations_dict.keys():
                                                current_list = self.event_locations_dict[int(tcenter)]
                                                current_list.append([int(ycenter), int(xcenter)])
                                                self.event_locations_dict[int(tcenter)] = current_list 
                                                self.event_locations_size_dict[(int(tcenter), int(ycenter), int(xcenter))] = [size, score]
                                else:
                                    current_list = []
                                    current_list.append([int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list    
                                    self.event_locations_size_dict[int(tcenter), int(ycenter), int(xcenter)] = [size, score]
                               
                                self.size_locations.append(size)
                                self.score_locations.append(score)
                                self.confidence_locations.append(confidence)
                                
        point_properties = {'score' : np.array(self.score_locations), 'confidence' : np.array(self.confidence_locations),
                'size' : np.array(self.size_locations)}    
             
        name_remove = ('Detections','Location Map')
        for layer in list(self.viewer.layers):
                            
                            if  any(name in layer.name for name in name_remove):
                                    self.viewer.layers.remove(layer) 
        if len(self.score_locations) > 0:                             
                self.viewer.add_points(self.event_locations,  properties = point_properties, symbol = 'square', blending = 'translucent_no_depth', name = 'Detections' + event_name, face_color = [0]*3, edge_color = "red") 
                              

    def create_volume_boxes(self, iou_classedboxes, volumetric = True, shape = None):
        
        
        self.iou_classedboxes = iou_classedboxes
        for (event_name, event_label) in self.key_categories.items():
            
            if event_label > 0:
                    
                    iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                    iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)
                    for iou_current_event_box in iou_current_event_boxes:
                        xcenter = iou_current_event_box['xcenter'] 
                        ycenter = iou_current_event_box['ycenter'] 
                        if volumetric:
                          zcenter = iou_current_event_box['zcenter']
                        else:
                            if shape is not None and len(shape) == 4:
                                zcenter = shape[1]//2
                            else:
                                zcenter = 0    
                        tcenter = iou_current_event_box['real_time_event']
                        confidence = iou_current_event_box['confidence']
                        score = iou_current_event_box[event_name]
                        
                        size = np.sqrt(
                            iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                                'width'] * iou_current_event_box['width'] +  iou_current_event_box['depth'] * iou_current_event_box['depth'] ) // 3
                        
                        if score > self.event_threshold:
                                self.event_locations.append([int(tcenter),int(zcenter), int(ycenter), int(xcenter)])   
                                
                                if int(tcenter) in self.event_locations_dict.keys():
                                                current_list = self.event_locations_dict[int(tcenter)]
                                                current_list.append([int(zcenter),int(ycenter), int(xcenter)])
                                                self.event_locations_dict[int(tcenter)] = current_list 
                                                self.event_locations_size_dict[(int(tcenter), int(zcenter), int(ycenter), int(xcenter))] = [size, score]
                                else:
                                    current_list = []
                                    current_list.append([int(zcenter),int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list    
                                    self.event_locations_size_dict[int(tcenter),int(zcenter), int(ycenter), int(xcenter)] = [size, score]
                               
                                self.size_locations.append(size)
                                self.score_locations.append(score)
                                self.confidence_locations.append(confidence)
                                
        point_properties = {'score' : np.array(self.score_locations), 'confidence' : np.array(self.confidence_locations),
                'size' : np.array(self.size_locations)}    
             
        name_remove = ('Detections','Location Map')
        for layer in list(self.viewer.layers):
                            
                            if  any(name in layer.name for name in name_remove):
                                    self.viewer.layers.remove(layer) 
        if len(self.score_locations) > 0:                             
                self.viewer.add_points(self.event_locations,  properties = point_properties, symbol = 'square', blending = 'translucent_no_depth', name = 'Detections' + event_name, face_color = [0]*4, edge_color = "red") 
                                        
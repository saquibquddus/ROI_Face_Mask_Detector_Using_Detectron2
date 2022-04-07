import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
from com_ineuron_utils.utils import encodeImageIntoBase64
import datetime

class Detector:

	def __init__(self,filename):

		# set model and test set
		self.model = 'faster_rcnn_R_50_FPN_3x.yaml'
		self.filename = filename

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		self.cfg.merge_from_file("config.yml")
		#self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model))

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		#self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
		self.cfg.MODEL.WEIGHTS = "model_final.pth"

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'


	def inference(self, file, cropped):
		start = datetime.datetime.now()
		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		length, width = im.shape[0:2]
		# updated_image=updated_image[0:int(length), int(width / 2):int(width)]
		# im[0:int(length), int(width / 2):int(width)]=(0,0,0)
		# cropped_im = im[0:length, int(width / 2):int(width)]
		# outputs = predictor(cropped_im)
		# outputs = predictor(im)
		if cropped=="LHR":
			updated_image=im.copy()
			updated_image[0:int(length), int(width / 2):int(width)]=(0,0,0)
			cv.imwrite('cropped_images/LHR.jpg', updated_image)
			cv.rectangle(im,(0,0),(int(width/2),int(length)),(0,0,255),3)
		elif cropped=="RHR":
			updated_image=im.copy()
			updated_image[0:int(length),0:int(width/2)]=(0,0,0)
			cv.imwrite('cropped_images/RHR.jpg', updated_image)
			cv.rectangle(im,(int(width/2),0),(int(width),int(length)),(0,0,255),3)
		elif cropped=="THR":
			updated_image=im.copy()
			updated_image[int(length/2):int(length),0:int(width)]=(0,0,0)
			cv.imwrite('cropped_images/THR.jpg', updated_image)
			cv.rectangle(im,(0,0),(int(width),int(length/2)),(0,0,255),3)
		elif cropped=="BHR":
			updated_image=im.copy()
			updated_image[0:int(length/2),0:int(width)]=(0,0,0)
			cv.imwrite('cropped_images/BHR.jpg', updated_image)
			cv.rectangle(im,(0,int(length/2)),(int(width),int(length)),(0,0,255),3)
		elif cropped=="MR":
			updated_image=im.copy()
			updated_image[0:int(length),0:int(width/4)]=(0,0,0)
			updated_image[0:int(length),int(width*3/4):int(width)]=(0,0,0)
			cv.imwrite('cropped_images/MR.jpg', updated_image)
			cv.rectangle(im,(int(width/4),0),(int(width*3/4),int(length)),(0,0,255),3)
		elif cropped=="BCR":
			updated_image=im.copy()
			updated_image[0:int(length),0:int(width/5)]=(0,0,0)
			updated_image[0:int(length),int(width*4/5):int(width)]=(0,0,0)
			updated_image[0:int(length/5),0:int(width)]=(0,0,0)
			updated_image[int(length*4/5):int(length),0:int(width)]=(0,0,0)
			cv.imwrite('cropped_images/BCR.jpg', updated_image)
			cv.rectangle(im,(int(width/5),int(length/5)),(int(width*4/5),int(length*4/5)),(0,0,255),3)
		else:
			updated_image=im.copy()
			cv.imwrite('cropped_images/actual_image.jpg', updated_image)
			cv.rectangle(im,(0,0),(int(width),int(length)),(0,0,255),3)
		
		outputs = predictor(updated_image)
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
		
		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		predicted_image = v.get_image()
		im_rgb = cv.cvtColor(predicted_image, cv.COLOR_RGB2BGR)

		end = datetime.datetime.now()
		total_seconds=(end-start).total_seconds()
		cv.putText(img=im_rgb, text=str(total_seconds),org=(0 + int(width/12),0 + int(length/10)), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,255,255), thickness=2)
		cv.imwrite('output_image.jpg', im_rgb)
		
		opencodedbase64 = encodeImageIntoBase64("output_image.jpg")
		
		result = {"image" : opencodedbase64.decode('utf-8') }
		
		print("time_difference = ",(end-start).total_seconds()," sec")
		return result





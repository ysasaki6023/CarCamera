import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
from yolo.utils.process_config import process_config

"""
parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure", default="",
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)
"""

img_size=[720,1280]
boxSize      = 64 # 448/7
h,w   = img_size
nh,nw = int(h/boxSize), int(w/boxSize)
hh,ww = nh*boxSize, nw*boxSize

batchNum = 1
common_params = {'image_size_x': ww, 'image_size_y': hh, 'num_classes': 20, 'batch_size':batchNum, "max_objects_per_image":20}
net_params = {'cell_size_x': nw, 'cell_size_y': nh, 'boxes_per_cell':2, 'weight_decay': 0.0005, "object_scale":1., "noobject_scale":0.5, "class_scale":1.0, "coord_scale": 5.}
dataset_params={"path":"data/pascal_voc.txt","thread_num":5}
#solver_params = {"learning_rate":0.0001, "moment":0.9, "max_iterators":100000000, "pretrain_model_path":"models/pretrain/yolo_tiny.ckpt","train_dir":"models/train", "load_local_params":False, "train_conv_params":False, "store_iterations":1000}
solver_params = {"learning_rate":0.00001, "moment":0.9, "max_iterators":100000000, "pretrain_model_path":"models/train/model.ckpt-4000","train_dir":"models/train", "load_local_params":True, "train_conv_params":False, "store_iterations":1000}

#common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = yolo.dataset.text_dataset2.TextDataSet(common_params, dataset_params)
net = yolo.net.yolo_tiny_net2.YoloTinyNet(common_params, net_params, solver_params)
solver = yolo.solver.yolo_solver2.YoloSolver(dataset, net, common_params, solver_params)
solver.solve()

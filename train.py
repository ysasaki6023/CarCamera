from yolo.solver import yolo_solver
from yolo.dataset import text_dataset
from yolo.net import yolo_tiny_net

if __name__=="__main__":
    batchNum=10
    common_params = {'image_size': 448, 'num_classes': 20, 'batch_size':batchNum}
    net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005,
            'object_scale': 5., 'noobject_scale': 0.5, 'class_scale': 1.0, 'coord_scape': 0.5}
    solver_params = {'moment':0.,'learning_rate':0.001, 'pretrain_model_path':, 'train_dir':,'max_iterators':}

    dataset_params = {'path':,'thread_num':}

    net = YoloTinyNet(common_params, net_params, test=False)

    image = tf.placeholder(tf.float32, (batchNum, 448, 448, 3))

    ys = YoloSolver(dataset=,
                    net=net,
                    common_params=common_params,
                    solver_params=)


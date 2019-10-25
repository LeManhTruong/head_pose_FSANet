"""
aioz.aiar.truongle - May 04, 2019
take yaw, pitch, roll from face
"""
import os
import src.FSANET_model as FSANET
from src.config import Config
from keras.layers import Input, Average
from keras.models import Model
from termcolor import colored

import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_fsanet_model(config):
    # Parameters
    S_set = [config.num_capsule, config.dim_capsule, config.routings, config.num_primcaps, config.m_dim]
    model1 = FSANET.FSA_net_Capsule(config.image_size, config.num_classes, config.stage_num, config.lambda_d, S_set)()
    model2 = FSANET.FSA_net_Var_Capsule(config.image_size, config.num_classes, config.stage_num, config.lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3
    S_set = [config.num_capsule, config.dim_capsule, config.routings, num_primcaps, config.m_dim]
    model3 = FSANET.FSA_net_noS_Capsule(config.image_size, config.num_classes, config.stage_num, config.lambda_d, S_set)()

    print(colored('[INFO] Loading models ...', "green", attrs=["bold"]))

    model1.load_weights(config.weight_file1)
    print(colored('[INFO] Finished loading model 1.', "cyan", attrs=["bold"]))

    model2.load_weights(config.weight_file2)
    print(colored('[INFO] Finished loading model 2.', "cyan", attrs=["bold"]))

    model3.load_weights(config.weight_file3)
    print(colored('[INFO] Finished loading model 3.', "cyan", attrs=["bold"]))
    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    return model


def main():
    config = Config()
    K.set_learning_phase(0)
    sess = K.get_session()
    model = load_fsanet_model(config=config)

    converted_output_node_names = [node.op.name for node in model.outputs]
    print("[INFO] in: ", model.inputs)
    print("[INFO] out: ", model.outputs)

    constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)
    if not os.path.isdir("graph"):
        os.mkdir("graph")
    tf.train.write_graph(constant_graph, './graph', 'fsanet.pbtxt', as_text=True)
    tf.train.write_graph(constant_graph, './graph', 'fsanet.pb', as_text=False)
    print(colored("[INFO] convert model is success, saved at ./graph/fsanet.pb", "cyan", attrs=['bold']))


if __name__ == '__main__':
    main()

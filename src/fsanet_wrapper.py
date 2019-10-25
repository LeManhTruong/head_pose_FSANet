"""
aioz.aiar.truongle - May 06, 2019
FSANET - head pose detection
"""
import tensorflow as tf
import numpy as np
from termcolor import colored

tf.logging.set_verbosity(tf.logging.ERROR)


class FsanetWrapper:
    def __init__(self, graph="graph/fsanet.pb", memory_fraction=0.7):
        self.graph_fb = graph
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction, allow_growth=True)
        # self.config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        # # Config for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        self.config.gpu_options.allow_growth = True
        self.config.log_device_placement = False

        self.sess = None

        self.__load_graph()
        self.__init_prediction()

    def __load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            # with gfile.FastGFile(self.graph_fb, 'rb') as fid:
            with tf.gfile.GFile(self.graph_fb, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def)
        # tf.get_default_graph().finalize()
        # print(colored("[INFO] load graph for FSANET is done", "green", attrs=['bold']))

    def __init_prediction(self):
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph, config=self.config)
            # for op in tf.get_default_graph().get_operations():
            #     print(str(op.name))
            self.output_tensor = self.graph.get_tensor_by_name('import/average_1/truediv:0')
            self.input_tensor = self.graph.get_tensor_by_name('import/input_27:0')
        # print(colored("[INFO] Init model is done", "green", attrs=['bold']))

    def predict(self, images):
        """require size of images: [?, 64, 64, 3]"""
        if images.ndim == 3:
            """Run without batch"""
            inputs = np.expand_dims(images, axis=0)
        elif images.ndim < 3 or images.ndim > 4:
            raise Exception("check images dims, require [?, 64, 64, 3], images dims {}".format(images.ndim))
        else:
            """Run with batch"""
            inputs = images
        outputs = self.sess.run([self.output_tensor], feed_dict={self.input_tensor: inputs})
        return np.asarray(outputs)[0]

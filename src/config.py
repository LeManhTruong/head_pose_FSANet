"""
aioz.aiar.truongle - May 04, 2019
config parameter
"""


class Config:
    def __init__(self):
        self.weight_file1 = 'weight/fsanet_capsule_3_16_2_21_5.h5'
        self.weight_file2 = 'weight/fsanet_var_capsule_3_16_2_21_5.h5'
        self.weight_file3 = 'weight/fsanet_noS_capsule_3_16_2_192_5.h5'
        self.graph_fsanet = 'graph/fsanet.pb'

        self.num_capsule = 3
        self.dim_capsule = 16
        self.routings = 2
        self.stage_num = [3, 3, 3]
        self.lambda_d = 1
        self.num_classes = 3
        self.image_size = 64
        self.num_primcaps = 7 * 3
        self.m_dim = 5
        self.ratio = 0.6
        self.threshold = 0.7

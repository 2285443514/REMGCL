#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import PIL
import random
from paddleocr import PaddleOCR, draw_ocr


torch.set_printoptions(profile="full")
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(
            3,
            3,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.blur_v = nn.Conv2d(
            3,
            3,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
            groups=3,
        )
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_block_indices(image_width, image_height, bounding_box, coverage_threshold=0.05):
    # 计算每个块的宽度和高度
    block_width = image_width / 14
    block_height = image_height / 14

    # 提取bounding box的坐标
    x1, y1, x2, y2 = bounding_box

    # 找到bounding box覆盖的块的范围
    start_block_x = int(x1 // block_width)
    end_block_x = int(x2 // block_width)
    start_block_y = int(y1 // block_height)
    end_block_y = int(y2 // block_height)

    # 获取覆盖的块的所有序号
    block_indices = []
    for i in range(start_block_y, end_block_y + 1):
        for j in range(start_block_x, end_block_x + 1):
            # 计算当前块的边界
            block_x1 = j * block_width
            block_y1 = i * block_height
            block_x2 = block_x1 + block_width
            block_y2 = block_y1 + block_height

            # 计算bounding box和当前块的交集
            inter_x1 = max(x1, block_x1)
            inter_y1 = max(y1, block_y1)
            inter_x2 = min(x2, block_x2)
            inter_y2 = min(y2, block_y2)

            # 计算交集面积
            inter_width = max(0, inter_x2 - inter_x1)
            inter_height = max(0, inter_y2 - inter_y1)
            inter_area = inter_width * inter_height

            # 计算当前块的面积
            block_area = block_width * block_height

            # 计算覆盖比例
            coverage = inter_area / block_area

            # 如果覆盖比例大于等于阈值，则添加当前块的序号
            if coverage >= coverage_threshold:
                block_index = i * 14 + j
                block_indices.append(block_index)

    return block_indices


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        self.stopwords = [
                [IGNORE_INDEX],
                [13],  # \n
                [29892],  # ,
                [29889],  # .
                [376],  # "
                [29915],  #'
                [29908],  # "
                [29899],  # -
                [29901],  #:
                [29871],  #
                [613],  # ",
                [1213],  # ."
                [1642],  # ".
                [1699],  # ,"
                [2688],  # They
                [450],  # The
                [1967],  # image
                [910],  # This
                [8833],  # displayed
                [5680],  # features
                [739],  # It
                [14661],  # suggests
                [512],  # In
                [1576],  # The
                [19814],  # Additionally
                [23941],  # indicating
                [26233],  # suggesting
                [14088],  # indicates
                [525, 29881],
                [525, 645],
                [525, 29885],
                [525, 276],
                [525, 29879],
                [525, 29873],
                [525, 345],
                [796, 29911],
                [796, 29999],
                [263],
                [263, 29915, 29879],
                [2221],
                [1048],
                [2038],
                [633, 303],
                [15017, 749],
                [5034],
                [16205],
                [4822],
                [1044],
                [2869],
                [2715],
                [12109],
                [16356],
                [15201],
                [6602, 292],
                [6602, 29879],
                [1156],
                [12335],
                [1449],
                [2750],
                [21023],
                [7216, 29915, 29873],
                [599],
                [2758],
                [6511],
                [4359],
                [7432],
                [3412],
                [2307],
                [884],
                [5998],
                [2337],
                [626],
                [4249],
                [22611],
                [385],
                [322],
                [7475, 346],
                [1790],
                [738],
                [16357],
                [738, 3525],
                [15128],
                [5019],
                [3099],
                [8763],
                [738, 1994],
                [12214],
                [12435],
                [13229],
                [2615],
                [11188],
                [8210],
                [14235],
                [526],
                [4038],
                [10161],
                [9455],
                [9455, 29915, 29873],
                [564, 296],
                [29030],
                [2820],
                [408],
                [17786],
                [2244],
                [4433],
                [6721],
                [19514],
                [6942],
                [472],
                [4817],
                [3625],
                [3448],
                [3773, 3730],
                [289],
                [1250],
                [1250, 287],
                [27436],
                [1250, 29879],
                [367],
                [3897],
                [1363],
                [4953],
                [7415],
                [14171],
                [1063],
                [1434],
                [1434, 3179],
                [4689],
                [3380],
                [6763],
                [1812, 2559, 886],
                [16410],
                [5742],
                [1641],
                [367, 886],
                [4658],
                [2400],
                [17620],
                [18034],
                [1900],
                [2253],
                [1546],
                [8724],
                [4802],
                [4768, 324],
                [1716],
                [11473],
                [23359],
                [541],
                [491],
                [274],
                [274, 29915, 3712],
                [274, 29915, 29879],
                [5777],
                [2996],
                [508],
                [508, 29915, 29873],
                [2609],
                [5107],
                [1206],
                [4251],
                [4556],
                [9946],
                [3058],
                [8959],
                [3620],
                [2821],
                [9436],
                [1302],
                [419],
                [2041],
                [5304],
                [19813],
                [14161, 2705],
                [2050],
                [13858],
                [1712],
                [6943],
                [3743],
                [6590],
                [1033],
                [8496, 29915, 29873],
                [1033, 593],
                [3236],
                [5279],
                [270],
                [2635],
                [11630],
                [8453],
                [5439],
                [15020],
                [1258],
                [3282, 29915, 29873],
                [1163],
                [1422],
                [17587],
                [5353],
                [437],
                [947],
                [1838, 29915, 29873],
                [2599],
                [1016, 29915, 29873],
                [2309],
                [1623],
                [1623, 287],
                [1623, 292],
                [1623, 29879],
                [1623, 2935],
                [2861],
                [2645],
                [321],
                [1269],
                [4688],
                [1226],
                [1226, 29884],
                [2779],
                [8087],
                [9475],
                [9475, 29891],
                [2845],
                [1683],
                [17551],
                [1095],
                [9698],
                [17140],
                [10614],
                [3307],
                [9186],
                [7148],
                [634],
                [634, 29899, 284],
                [2992],
                [1584],
                [1584, 368],
                [3926],
                [1432],
                [26077],
                [14332],
                [4129],
                [16978],
                [429],
                [3721],
                [1342],
                [5174],
                [285],
                [3700],
                [17240],
                [2114],
                [17099],
                [2215],
                [7091],
                [2846],
                [14336],
                [18615],
                [1284],
                [14061],
                [937],
                [5320],
                [2329],
                [5643],
                [1494],
                [4477],
                [363],
                [4642],
                [21510],
                [11483],
                [1476],
                [3023],
                [515],
                [2989],
                [8072],
                [4340],
                [4340, 287],
                [4340, 292],
                [4340, 5514],
                [3261, 3341],
                [330],
                [4846],
                [2498],
                [6892],
                [679],
                [4947],
                [2805],
                [2367],
                [2183],
                [4076],
                [6820],
                [748],
                [5771],
                [2675],
                [7695],
                [1781],
                [22535],
                [2355],
                [2355, 841],
                [2107],
                [7621],
                [14176],
                [1395, 300, 886],
                [2318],
                [27831],
                [27270],
                [6471],
                [298],
                [750],
                [27222, 29915, 29873],
                [5930],
                [15155],
                [756],
                [22602, 29915, 29873],
                [505],
                [7359, 29915, 29873],
                [2534],
                [540],
                [540, 29915, 29879],
                [298, 287],
                [22172],
                [1371],
                [8151],
                [902],
                [1244],
                [1244, 29915, 29879],
                [1244, 7045],
                [1244, 1609],
                [1244, 262],
                [902, 267],
                [1244, 786, 265],
                [7955],
                [8735],
                [19066],
                [7251],
                [20552],
                [1880],
                [6133],
                [9939],
                [1075],
                [3654],
                [670],
                [298, 2121],
                [3271],
                [27581],
                [920],
                [920, 6205],
                [3138],
                [6893],
                [474],
                [474, 29915, 29881],
                [474, 29915, 645],
                [474, 29915, 29885],
                [474, 29915, 345],
                [1178],
                [19282],
                [565],
                [17262],
                [527],
                [16800],
                [7389],
                [13500],
                [4100],
                [297],
                [297, 11625, 987],
                [5528],
                [3160],
                [6200],
                [2380],
                [12266],
                [18694],
                [14088],
                [2472],
                [6426],
                [297, 578, 15641],
                [2012],
                [4066],
                [8852],
                [8031],
                [20017],
                [964],
                [297, 7316],
                [297, 1328],
                [338],
                [3508, 29915, 29873],
                [372],
                [372, 29915, 29881],
                [372, 29915, 645],
                [372, 29915, 29879],
                [372, 29881],
                [967],
                [3528],
                [432],
                [925],
                [413],
                [3013],
                [14874],
                [8126],
                [6611],
                [12118],
                [2924],
                [2383],
                [6363],
                [1073],
                [2998],
                [9906],
                [301],
                [2919],
                [18425],
                [1833],
                [301, 2486],
                [2678],
                [9281],
                [7480],
                [7480, 368],
                [3203],
                [3109],
                [301, 342],
                [1235],
                [1235, 29915, 29879],
                [16869],
                [763],
                [23289],
                [5517],
                [1196],
                [2217],
                [1472],
                [5520],
                [27217],
                [1106],
                [3063],
                [3430],
                [301, 1594],
                [286],
                [1754],
                [14364],
                [1207],
                [3732],
                [3907],
                [767],
                [1784],
                [1122],
                [5505],
                [592],
                [2099],
                [2794],
                [6839, 603],
                [2099, 8000],
                [4509],
                [5144],
                [1757],
                [13586],
                [286, 29887],
                [1795],
                [7284],
                [3052],
                [286, 29880],
                [901],
                [25409],
                [1556],
                [11149],
                [286, 29878],
                [286, 2288],
                [1568],
                [286, 688],
                [1818],
                [590],
                [6142],
                [302],
                [302, 29915, 29873],
                [1055],
                [1024],
                [18451],
                [302, 388],
                [29871, 299],
                [2978],
                [8886],
                [12695],
                [5181],
                [817],
                [4312],
                [817, 292],
                [4225],
                [9561],
                [2360],
                [2360, 16561],
                [716],
                [20687],
                [716, 342],
                [2446],
                [14183],
                [17081, 3305],
                [694],
                [23196],
                [1661],
                [5642],
                [1661, 621, 6393],
                [694, 650],
                [3643],
                [12891],
                [7814],
                [451],
                [11682],
                [3078],
                [9554],
                [1286],
                [1286, 4150],
                [1353],
                [3694],
                [288],
                [4017],
                [7625],
                [12879],
                [310],
                [1283],
                [4049],
                [9360],
                [3431],
                [20759],
                [2030],
                [9642],
                [23947],
                [25811],
                [373],
                [2748],
                [697],
                [6743],
                [871],
                [11480],
                [1722],
                [6496],
                [8718],
                [13246],
                [470],
                [4356],
                [1797],
                [10372],
                [20520],
                [11299],
                [916],
                [4045],
                [6467],
                [12722],
                [1749],
                [1749, 29879],
                [20278],
                [714],
                [5377],
                [975],
                [12463],
                [8152, 292],
                [1914],
                [282],
                [1813],
                [6515],
                [760],
                [760, 287],
                [3153],
                [10734],
                [760, 292],
                [5633],
                [4940],
                [639],
                [6060],
                [2058],
                [7180],
                [7600],
                [3113],
                [2298],
                [1298],
                [11520],
                [13330],
                [3291],
                [6460, 368],
                [1950],
                [10075],
                [19998],
                [6499],
                [758, 24130, 10835],
                [2198],
                [9132],
                [2198, 292],
                [22981],
                [2225, 24873],
                [9251],
                [19434],
                [3117],
                [1108],
                [4828],
                [9508, 368],
                [22314],
                [8128],
                [1925],
                [15223],
                [3855],
                [712],
                [9098],
                [3755],
                [3855, 29894],
                [364],
                [6350],
                [3265],
                [364, 29881],
                [337],
                [28520],
                [2289],
                [2769, 2197],
                [7786],
                [10325],
                [2143],
                [2143, 29879],
                [11211],
                [17126],
                [21778],
                [4475],
                [13774],
                [5925],
                [8307],
                [20601],
                [9819],
                [2582],
                [1492],
                [5716],
                [19600],
                [1065],
                [269],
                [1497],
                [1021],
                [4446],
                [1827],
                [5934],
                [4083],
                [5226],
                [1473],
                [1473, 368],
                [6923],
                [4004],
                [1074],
                [8790],
                [2833],
                [6140],
                [2833, 292],
                [2444],
                [3595],
                [18553],
                [1583],
                [5535, 1960],
                [25182],
                [2665],
                [10676],
                [25798],
                [9881],
                [3196],
                [4091],
                [1183],
                [1183, 29915, 645],
                [28453],
                [528, 267],
                [881],
                [9273, 29915, 29873],
                [1510],
                [10018],
                [6445],
                [4318],
                [4318, 29879],
                [3697],
                [2625],
                [11192],
                [7282],
                [16951],
                [2788],
                [22829],
                [1951],
                [4832],
                [10029],
                [2319],
                [7968],
                [19087],
                [577],
                [777],
                [18462],
                [10431],
                [4856],
                [1047, 621, 273],
                [1554],
                [1047, 5410],
                [6041],
                [10579],
                [9051],
                [4720],
                [7423],
                [10816],
                [6790],
                [6084],
                [22146],
                [2106],
                [5922],
                [1603],
                [5040],
                [13818],
                [1014],
                [23228, 368],
                [8472],
                [1316],
                [18430],
                [4368],
                [13159],
                [1854],
                [260],
                [260, 29915, 29879],
                [2125],
                [4586],
                [5622],
                [2649],
                [29867],
                [266],
                [1135],
                [6452],
                [3969],
                [1135, 29916],
                [393],
                [393, 29915, 645],
                [393, 29915, 29879],
                [393, 29915, 345],
                [20952],
                [278],
                [1009],
                [1009, 29879],
                [963],
                [6053],
                [769],
                [266, 663],
                [727],
                [727, 29915, 645],
                [727, 29915, 29879],
                [727, 29915, 345],
                [727, 7045],
                [27999],
                [266, 14561],
                [5480],
                [727, 262],
                [727, 974],
                [29220, 406],
                [266, 11175],
                [29220, 10896],
                [727, 786, 265],
                [1438],
                [896],
                [896, 29915, 29881],
                [896, 29915, 645],
                [896, 29915, 276],
                [896, 29915, 345],
                [896, 29881],
                [896, 276],
                [2655],
                [2712],
                [1348],
                [22405],
                [4654],
                [445],
                [17826],
                [26606],
                [1906],
                [12595],
                [2466],
                [2466, 29882],
                [2714],
                [13133],
                [10405],
                [2211],
                [1468, 692],
                [1549],
                [10106],
                [266, 582],
                [4550],
                [6928],
                [6872],
                [304],
                [9826],
                [4208],
                [2086],
                [3614],
                [11183],
                [7113],
                [1898],
                [14335],
                [19781],
                [1018],
                [1811],
                [18696],
                [2507],
                [6077],
                [14712],
                [12169],
                [8951],
                [1023],
                [318],
                [443],
                [1090],
                [15428],
                [6521],
                [25531],
                [25057],
                [2745],
                [20550],
                [701],
                [2501],
                [24081],
                [502],
                [671],
                [1304],
                [5407],
                [671, 3730],
                [5407, 2264],
                [3913],
                [773],
                [5491],
                [318, 1682, 29886],
                [325],
                [995],
                [5164],
                [1407],
                [3025],
                [25294],
                [1700],
                [1700, 29879],
                [7186],
                [281],
                [864],
                [5131],
                [24507],
                [10753],
                [471],
                [9007, 29915, 29873],
                [982],
                [5837],
                [591],
                [591, 29915, 29881],
                [591, 29915, 645],
                [591, 29915, 276],
                [591, 29915, 345],
                [14837],
                [12853],
                [1532],
                [1532, 29879],
                [3512],
                [892],
                [2949, 264, 29915, 29873],
                [825],
                [825, 29915, 645],
                [825, 29915, 29879],
                [6514],
                [825, 29879],
                [746],
                [377, 663],
                [10940],
                [988],
                [988, 29915, 29879],
                [988, 7045],
                [13452],
                [988, 1609],
                [988, 262],
                [377, 11175],
                [988, 786, 265],
                [29693],
                [3692],
                [607],
                [1550],
                [377, 326],
                [377, 2121],
                [1058],
                [1058, 29915, 645],
                [1058, 29915, 29879],
                [377, 397],
                [1058, 1310],
                [3353],
                [6029],
                [377, 608, 369],
                [377, 359],
                [5069],
                [2020],
                [17644],
                [674],
                [17762],
                [6398],
                [411],
                [2629],
                [1728],
                [2113, 29915, 29873],
                [4997],
                [3838],
                [664],
                [3796],
                [1985],
                [1736],
                [3186],
                [723],
                [7656, 29915, 29873],
                [7821],
                [921],
                [343],
                [1629],
                [2440],
                [4874],
                [3447],
                [366],
                [366, 29915, 29881],
                [366, 29915, 645],
                [366, 29915, 276],
                [366, 29915, 345],
                [366, 29881],
                [4123],
                [20023],
                [4123, 342],
                [596],
                [366, 276],
                [15850],
                [7535],
                [596, 5210],
                [503],
                [5225],
                [503, 29873],
                [503, 29920],
            ]


        # print(type(self.get_model().get_vision_tower()))
        # print(self.get_model().get_vision_tower().size)
        # self.init_simclr_pipeline_transform()
        # Initialize weights and apply final processing
        self.post_init()

    def create_filter_mask(self, target_tensor):
        # 初始化掩码，默认全为True（保留所有元素）
        mask = torch.ones_like(target_tensor, dtype=torch.bool)

        # 逐行处理target_tensor
        for row_idx in range(target_tensor.size(0)):
            # 提取当前行
            row = target_tensor[row_idx]

            # 遍历停用词列表
            for sw in self.stopwords:
                if len(sw) == 1:
                    # 处理单个数字停用词
                    mask[row_idx] &= row != sw[0]
                else:
                    # 处理序列停用词
                    seq_len = len(sw)
                    # 创建滑动窗口，检查每个子序列是否与停用词匹配
                    for i in range(len(row) - seq_len + 1):
                        if torch.equal(
                            row[i : i + seq_len], torch.tensor(sw).to(row.device)
                        ):
                            mask[row_idx, i : i + seq_len] = False

        return mask
    
    def create_filter_mask_conv_accu(self,target_tensor):
        mask = torch.ones_like(target_tensor, dtype=torch.bool)
        
        for sw in self.stopwords:
            sw_tensor = torch.tensor(sw, dtype=torch.float).to(target_tensor.device)
            seq_len = len(sw)
            
            if seq_len == 1:
                mask &= target_tensor != sw_tensor[0]
            else:
                kernel = sw_tensor.unsqueeze(0).unsqueeze(0).float()
                target_tensor_float = target_tensor.unsqueeze(1).float()
                conv_out = F.conv1d(target_tensor_float, kernel, padding=0)
                target_sum_squared = torch.sum(sw_tensor ** 2)
                
                # Calculate the Euclidean distance
                match_mask = conv_out.squeeze(1).eq(target_sum_squared)
            
                # Create the final mask for matched positions
                for j in range(target_tensor.size(0)):
                    for i in range(match_mask.size(1)):
                        if match_mask[j, i]:
                            # Verify the match by checking if the squared difference is zero
                            segment = target_tensor[j, i:i + seq_len].float()
                            segment_sum_squared = torch.sum((segment - sw_tensor.float()) ** 2)
                            if segment_sum_squared.item() < 1e-6:
                                mask[j, i:i + seq_len] = False

        return mask
    
    

    def create_filter_mask_unfold(self,target_tensor):
        mask = torch.ones_like(target_tensor, dtype=torch.bool)
        
        for sw in self.stopwords:
            sw_tensor = torch.tensor(sw, dtype=torch.float).to(target_tensor.device)
            seq_len = len(sw)
            
            if seq_len == 1:
                # 单字符匹配
                mask &= target_tensor != sw_tensor[0]
            else:
                # 创建分段张量
                target_padded = F.pad(target_tensor, (0, seq_len - 1), mode='constant', value=0)
                segment = target_padded.unfold(1, seq_len, 1)
                
                # 计算平方差异
                sw_tensor_squared = sw_tensor.float().unsqueeze(0).unsqueeze(0)
                segment_squared = torch.sum(segment ** 2, dim=2)
                sw_tensor_squared_sum = torch.sum(sw_tensor_squared ** 2, dim=2)
                cross_term = 2 * torch.sum(segment * sw_tensor_squared, dim=2)
                squared_diff = segment_squared - cross_term + sw_tensor_squared_sum
                
                # 确定符合条件的滑动窗口位置
                matches = (squared_diff <= 1e-6)
                
                # 更新掩码, 将所有匹配位置的停用词长度区域标记为 False
                for i in range(matches.size(0)):  # 遍历 batch
                    for j in range(matches.size(1)):  # 遍历每个序列中的滑动窗口起始位置
                        if matches[i, j]:
                            mask[i, j:j+seq_len] = False
        return mask

    def get_model(self):
        return self.model

    def init_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        raw_images: Optional[PIL.Image.Image] = None,
        pad_indices: Optional[List[int]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_embeds,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        # Global + patch + stopwords
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # contrastive loss
            # image_embeds.shape torch.Size([16, 256, 4096])
            hidden_states = outputs[0]
            contra_temperature = 0.5
            contra_loss_weight = 0.01

            # 使用torch.masked_select()函数来过滤出非IGNORE_INDEX的值
            # label_mask = ~torch.isin(labels, self.stopwords.to(labels.device))
            label_mask = self.create_filter_mask_conv_accu(labels).to(labels.device)
            filtered_labels = torch.masked_select(labels, label_mask)
            # print("filtered_labels", filtered_labels)

            label_embedding = self.model.embed_tokens(filtered_labels)
            # print("labels", labels.shape)
            # print("label_embedding", label_embedding.shape)
            # print("raw_images", raw_images)
            # print("image_embeds.shape", image_embeds.shape)
            # print("pad_indices.shape", len(pad_indices))
            # print("pad_indices.shape", len(pad_indices[0]))
            # print("image_embeds", image_embeds)

            # 可视化patch
            # for raw_image_index in range(len(raw_images)):
            #     image = raw_images[raw_image_index]
            #     image_width = image.width  # 图片的宽
            #     image_height = image.height  # 图片的高

            #     draw = PIL.ImageDraw.Draw(image, "RGBA")

            #     block_width = image_width / 14
            #     block_height = image_height / 14

            #     # 将对应的块涂上半透明的蓝色
            #     for index in pad_indices[raw_image_index]:
            #         i = index // 14
            #         j = index % 14
            #         block_x1 = j * block_width
            #         block_y1 = i * block_height
            #         block_x2 = block_x1 + block_width
            #         block_y2 = block_y1 + block_height

            #         draw.rectangle([block_x1, block_y1, block_x2, block_y2], fill=(0, 0, 255, 64))
            #     image_save_path = f'path/to/codes/LLaVA/output/7/{random.randint(0, 1000000)}.png'
            #     image.save(image_save_path)

            # 重新构建为原始的二维结构
            num_rows = label_mask.sum(dim=1)  # 获取每行保留的元素数量
            reshaped_labels = torch.split_with_sizes(label_embedding, num_rows.tolist())
            labels_embeds_mean = torch.stack(
                [torch.mean(label, dim=0) for label in reshaped_labels]
            )

            batch_size, feature_dim, embedding_dim = image_embeds.shape

            # mean_embed_list = []
            # # 对每个batch进行处理
            # for i in range(batch_size):
            #     # 创建一个掩码
            #     mask = torch.ones(feature_dim, dtype=torch.bool)
            #     mask[pad_indices[i]] = False

            #     # 筛选掉不需要的元素
            #     filtered_embed = image_embeds[i][mask]
            #     # print("filtered_embed",filtered_embed.shape)
            #     # 计算均值并添加到列表中
            #     mean_embed_list.append(filtered_embed.mean(dim=0))

            # 使用stack将列表合并为一个tensor
            # image_embeds_mean = torch.stack(mean_embed_list)

            image_embeds_mean = torch.mean(image_embeds, dim=1)
            # print("mean_embed",mean_embed.shape)

            # image_embeds_mean = torch.mean(image_embeds, dim=1)
            # print("image_embeds_mean",image_embeds_mean.shape)

            similarity_matrix = F.cosine_similarity(
                labels_embeds_mean, image_embeds_mean
            )

            similarity_matrix = torch.exp(similarity_matrix / contra_temperature)

            positives = similarity_matrix

            loss_partial = -torch.log(positives)
            contra_loss = torch.sum(loss_partial) / batch_size
            print("contra_loss", contra_loss)

            loss += contra_loss_weight * contra_loss

            raw_images_np = [np.array(raw_image) for raw_image in raw_images]
            # print("raw_images_np", raw_images_np)
            ocr_results = [
                self.ocr.ocr(raw_image_np, cls=False) for raw_image_np in raw_images_np
            ]
            # print("ocr_results", ocr_results)

            image_local_features_mean = []
            text_local_features_mean = []
            image_text_local_weight = []
            for ocr_index, ocr_res in enumerate(ocr_results):
                result = ocr_res[0]
                if result is not None:
                    for ocr_local in range(len(result)):
                        image = raw_images[ocr_index]
                        image_width = image.width  # 图片的宽
                        image_height = image.height  # 图片的高

                        xmin = min(
                            result[ocr_local][0][0][0],
                            result[ocr_local][0][1][0],
                            result[ocr_local][0][2][0],
                            result[ocr_local][0][3][0],
                        )
                        ymin = min(
                            result[ocr_local][0][0][1],
                            result[ocr_local][0][1][1],
                            result[ocr_local][0][2][1],
                            result[ocr_local][0][3][1],
                        )
                        xmax = max(
                            result[ocr_local][0][0][0],
                            result[ocr_local][0][1][0],
                            result[ocr_local][0][2][0],
                            result[ocr_local][0][3][0],
                        )
                        ymax = max(
                            result[ocr_local][0][0][1],
                            result[ocr_local][0][1][1],
                            result[ocr_local][0][2][1],
                            result[ocr_local][0][3][1],
                        )
                        bbox = (xmin, ymin, xmax, ymax)
                        block_indices = get_block_indices(
                            image_width, image_height, bbox
                        )
                        if not block_indices:
                            continue
                        # print("block_indices",block_indices)

                        raw_text = result[ocr_local][1][0]
                        confidence_score = result[ocr_local][1][1]

                        # print(raw_text)
                        # print(confidence_score)

                        # text_inputs_id = self.tokenizer(raw_text, padding=False, truncation=True, return_tensors="pt")["input_ids"][0:]
                        # print(raw_text)
                        # print("text_inputs_id",text_inputs_id)
                        # print("text_inputs_id tokenize",self.tokenizer.batch_decode(text_inputs_id))

                        # text_local_embedding = self.model.embed_tokens(text_inputs_id.to(image_embeds_mean.device))
                        # print("text_local_embedding",text_local_embedding.shape)

                        # raw_text = raw_text.lower()

                        text_inputs_id = self.tokenizer(
                            raw_text,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                        )["input_ids"]
                        # print(raw_text)
                        # print("text_inputs_id",text_inputs_id)
                        text_inputs_id = text_inputs_id[0][1:]
                        # print("text_inputs_id",text_inputs_id)
                        # print("text_inputs_id tokenize",self.tokenizer.batch_decode(text_inputs_id))

                        text_local_embedding_lower = self.model.embed_tokens(
                            text_inputs_id.to(image_embeds_mean.device)
                        )
                        # print("text_local_embedding_lower",text_local_embedding_lower.shape)

                        # text_local_features = torch.mean(text_local_embedding, dim=1)
                        text_local_features_lower = torch.mean(
                            text_local_embedding_lower, dim=0
                        )
                        # print("cos sim",torch.nn.functional.cosine_similarity(text_local_features,text_local_features_lower))
                        # print("text_local_features_lower",text_local_features_lower)

                        image_local_features = image_embeds[ocr_index][block_indices]
                        # print("image_local_features",image_local_features.shape)
                        image_local_features = torch.mean(image_local_features, dim=0)

                        # print("image_local_features",image_local_features.shape)
                        # print("text_local_features_lower",text_local_features_lower.shape)
                        image_local_features_mean.append(image_local_features)
                        text_local_features_mean.append(text_local_features_lower)
                        image_text_local_weight.append(torch.tensor(confidence_score))

                        # draw = PIL.ImageDraw.Draw(image, "RGBA")

                        # block_width = image_width / 14
                        # block_height = image_height / 14

                        # # 将对应的块涂上半透明的蓝色
                        # for index in block_indices:
                        #     i = index // 14
                        #     j = index % 14
                        #     block_x1 = j * block_width
                        #     block_y1 = i * block_height
                        #     block_x2 = block_x1 + block_width
                        #     block_y2 = block_y1 + block_height

                        #     draw.rectangle([block_x1, block_y1, block_x2, block_y2], fill=(0, 0, 255, 64))
                        # draw.rectangle(bbox, outline="red",width=3)

                        # boxes = [line[0] for line in result]
                        # txts = [line[1][0] for line in result]
                        # scores = [line[1][1] for line in result]
                        # image_save_path = f'path/to/codes/LLaVA/output/3/{random.randint(0, 1000000)}.png'
                        # im_show = draw_ocr(image, boxes, txts, scores, font_path=r'path/to/codes/LLaVA/simfang.ttf')
                        # im_show = PIL.Image.fromarray(im_show)
                        # image.save(image_save_path)
                        # im_show.save(image_save_path)
                        # if len(result) > 120:
                        #     result = result[:120]
            if image_local_features_mean and text_local_features_mean:
                contra_temperature_local = 0.5
                contra_loss_weight_local = 0.01
                similarity_matrix_local = F.cosine_similarity(
                    torch.stack(image_local_features_mean),
                    torch.stack(text_local_features_mean),
                )
                similarity_matrix_local = torch.exp(
                    similarity_matrix_local / contra_temperature_local
                )

                positives_local = similarity_matrix_local

                loss_local = -torch.log(positives_local)

                # Add Weight
                # ocr_weight = torch.tensor(image_text_local_weight).to(loss_local.device)
                # ocr_weight = ocr_weight + (1 - ocr_weight.mean())

                # print("ocr_weight",ocr_weight)
                # print("loss_local",loss_local)
                # loss_local = ocr_weight * loss_local

                contra_loss_local = torch.sum(loss_local) / len(
                    image_local_features_mean
                )
                print("contra_loss_local", contra_loss_local)

                loss += contra_loss_weight_local * contra_loss_local

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

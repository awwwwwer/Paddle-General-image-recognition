# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

from ppcls.arch import backbone
from ppcls.utils.save_load import load_dygraph_pretrain
import paddle
import paddle.nn.functional as F
from paddle.jit import to_static
from paddleslim.dygraph.quant import QAT

from pact_helper import get_default_quant_config


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("-o", "--output_path", type=str, default="./inference")
    parser.add_argument("--class_dim", type=int, default=1000)
    parser.add_argument("--load_static_weights", type=str2bool, default=False)
    parser.add_argument("--img_size", type=int, default=224)

    return parser.parse_args()


class Net(paddle.nn.Layer):
    def __init__(self, net, class_dim, model=None):
        super(Net, self).__init__()
        self.pre_net = net(class_dim=class_dim)
        self.model = model

    def forward(self, inputs):
        x = self.pre_net(inputs)
        if self.model == "GoogLeNet":
            x = x[0]
        x = F.softmax(x)
        return x


def main():
    args = parse_args()

    net = backbone.__dict__[args.model]
    model = Net(net, args.class_dim, args.model)

    # get QAT model
    quant_config = get_default_quant_config()
    # TODO(littletomatodonkey): add PACT for export model
    # quant_config["activation_preprocess_type"] = "PACT"
    quanter = QAT(config=quant_config)
    quanter.quantize(model)

    load_dygraph_pretrain(
        model.pre_net,
        path=args.pretrained_model,
        load_static_weights=args.load_static_weights)
    model.eval()

    save_path = os.path.join(args.output_path, "inference")
    quanter.save_quantized_model(
        model,
        save_path,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    print('inference QAT model is saved to {}'.format(save_path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate init_net.pbtxt and predict_net.pbtxt files from init_net.pb and predict_net.pb files."""
import argparse
import os.path

from caffe2.python import core
from caffe2.proto import caffe2_pb2

def getArgs():
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='model directory', default='.')
    args = parser.parse_args()
    return args

def main(args):

    # expand model path
    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.model)))

    init_def = caffe2_pb2.NetDef()
    with open(model_path + "/init_net.pb") as f:
        init_def.ParseFromString(f.read())
    predict_def = caffe2_pb2.NetDef()
    with open(model_path + "/predict_net.pb") as f:
        predict_def.ParseFromString(f.read())
    
    # write init_net.pbtxt
    filename = model_path + "/init_net.pbtxt"
    with open(filename, 'wb') as f:
        f.write(str(init_def))

    # write predict_net.pbtxt
    filename = model_path + "/predict_net.pbtxt"
    with open(filename, 'wb') as f:
        f.write(str(predict_def))

#    p = workspace.Predictor(init_net, predict_net)
#    print(type(p))
#    print([method_name for method_name in dir(p) if callable(getattr(p, method_name))])
#    print(dir(p))

if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    main(getArgs())

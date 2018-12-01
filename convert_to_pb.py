#!/usr/bin/env python
"""Generate init_net.pb and predict_net.pb files from init_net.pbtxt and predict_net.pbtxt files."""
import argparse
import os.path

from caffe2.python import core
from caffe2.proto import caffe2_pb2
from google.protobuf import text_format

def getArgs():
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='model directory', default='.')
    args = parser.parse_args()
    return args

def main(args):

    # expand model path
    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.model)))

    # read .pbtxt files using google.protobuf.text_format
    with open(model_path + "/init_net.pbtxt") as f:
        init_net = text_format.Parse(f.read(), caffe2_pb2.NetDef())
    with open(model_path + "/predict_net.pbtxt") as f:
        predict_net = text_format.Parse(f.read(), caffe2_pb2.NetDef())
    
    # write init_net.pb
    with open(model_path + "/init_net.pb", 'wb') as f:
        f.write(init_net.SerializeToString())

    # write predict_net.pbtxt
    with open(model_path + "/predict_net.pb", 'wb') as f:
        f.write(predict_net.SerializeToString())

if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    main(getArgs())

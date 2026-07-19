import torch
import collections
import numpy as np
import torch.nn as nn
from spikingjelly.clock_driven import neuron, encoding, functional

class bayesian_linear(torch.nn.Module):
    def __init__(self,dim_input, num_hidden_units, tau, v_threshold, v_reset):
        super(bayesian_linear, self).__init__()
        # self.num_hidden_units = num_hidden_units
        # self.num_layers = len(num_hidden_units)-1
        self.dim_input = dim_input
        self.num_hidden_units = num_hidden_units
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        #self.device = device
        self.num_layers = 1
        #self.device = torch.device("cuda:0")


    def forward(self, x, weight):
        self.weight=weight

        flatten = nn.Flatten()
        out=flatten.forward(x)
        w = self.weight['w1']
        b = self.weight['b1']
        # w.requires_grad_()
        # b.requires_grad_()
        # print("out.size():",out.size(),"self.weight:", w.size(),b.size())
        # print("self.weight:",self.weight.size())
        device = out.device
        w = w.to(device)
        b = b.to(device)
        out = torch.nn.functional.linear(input=out, weight=w, bias=b)

        self.lif_layer = neuron.LIFNode(tau=self.tau, v_threshold=self.v_threshold, v_reset=self.v_reset)
        # print("out0:",out)
        out = self.lif_layer.forward(out)

        return out

    def sample_nn_weight(self,meta_params):
        w = {}
        for key in meta_params['mean'].keys():
            eps_sampled = torch.randn_like(input=meta_params['mean'][key])
            w[key] = meta_params['mean'][key] + eps_sampled * torch.exp(meta_params['logSigma'][key])
        return w

    def get_weight_shape(self):
        w_shape = collections.OrderedDict()
        num_hidden_units = [self.dim_input]
        print("num_hidden_units:",num_hidden_units)
        num_hidden_units.append(self.num_hidden_units)

        w_shape['w1'] = (num_hidden_units[1], num_hidden_units[0])  # (output_dim, input_dim)
        w_shape['b1'] = num_hidden_units[1]  # (output_dim,)
        return w_shape

import helper.utils as u
import numpy as np
import theano
import theano.tensor.signal.pool as pool
import theano.tensor.nnet as nn
import theano.tensor as T
from helper.utils import init_weight,init_bias,get_err_fn
from theano.tensor.nnet import conv2d #Check if it is using gpu or not

class LogisticRegression(object):
    def __init__(self,  rng,input, n_in, n_out,W=None,b=None):
        shape=(n_in, n_out)
        if(W ==None):
            W = u.init_weight(shape=shape,rng=rng,name='W_xreg',sample='glorot')
            b=u.init_bias(n_out,rng=rng)
        self.W = W
        self.b = b
        self.y_pred = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]
        self.input = input

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,W=None,b=None,activation=T.tanh):
        self.input = input
        shape=[n_in,n_out]
        if(W ==None):
            W =u.init_weight(shape=shape,rng=rng,name="w_hid",sample="glorot")
            b=u.init_bias(n_out,rng)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]

class LSTMLayer(object):
    def __init__(self, rng, layer_id, n_in, n_lstm):
       layer_id=str(layer_id)
       self.n_in = n_in
       self.n_lstm = n_lstm
       self.W_xi = init_weight((self.n_in, self.n_lstm),rng=rng,name='W_xi_'+layer_id,sample= 'glorot')
       self.W_hi = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_hi_'+layer_id, sample='glorot')
       self.W_ci = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_ci_'+layer_id,sample= 'glorot')
       self.b_i  = init_bias(self.n_lstm,rng=rng, sample='zero',name='b_i_'+layer_id)
       self.W_xf = init_weight((self.n_in, self.n_lstm),rng=rng,name='W_xf_'+layer_id,sample= 'glorot')
       self.W_hf = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_hf_'+layer_id,sample= 'glorot')
       self.W_cf = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_cf_'+layer_id, sample='glorot')
       self.b_f = init_bias(self.n_lstm, rng=rng,sample='one',name='b_f_'+layer_id)
       self.W_xc = init_weight((self.n_in, self.n_lstm),rng=rng,name='W_xc_'+layer_id, sample='glorot')
       self.W_hc = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_hc_'+layer_id, sample='ortho')
       self.b_c = init_bias(self.n_lstm, rng=rng,sample='zero',name='b_c_'+layer_id)
       self.W_xo = init_weight((self.n_in, self.n_lstm),rng=rng,name='W_xo_'+layer_id,sample= 'glorot')
       self.W_ho = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_ho_'+layer_id, sample='glorot')
       self.W_co = init_weight((self.n_lstm, self.n_lstm),rng=rng,name='W_co_'+layer_id,sample= 'glorot')
       self.b_o = init_bias(self.n_lstm,rng=rng, sample='zero',name='b_o_'+layer_id)

       self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                      self.W_xf, self.W_hf, self.W_cf, self.b_f,
                      self.W_xc, self.W_hc, self.b_c,  self.W_xo,
                      self.W_ho, self.W_co, self.b_o]

    def run(self,x_t,h_tm1,c_tm1):
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1,self.W_ci) + self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c)
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t,self.W_co)  + self.b_o)
        h_t = o_t * T.tanh(c_t)
        y_t = h_t
        return [h_t,c_t,y_t]

class ConvLayer(object):
    def __init__(self, rng, input,filter_shape,input_shape,border_mode,subsample, activation=nn.relu,W=None,b=None,only_conv=0):
        # e.g. input_shape= (samples, channels, rows, cols)
        #    assert border_mode in {'same', 'valid'}

        self.input = input
        nb_filter=filter_shape[0]

        # W,b=None,None
        if(W ==None):
            W =u.init_weight(filter_shape,rng=rng, name="w_conv", sample='glorot')
            b=u.init_bias(nb_filter,rng=rng)
        self.W = W
        self.b = b

        b_mode=border_mode
        if(border_mode=='same'):
            b_mode='half'

        #image_shape: (batch size, num input feature maps,image height, image width)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=input_shape,
            border_mode=b_mode,subsample=subsample

        )

        if border_mode == 'same':
            if filter_shape[2] % 2 == 0:
                conv_out = conv_out[:, :, :(input.shape[2] + subsample[0] - 1) // subsample[0], :]
            if filter_shape[3] % 2 == 0:
                conv_out = conv_out[:, :, :, :(input.shape[3] + subsample[1] - 1) // subsample[1]]

        if(only_conv==0):
            output = conv_out + b.dimshuffle('x', 0, 'x', 'x')
            self.output = activation(output, 0)
        else:
            self.output = conv_out


        # parameters of the model
        self.params = [self.W, self.b]

        rows = input_shape[2]
        cols = input_shape[3]

        rows = u.conv_output_length(rows, filter_shape[2],border_mode, subsample[0])
        cols = u.conv_output_length(cols, filter_shape[3], border_mode, subsample[1])

        self.output_shape=(input_shape[0], nb_filter, rows, cols)

class DropoutLayer(object):
    def __init__(self, rng, input, prob,is_train,mask=None):
        retain_prob = 1. - prob
        if(mask==None):
            ret_input =input *rng.binomial(size=input.shape, p=retain_prob, dtype=input.dtype)
        else:
            ret_input=input* mask
        test_output = input*retain_prob
        self.output= T.switch(T.neq(is_train, 0), ret_input, test_output)

class PoolLayer(object):
    def __init__(self, input,pool_size,input_shape, pool_mode="max"):
        pooled_out = pool.pool_2d(
            input=input,
            ds=pool_size,
            mode=pool_mode,
            ignore_border=True
        )


        rows = input_shape[2]
        cols = input_shape[3]
        rows = rows /  pool_size[0]
        cols = cols /  pool_size[1]
        self.output_shape= (input_shape[0], input_shape[1], rows, cols)

        self.output= pooled_out

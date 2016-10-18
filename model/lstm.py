import numpy as np
import theano
from layers import LSTMLayer,DropoutLayer
import theano.tensor as T
from theano import shared
from helper.utils import init_weight,init_bias,get_err_fn,count_params
from helper.optimizer import RMSprop

dtype = T.config.floatX

class lstm:
   def __init__(self,rng, params,cost_function='mse',optimizer = RMSprop):
       batch_size=params['batch_size']
       sequence_length=params["seq_length"]

       lr=params['lr']
       self.n_in = 1024
       self.n_lstm = params['n_hidden']
       self.n_out = params['n_output']

       self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
       self.b_y = init_bias(self.n_out,rng=rng, sample='zero')

       layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)

       self.params = layer1.params
       self.params.append(self.W_hy)
       self.params.append(self.b_y)

       def step_lstm(x_t,h_tm1_1,c_tm1_1):
           [h_t_1,c_t_1,y_t_1]=layer1.run(x_t,h_tm1_1,c_tm1_1)
           y = T.dot(y_t_1, self.W_hy) + self.b_y
           return [h_t_1,c_t_1,y]

       X = T.tensor3() # batch of sequence of vector
       Y = T.tensor3() # batch of sequence of vector
       is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
       H = T.matrix(name="H",dtype=dtype) # initial hidden state
       C = T.matrix(name="C",dtype=dtype) # initial hidden state

       noise= rng.normal(size=(batch_size,sequence_length,self.n_in), std=0.0002, avg=0.0,dtype=theano.config.floatX)
       X_train=noise+X

       X_tilde= T.switch(T.neq(is_train, 0), X_train, X)

       [h_t_1,c_t_1,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[X_tilde.dimshuffle(1,0,2)],
                                         outputs_info=[H, C, None])

       self.output = y_vals.dimshuffle(1,0,2)
       cost=get_err_fn(self,cost_function,Y)

       _optimizer = optimizer(
            cost,
            self.params,
            lr=lr
        )

       self.train = theano.function(inputs=[X,Y,is_train,H,C],outputs=[cost,h_t_1[-1],c_t_1[-1]],updates=_optimizer.getUpdates(),allow_input_downcast=True)
       self.predictions = theano.function(inputs = [X,is_train,H,C], outputs = [self.output,h_t_1[-1],c_t_1[-1]],allow_input_downcast=True)
       self.n_param=count_params(self.params)

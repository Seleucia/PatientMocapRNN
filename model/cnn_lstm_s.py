from theano import shared
import numpy as np
import theano
import theano.tensor as T
from layers import ConvLayer,PoolLayer,HiddenLayer,DropoutLayer,LSTMLayer
import theano.tensor.nnet as nn
from helper.utils import init_weight,init_bias,get_err_fn,count_params, do_nothing
from helper.optimizer import RMSprop

# theano.config.exception_verbosity="high"
dtype = T.config.floatX

class cnn_lstm_s(object):
    def __init__(self,rng,params,cost_function='mse',optimizer = RMSprop):

        lr=params["lr"]
        n_lstm=params['n_hidden']
        n_out=params['n_output']
        batch_size=params["batch_size"]
        sequence_length=params["seq_length"]

        # minibatch)
        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

        #CNN global parameters.
        subsample=(1,1)
        p_1=0.5
        border_mode="valid"
        cnn_batch_size=batch_size*sequence_length
        pool_size=(2,2)

        #Layer1: conv2+pool+drop
        filter_shape=(64,1,9,9)
        input_shape=(cnn_batch_size,1,120,60) #input_shape= (samples, channels, rows, cols)
        input= X.reshape(input_shape)
        c1=ConvLayer(rng, input,filter_shape, input_shape,border_mode,subsample, activation=nn.relu)
        p1=PoolLayer(c1.output,pool_size=pool_size,input_shape=c1.output_shape)
        dl1=DropoutLayer(rng,input=p1.output,prob=p_1,is_train=is_train)

        #Layer2: conv2+pool
        filter_shape=(128,p1.output_shape[1],3,3)
        c2=ConvLayer(rng, dl1.output, filter_shape,p1.output_shape,border_mode,subsample, activation=nn.relu)
        p2=PoolLayer(c2.output,pool_size=pool_size,input_shape=c2.output_shape)


        #Layer3: conv2+pool
        filter_shape=(128,p2.output_shape[1],3,3)
        c3=ConvLayer(rng, p2.output,filter_shape,p2.output_shape,border_mode,subsample, activation=nn.relu)
        p3=PoolLayer(c3.output,pool_size=pool_size,input_shape=c3.output_shape)

        #Layer4: hidden
        n_in= reduce(lambda x, y: x*y, p3.output_shape[1:])
        x_flat = p3.output.flatten(2)
        h1=HiddenLayer(rng,x_flat,n_in,1024,activation=nn.relu)
        n_in=1024
        rnn_input = h1.output.reshape((batch_size,sequence_length, n_in))


        #Layer5: LSTM
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_hy = init_weight((self.n_lstm, self.n_out), rng=rng,name='W_hy', sample= 'glorot')
        self.b_y = init_bias(self.n_out,rng=rng, sample='zero')

        layer1=LSTMLayer(rng,0,self.n_in,self.n_lstm)

        self.params = layer1.params
        self.params.append(self.W_hy)
        self.params.append(self.b_y)

        def step_lstm(x_t,h_tm1,c_tm1):
           [h_t,c_t,y_t]=layer1.run(x_t,h_tm1,c_tm1)
           y = T.dot(y_t, self.W_hy) + self.b_y
           return [h_t,c_t,y]

        H = T.matrix(name="H",dtype=dtype) # initial hidden state
        C = T.matrix(name="C",dtype=dtype) # initial hidden state

        [h_t,c_t,y_vals], _ = theano.scan(fn=step_lstm,
                                         sequences=[rnn_input.dimshuffle(1,0,2)],
                                         outputs_info=[H, C, None])

        self.output = y_vals.dimshuffle(1,0,2)

        self.params =c1.params+c2.params+c3.params+h1.params+self.params

        cost=get_err_fn(self,cost_function,Y)
        L2_reg=0.0001
        L2_sqr = theano.shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param ** 2))

        cost += L2_reg*L2_sqr
        _optimizer = optimizer(cost, self.params, lr=lr)
        self.train = theano.function(inputs=[X,Y,is_train,H,C],outputs=[cost,h_t[-1],c_t[-1]],updates=_optimizer.getUpdates(),allow_input_downcast=True)
        self.predictions = theano.function(inputs = [X,is_train,H,C], outputs = [self.output,h_t[-1],c_t[-1]],allow_input_downcast=True)
        self.n_param=count_params(self.params)
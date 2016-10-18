import numpy
import math
from random import uniform
import theano
import theano.tensor as T
import os
import datetime
import numpy as np
from theano import shared
from random import randint
import pickle

dtype = T.config.floatX

def rescale_weights(params, incoming_max):
    incoming_max = np.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * np.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=T.config.floatX)

def conv_output_length(input_length, filter_size, border_mode, stride):
    #https://github.com/fchollet/keras/
    if input_length is None:
        return None
    assert border_mode in {'full', 'valid','same'}
    if border_mode == 'full':
        output_length = input_length+ filter_size-1
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    elif border_mode == 'same':
        output_length = input_length
    return (output_length + stride - 1) // stride

      # 'valid'only apply filter to complete patches of the image. Generates
      #  output of shape: image_shape - filter_shape + 1.
      #  'full' zero-pads image to multiple of filter shape to generate output
      #  of shape: image_shape + filter_shape - 1.

def do_nothing(x):
    return x

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        fan_in = np.prod(shape[1:])
        fan_out = shape[0]
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def init_bias(n_out,rng, sample='zero',name='b'):
    if sample == 'zero':
        b = np.zeros((n_out,), dtype=dtype)
    elif sample == 'one':
        b = np.ones((n_out,), dtype=dtype)
    elif sample == 'uni':
        b=shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_out)))
    elif sample == 'big_uni':
        b=np.asarray(rng.uniform(low=-5,high=5,size=n_out),dtype=dtype);
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)
    b = theano.shared(value=b, name=name)
    return b

def init_weight(shape, rng,name, sample='glorot', seed=None):
    #https://github.com/fchollet/keras/
    if sample == 'unishape':
        fan_in, fan_out =get_fans(shape)
        values = rng.uniform(
            low=-np.sqrt(6. / (fan_in + fan_out)),
            high=np.sqrt(6. / (fan_in + fan_out)),
            size=shape).astype(dtype)

    elif sample == 'svd':
        values = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]

    elif sample == 'uni':
        values = rng.uniform(low=-0.1, high=0.1, size=shape).astype(dtype)

    elif sample == 'glorot':
        ''' Reference: Glorot & Bengio, AISTATS 2010
        '''
        fan_in, fan_out =get_fans(shape)
        s = np.sqrt(2. / (fan_in + fan_out))
        values=np.random.normal(loc=0.0, scale=s, size=shape).astype(dtype)
    elif sample == 'ortho':
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
        '''
        scale=1.1
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        values= np.asarray(scale * q[:shape[0], :shape[1]], dtype=dtype)

    elif sample == 'ortho1':
        W = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        u, s, v = numpy.linalg.svd(W)
        values= u.astype(dtype)

    elif sample == 'zero':
        values = np.zeros(shape=shape, dtype=dtype)

    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)

    return shared(values, name=name, borrow=True)

def prep_pred_file(params):
    f_dir=params["wd"]+"/pred/";
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    f_dir=params["wd"]+"/pred/"+params["model"];
    if not os.path.exists(f_dir):
            os.makedirs(f_dir)
    map( os.unlink, (os.path.join( f_dir,f) for f in os.listdir(f_dir)) )

def write_auto_pred(est,F_list,params):
    f_dir="/mnt/hc/auto/"
    ist=0
    for b in range(len(est)):
        action=F_list[b].split('/')[-2]
        sb=F_list[b].split('/')[-3]
        if not os.path.exists(f_dir+sb):
                os.makedirs(f_dir+sb)
        if not os.path.exists(f_dir+sb+'/'+action):
                os.makedirs(f_dir+sb+'/'+action)
        vec=est[b]
        vec_str = '\n'.join(['%f' % num for num in vec])
        p_file=f_dir+sb+'/'+action+'/'+os.path.basename(F_list[b])
        if os.path.exists(p_file):
            print p_file
        with open(p_file, "a") as p:
            p.write(vec_str)


def write_pred(est,bindex,G_list,params):
    batch_size=est.shape[0]
    seq_length=est.shape[1]
    s_index=params["batch_size"]*bindex*seq_length
    f_dir=params["wd"]+"/pred/"+params["model"]+"/"
    for b in range(batch_size):
        for s in range(seq_length):
            diff_vec=est[b][s]*2
            vec_str = ' '.join(['%.6f' % num for num in diff_vec])
            p_file=f_dir+os.path.basename(G_list[s_index])
            with open(p_file, "a") as p:
                p.write(vec_str)
            s_index+=1

def get_loss(params,gt,est):
    gt= np.asarray(gt)
    # print(gt.shape)
    batch_size=gt.shape[0]
    loss=[]
    if(len(gt.shape)==2):
        for b in range(batch_size):
            diff_vec=np.abs(gt[b].reshape(params['n_output']/3,3) - est[b].reshape(params['n_output']/3,3)) #13*3
            diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
            sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
            loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)
    else:
        seq_length=gt.shape[1]
        for b in range(batch_size):
            for s in range(seq_length):
                diff_vec=np.abs(gt[b][s].reshape(params['n_output']/3,3) - est[b][s].reshape(params['n_output']/3,3)) #13*3
                diff_vec=diff_vec[~np.any(np.isnan(diff_vec), axis=1)]
                sq_m=np.sqrt(np.sum(diff_vec**2,axis=1))
                loss.append(np.mean(sq_m))
        loss=np.nanmean(loss)

    return (loss)


def start_log(params):
    log_file=params["log_file"]
    create_file(log_file)
    ds= get_time()
    log_write("Run Id: %s"%(params['rn_id']),params)
    log_write("Deployment notes: %s"%(params['notes']),params)
    log_write("Running mode: %s"%(params['run_mode']),params)
    log_write("Running model: %s"%(params['model']),params)
    log_write("Batch size: %s"%(params['batch_size']),params)
    log_write("Load mode: %s"%(params['load_mode']),params)
    log_write("Sequence size: %s"%(params['seq_length']),params)
    log_write("Learnig rate: %s"%(params['lr']),params)
    log_write("Data Dir: %s"%(params['data_dir']),params)

    log_write("Starting Time:%s"%(ds),params)
    log_write("size of training data:%f"%(params["len_train"]),params)
    log_write("size of test data:%f"%(params["len_test"]),params)

def get_time():
    return str(datetime.datetime.now().time()).replace(":","-").replace(".","-")

def create_file(log_file):
    log_dir= os.path.dirname(log_file)
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if(os.path.isfile(log_file)):
        with open(log_file, "w"):
            pass
    else:
        os.mknod(log_file)

def log_to_file(str,params):
    with open(params["log_file"], "a") as log:
        log.write(str)

def log_write(str,params):
    print(str)
    ds= get_time()
    str=ds+" | "+str+"\n"
    log_to_file(str,params)

def count_params(model_params):
    coun_params=np.sum([np.prod(p.shape.eval()) for p in model_params ])
    return coun_params

def write_params(mparams,params,ext):
    wd=params["wd"]
    filename=params['model']+"_"+params["rn_id"]+"_"+ext
    fpath=wd+"/cp/"+filename
    if os.path.exists(fpath):
        os.remove(fpath)
    with open(fpath,"a") as f:
        pickle.dump([param.get_value() for param in mparams], f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved: "+filename)

def read_params(params):
    wd=params["wd"]

    if(len(params['mfile'].split(','))<2):
        with open(wd+"/cp/"+params['mfile']) as f:
            mparams=pickle.load(f)
            return mparams

    if(params['model']=='cnn_lstm_auto'):
        lst=params['mfile'].split(',')
        mparams=[]
        with open(wd+"/cp/"+lst[0]) as f:
            mparams.extend(pickle.load(f))

        with open(wd+"/cp/"+lst[1]) as f:
            mparams.extend(pickle.load(f))
        return mparams
    elif(params['model']=='cnn_decoder'):
        lst=params['mfile'].split(',')
        mparams=[]
        with open(wd+"/cp/"+lst[0]) as f:
            mparams.extend(pickle.load(f))
        auto=[]
        with open(wd+"/cp/"+lst[1]) as f:
            auto=pickle.load(f)
        mparams.append(auto[2].T)
        mparams.append(auto[4])
        mparams.append(auto[0].T)
        mparams.append(auto[5])
        return mparams
    return None

def set_params(model,mparams):
    counter=0
    # for p in mparams[0:-2]:
    for p in mparams:
        assert (list(p.shape)==model.params[counter].shape.eval().tolist())
        model.params[counter].set_value(p)
        counter=counter+1
    return model
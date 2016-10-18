
from model.lstm import  lstm
from model.cnn_lstm_s import  cnn_lstm_s
from model.lstm2erd import lstm2erd
from model.gru import gru
from model.cnn_lstm import cnn_lstm
from model.cnn import cnn
from model.cnn2 import cnn2
from model.cnn3 import cnn3
from model.cnn4 import cnn4
from model.cnn5 import cnn5
from model.cnn6 import cnn6
from model.cnn7 import cnn7
from model.cnn8 import cnn8
from model.cnn9 import cnn9
from model.autoencoder import autoencoder
from model.cnn_decoder import cnn_decoder
from model.lstm_auto import lstm_auto
from model.lstm_joints import lstm_joints
from model.lstm_mdn import lstm_mdn
from model.lstm_3layer_mdn import lstm_3layer_mdn
from model.lstm_3layer_joints import lstm_3layer_joints
from model.lstm_3layer_joints2 import lstm_3layer_joints2
from model.cnn_lstm_auto import cnn_lstm_auto
from model.lstm_auto_3layer import lstm_auto_3layer
from model.lstm_skelton import lstm_skelton
from model.real_rcnn import real_rcnn
from helper.optimizer import ClipRMSprop, RMSprop,Adam,AdamClip
import helper.utils as u

def get_model(params,rng):
    if(params["model"]=="lstm"):
        model = lstm(rng,params, optimizer=Adam)
    elif(params["model"]=="cnn_lstm_s"):
        model = cnn_lstm_s(rng,params, optimizer=Adam)
    elif(params["model"]=="cnn_lstm"):
        model = cnn_lstm(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="real_rcnn"):
        model = real_rcnn(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn"):
        model = cnn(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn2"):
        model = cnn2(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn3"):
        model = cnn3(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn4"):
        model = cnn4(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn5"):
        model = cnn5(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn6"):
        model = cnn6(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn7"):
        model = cnn7(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="cnn8"):
        model = cnn8(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn9"):
        model = cnn9(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="autoencoder"):
        model = autoencoder(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="cnn_decoder"):
        model = cnn_decoder(rng=rng,params=params,optimizer=Adam)
    elif(params["model"]=="lstm_auto"):
        model = lstm_auto(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_joints"):
        model = lstm_joints(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_3layer_joints"):
        model = lstm_3layer_joints(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_3layer_joints2"):
        model = lstm_3layer_joints2(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="cnn_lstm_auto"):
        model = cnn_lstm_auto(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_auto_3layer"):
        model = lstm_auto_3layer(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_skelton"):
        model = lstm_skelton(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_mdn"):
        model = lstm_mdn(rng=rng,params=params,optimizer=AdamClip)
    elif(params["model"]=="lstm_3layer_mdn"):
        model = lstm_3layer_mdn(rng=rng,params=params,optimizer=AdamClip)
    else:
        raise Exception('Wrong model calling....') #
    return model

def get_model_pretrained(params,rng):
    mparams=u.read_params(params)
    model=get_model(params,rng)
    model=u.set_params(model,mparams)
    return model

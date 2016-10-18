from model.cnn_lstm import cnn_lstm
from model.lstm import lstm
from helper.optimizer import ClipRMSprop, RMSprop,Adam,AdamClip
import helper.utils as u

def get_model(params,rng):
    if(params["model"]=="lstm"):
        model = lstm(rng,params, optimizer=Adam)
    elif(params["model"]=="cnn_lstm"):
        model = cnn_lstm(rng,params, optimizer=Adam)
    else:
        raise Exception('Wrong model calling....') #
    return model

def get_model_pretrained(params,rng):
    mparams=u.read_params(params)
    model=get_model(params,rng)
    model=u.set_params(model,mparams)
    return model

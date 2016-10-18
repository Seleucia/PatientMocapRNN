import os
import utils
import platform

def get_params():
   global params
   params={}
   params['run_mode']=0 #0,full,1:resume, 2 = combine models

   params["rn_id"]="lstm" #running id, model
   params["load_mode"]=0#0=full training,1=only test set,2=full dataset, 3=hyper param searc
   params["notes"]="LSTM training with batches and new loading..." #running id
   params["model"]="lstm"#kccnr,dccnr
   params["optimizer"]="Adam" #1=classic kcnnr, 2=patch, 3=conv, 4 =single channcel
   params['mfile']=""

   params['lr']=0.0001
   params['mtype']="rnn"#rnn,cnnrnn
   params['shufle_data']=0
   params['nlayer']= 1 #LSTM
   params['batch_size']=5
   params['seq_length']= 50
   params['reset_state']= 10#-1=Never, n=every n batch
   params["corruption_level"]=0.5

   #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   params["model_file"]=wd+"/cp/"


   params['momentum']=0.9    # the params for momentum
   params['n_epochs']=25600
   params['n_hidden']= 1024
   params['n_output']= 48

   params["data_dir"]="/mnt/Data1/hc/joints16/" #joints with 16, cnn+lstm
   params['max_count']=-1

   return params

def update_params(params):
   params['log_file']=params["wd"]+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+str(params['run_mode'])+"_"+utils.get_time()+".txt"
   return params

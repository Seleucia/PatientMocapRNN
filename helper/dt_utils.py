import glob
import random
from random import shuffle
import math
from PIL import Image
import os
import numpy
import collections
from multiprocessing.dummy import Pool as ThreadPool
import theano
import theano.tensor as T
from random import randint
dtype = T.config.floatX

def load_pose(params,db_train=dict(),db_test=dict(),):
   data_dir=params["data_dir"]
   max_count=params["max_count"]
   seq_length=params["seq_length"]
   load_mode=params["load_mode"]
   sindex=params["sindex"]
   get_flist=False

   if params['mtype']=="cnnrnn":
       mode=0
       db_train,X_train,Y_train,F_list_train,G_list_train,S_Train_list=multi_thr_read_full_joints_sequence_cnn(db_train,data_dir,max_count,seq_length,sindex,mode,get_flist)
       mode=1
       sindex=0
       (db_test,X_test,Y_test,F_list_test,G_list_test,S_Test_list)= multi_thr_read_full_joints_sequence_cnn(db_test,data_dir,max_count,seq_length,sindex,mode,get_flist)
       return (db_train,X_train,Y_train,S_Train_list,F_list_train,G_list_train,db_test,X_test,Y_test,S_Test_list,F_list_test,G_list_test)
   else:
       return load_lstm_dataset()


def multi_thr_read_full_joints_sequence_cnn(base_file,max_count,p_count,sindex,mode,get_flist=False):
    joints_file=base_file
    img_folder=base_file.replace('joints16','h36m_rgb_img_crop')
    if mode==0:#load training data.
        lst_act=['S1','S5','S6','S7','S8']
    elif mode==1:#load test data
        lst_act=['S9','S11']
    elif mode==2:#load full data
        lst_act=['S1','S5','S6','S7','S8','S9','S11']
    else:
        raise Exception('You should pass mode argument for data loading.!') #
    X_D=[]
    Y_D=[]
    F_L=[]
    G_L=[]
    S_L=[]
    seq_id=0
    for actor in lst_act:
        tmp_folder=base_file+actor+"/"
        lst_sq=os.listdir(tmp_folder)
        for sq in lst_sq:
            X_d=[]
            Y_d=[]
            F_l=[]
            seq_id+=1
            tmp_folder=joints_file+actor+"/"+sq+"/"
            tmp_folder_img=img_folder+actor+"/"+sq.replace('.cdf','')+"/"
            id_list=os.listdir(tmp_folder)
            if os.path.exists(tmp_folder_img)==False:
                continue
            img_count=len(os.listdir(tmp_folder_img))
            min_count=img_count
            if(len(id_list)<img_count):
                min_count=len(id_list)
            if min_count==0:
                continue
            seq_id+=1
            id_list=id_list[0:min_count]
            joint_list=[tmp_folder + p1 for p1 in id_list]
            img_list=[img_folder+actor+'/'+sq.replace('.cdf','')+'/frame_'+(p1.replace('.txt','')).zfill(5)+'.png' for p1 in id_list]
            pool = ThreadPool(1000)
            results = pool.map(load_file, joint_list)
            pool.close()

            for r in range(len(results)):
                rs=results[r]
                f=img_list[r]
                Y_d.append(rs)
                F_l.append(f)
                if len(Y_d)==p_count and p_count>0:
                        Y_D.append(Y_d)
                        F_L.append(F_l)
                        S_L.append(seq_id)
                        Y_d=[]
                        F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)
        if(len(Y_d)>0):
            residual=len(Y_d)%p_count
            residual=p_count-residual
            y=residual*[Y_d[-1]]
            f=residual*[F_l[-1]]
            Y_d.extend(y)
            F_l.extend(f)
            if len(Y_d)==p_count and p_count>0:
                S_L.append(seq_id)
                Y_D.append(Y_d)
                F_L.append(F_l)
                Y_d=[]
                F_l=[]
                if len(Y_D)>=max_count:
                    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)


    return (numpy.asarray(X_D,dtype=numpy.float32),numpy.asarray(Y_D,dtype=numpy.float32),numpy.asarray(F_L),G_L,S_L)

def load_lstm_dataset(base_file,max_count,p_count,sindex):
    X_train,Y_train,F_list_train,G_list_train=load_test_LSTM_pose(base_file,max_count,p_count,sindex)
    X_test,Y_test,F_list_test,G_list_test=load_train_LSTM_pose(base_file,max_count,p_count,sindex)
    return (X_train,Y_train,F_list_train,G_list_train,X_test,Y_test,F_list_test,G_list_test)

def load_test_LSTM_pose(base_file,max_count,p_count,sindex):
    base_file=base_file+"test/"
    ft_prefix="feature_"
    gt_prefix="label_"
    sql=range(13,15,1);
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    F_L=[]
    G_L=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(sindex,1801,1):
           if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)
           fl=base_file+ft_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
           gl=base_file+gt_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
           if not os.path.isfile(gl):
               continue
           with open(gl, "rb") as f:
              data=f.read().strip().split(' ')
              y_d= [float(val) for val in data]
              # if(numpy.isnan(numpy.sum(y_d))):
              #     continue;
              Y_d.append(numpy.asarray(y_d))
           with open(fl, "rb") as f:
               data=f.read().strip().split(' ')
               x_d = [float(val) for val in data]
               X_d.append(numpy.asarray(x_d))
           F_L.append(fl)
           G_L.append(gl)
           if len(X_d)>p_count and p_count>0:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

        if p_count==-1 and len(X_d)>0:
          X_D.append(X_d)
          Y_D.append(Y_d)
          p_index=p_index+1


    return (numpy.asarray(X_D),numpy.asarray(Y_D),F_L,G_L)

def load_train_LSTM_pose(base_file,max_count,p_count,sindex):
    base_file=base_file+"train/"
    ft_prefix="feature_"
    gt_prefix="label_"
    sql=range(1,14,1);
    X_D=[]
    Y_D=[]
    p_index=0
    X_d=[]
    Y_d=[]
    for sq in sql:
        X_d=[]
        Y_d=[]
        for fm in range(sindex,1801,1):
            if len(X_D)>max_count:
               return (numpy.asarray(X_D),numpy.asarray(Y_D))
            fl=base_file+ft_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            gl=base_file+gt_prefix+"seq"+str(sq)+"_frame"+str(fm)+".txt"
            if not os.path.isfile(fl):
               continue

            with open(gl, "rb") as f:
              data=f.read().strip().split(' ')
              y_d= [float(val) for val in data]
              # if(numpy.isnan(numpy.sum(y_d))):
              #     continue;
              Y_d.append(numpy.asarray(y_d))

            with open(fl, "rb") as f:
               data=f.read().strip().split(' ')
               x_d = [float(val) for val in data]
               X_d.append(numpy.asarray(x_d))

            if len(X_d)>p_count:
               X_D.append(X_d)
               Y_D.append(Y_d)
               X_d=[]
               Y_d=[]
               p_index=p_index+1

    return (numpy.asarray(X_D),numpy.asarray(Y_D))

def multi_thr_load_cnn_lstm_batch(my_list):
    lst=my_list
    pool = ThreadPool(len(lst))
    results = pool.map(load_file_cnn_lstm_patch, lst)
    pool.close()
    return numpy.asarray(results)

def load_file(fl):
    with open(fl, "rb") as f:
        data=f.read().strip().split(' ')
        y_d= [numpy.float32(val) for val in data]
        y_d=numpy.asarray(y_d,dtype=numpy.float32)/1000
        f.close()
        return y_d

def load_file_cnn_lstm_patch(fl):
    patch_margin=(0,0)
    orijinal_size=(128,128)
    size=(112,112)
    x1=randint(patch_margin[0],orijinal_size[0]-(patch_margin[0]+size[0]))
    x2=x1+size[0]
    y1=randint(patch_margin[1],orijinal_size[1]-(patch_margin[1]+size[1]))
    y2=y1+size[1]
    normalizer=255
    patch_loc= (x1,y1,x2,y2)
    img = Image.open(fl)
    img = img.crop(patch_loc)
    arr=numpy.asarray(img)
    arr.flags.writeable = True
    arr.flags.writeable = True
    arr=arr-(130.70753799,84.31474484,72.801691)
    arr/=normalizer
    arr=numpy.squeeze(arr)
    arr=arr.reshape(3*112*112)
    return arr

def prepare_cnn_lstm_batch(index_train_list, minibatch_index, batch_size, S_Train_list, sid, H, C, F_list, params, Y, X):
    id_lst=index_train_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    tmp_sid=S_Train_list[(minibatch_index + 1) * batch_size-1]
    if(sid==0):
      sid=tmp_sid
    if(tmp_sid!=sid):
      sid=tmp_sid
      H=C=numpy.zeros(shape=(batch_size,params['n_hidden']), dtype=dtype) # resetting initial state, since seq change
    x_fl=F_list[id_lst][0]
    result=multi_thr_load_cnn_lstm_batch(my_list=x_fl)
    x_lst=[]
    x_lst.append(result)
    x=numpy.asarray(x_lst)
    y=Y[id_lst]
    return (sid,H,C,x,y)

def get_seq_indexes(params,S_L):
    bs=params['batch_size']
    new_S_L=[]
    counter=collections.Counter(S_L)
    lst=[list(t) for t  in counter.items()]
    a=numpy.asarray(lst)
    ss=a[a[:,1].argsort()][::-1]
    b_index=0
    new_index_lst=dict()
    b_index_lst=dict()

    for item in ss:
        seq_srt_intex= numpy.sum(a[0:item[0]-1],axis=0)[1]
        seq_end_intex= seq_srt_intex+item[1]
        sub_idx_lst=S_L[seq_srt_intex:seq_end_intex]
        new_S_L.extend(sub_idx_lst)

    for i in range(bs):
        b_index_lst[i]=0
    batch_inner_index=0
    for l_idx in range(len(new_S_L)):
        l=new_S_L[l_idx]
        if(l_idx>0):
            if(l!=new_S_L[l_idx-1]):
                for i in range(bs):
                    if(b_index>b_index_lst[i]):
                        b_index=b_index_lst[i]
                        batch_inner_index=i

        index=b_index*bs+batch_inner_index
        if(index in new_index_lst):
            print 'exist'
        new_index_lst[index]=l_idx
        b_index+=1
        b_index_lst[batch_inner_index]=b_index


    mx=max(b_index_lst.values())
    for b in b_index_lst.keys():
        b_index=b_index_lst[b]
        diff=mx-b_index
        if(diff>0):
            index=(b_index-1)*bs+b
            rpt=new_index_lst[index]
            for inc in range(diff):
                new_index=(b_index+inc)*bs+b
                new_index_lst[new_index]=rpt

    new_lst = collections.OrderedDict(sorted(new_index_lst.items())).values()
    return (new_lst,numpy.asarray(new_S_L))

def get_batch_indexes(S_list):
   SID_List=[]
   counter=collections.Counter(S_list)
   grb_count=counter.values()
   s_id=0
   index_list=range(0,len(S_list),1)
   for mx in grb_count:
       SID_List.extend(numpy.repeat(s_id,mx))
       s_id+=1
   return (index_list,SID_List)

def shuffle_in_unison_inplace(a, b):
   assert len(a) == len(b)
   p = numpy.random.permutation(len(a))
   return a[p],b[p]

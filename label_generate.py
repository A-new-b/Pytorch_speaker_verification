import os
import random
import time
import torch
import pdb
import numpy as np
from torch.utils.data import DataLoader
from utils import calculate_eer, step_decay, extract_all_feat
from tqdm import tqdm as tqdm
from hparam import hparam as hp
from data_load import VoxCeleb, VoxCeleb_utter
from speech_embedder_net import Resnet34_VLAD, SpeechEmbedder, GE2ELoss, SILoss, get_centroids, \
get_cossim, HybridLoss

import glob


torch.manual_seed(hp.seed)
np.random.seed(hp.seed)
random.seed(hp.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_path="./embedding"

def save_embedding(mode):
    
    model_path = os.path.join(hp.train.checkpoint_dir, hp.model.model_path)
    print('==> loading model({})'.format(model_path))
    embedder_net = Resnet34_VLAD()
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    # train_dataset = VoxCeleb_utter()
    # train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,num_workers=hp.train.num_workers, drop_last=True)
    # total_loss = 0

    if mode == "train":
        audio_path = glob.glob(hp.data.train_path + '/spec/*')
        for i, folder in enumerate(audio_path):
            save_np = None
            # print("%d speaker processing..."%i)
            folder = hp.data.train_path + '/spec/speaker%d'%i+'/*'
            print(folder)
            for j, utter_name in enumerate(glob.glob(folder)):
                if utter_name[-4:] == '.npy':
                    spec = np.load(os.path.join(hp.data.train_path,folder[:-1],utter_name), allow_pickle=True)
                    s1 = torch.Tensor(spec).unsqueeze(0)
                    e1 = embedder_net(s1.cuda())
                    e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
                    e1 = e1.cpu().detach().numpy()
                    if save_np is None:
                        save_np = e1
                    else:
                        save_np =  np.append(save_np,e1,axis=0)                
            np.save(save_path+'/train/train%d'%i, save_np)

    if mode == "test":
        test_file = glob.glob(hp.data.test_path + '/*')
        for i, file in enumerate(test_file):
            print("%d test processing..."%i)
            spec = np.load(os.path.join(hp.data.test_path,'utt%d.npy'%i), allow_pickle=True)
            s1 = torch.Tensor(spec).unsqueeze(0)
            e1 = embedder_net(s1.cuda())
            e1 = e1 / torch.norm(e1, dim=1).unsqueeze(1)
            e1 = e1.cpu().detach().numpy()
            e1 = e1.reshape(1,512)               
            np.save(save_path+'/test/test%d'%i, e1)


def load_train(i):
    embedder = np.load(save_path+'/train%d.npy'%i)
    return embedder

def classify():
    # a1 = load_train(1)

    test_em = glob.glob(save_path + '/test/*')
    train_em = glob.glob(save_path + '/train/*')

    f = open("result3.txt", 'w')
    
    for i, _ in enumerate(test_em):
        test = np.load(save_path + '/test/test%d'%i+'.npy')
        test = torch.from_numpy(test)
        result_list = None
        print("%d result processing..."%i)
        for j,_ in enumerate(train_em):
            spk = np.load(save_path + '/train/train%d.npy'%j)
            spk = torch.from_numpy(spk).transpose(0,1)
            result = torch.mm(test, spk)
            # result_value = torch.reshape(torch.mean(result),(1,1))
            result_value = torch.reshape(result.max(),(1,1))

            if result_list is None:
                result_list = result_value
            else:
            #     result = torch.cat((result,torch.mm(test, spk)),0)
                result_list = torch.cat((result_list,result_value),0)
        
        print(result_list.argmax())
        print(result_list.max())
        f.write(str(result_list.argmax())+" "+str(result_list.max())+"\n")
            

        # result = result.cpu().detach().numpy()
        # np.save(save_path+'/result/result%d'%i, result)

def classify_loss():

    model_path = os.path.join(hp.train.checkpoint_dir, hp.model.model_path)
    print('==> loading model({})'.format(model_path))
    embedder_net = Resnet34_VLAD()
    embedder_net = torch.nn.DataParallel(embedder_net)
    embedder_net = embedder_net.cuda()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    loss_fn = SILoss(hp.model.proj, 250).cuda()

    loss_fn.load_state_dict(torch.load(os.path.join(hp.train.checkpoint_dir, "loss_" + hp.model.model_path)))

    f = open("result.txt", 'w')
    test_file = glob.glob(hp.data.test_path_unprocessed + '/*')
    for j, file in enumerate(test_file):
        e1 = extract_all_feat(file, mode = 'train')
        s1 = torch.Tensor(e1).unsqueeze(0)
        e1 = embedder_net(s1.cuda())
        result_list = None

        for i in range(250):
            spk_test = torch.tensor([i]).cuda()
            test,_ = loss_fn(e1, spk_test)
            test = test.unsqueeze(0)

            if result_list is None:
                result_list = test
            else:
                result_list =  torch.cat((result_list,test),0)
        
        print(result_list.argmin())
        print(result_list.min())
        f.write(str(int(result_list.argmin()))+"\n")
if __name__=="__main__":
    # load_train()
    # save_embedding("test") 
    

    classify_loss()


    


    # loss_fn = SILoss(hp.model.proj, train_dataset.num_of_spk).cuda()

    # for batch_id, (mel_db_batch, spk_id) in enumerate(train_loader):
    #     embedder_net.train().cuda()
    #     mel_db_batch = mel_db_batch.cuda()

    #     spk_id = spk_id.cuda()
    #     mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2),mel_db_batch.size(3)))

    #     embeddings = embedder_net(mel_db_batch)

    #     loss,_ = loss_fn(embeddings, spk_id) #wants (Speaker, Utterances, embedding)
    #     print(mel_db_batch)
    #     print(spk_id)
    #     print(loss)

from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data
from utils import *
from graph import *

cpu=False
mode='train' #run this file in train of evaluation mode
save='logs' # Part of path to log file during training
log_freq=100 # Frequency to log training result

data_root='data/miniImagenet/'
n_shot=5
n_eval=15
n_workers=4
pin_mem=True
episode=50000 
episode_val=100
n_class=5
image_size=84
bn_eps=1e-3
bn_momentum=0.95
input_size=4
hidden_size=20
lr=1e-3
epoch=8
batch_size=25
grad_clip=0.5
val_freq=1000 # frequency to evaluate metalearner on validation set
seed=None
if seed is None:
        seed=random.randint(0, 1e3)

#test
resume=None #'logs-719/ckpts/meta-learner-42000.pth.tar'


 
def meta_test(eps, eval_loader, learner_w_grad, learner_wo_grad, metalearner,logger,dev,n_shot, n_class, n_eval):

    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        train_input = episode_x[:, :n_shot].reshape(-1, *episode_x.shape[-3:]).to(dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(n_class), n_shot)).to(dev) # [n_class * n_shot]
        test_input = episode_x[:, n_shot:].reshape(-1, *episode_x.shape[-3:]).to(dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(n_class), n_eval)).to(dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, epoch, batch_size)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=episode_val, phase='evaldone')


def train_learner(learner_w_grad, metalearner, train_input, train_target, epoch, batch_size):
    cI = metalearner.metalstm.cI.data #Get metalearner parameters
    hs = [None]
    for _ in range(epoch): # Loop through the dataset. Default is 8 times
        for i in range(0, len(train_input), batch_size): #train_input is one episode which has class*data/pclass. Take 8 samples from the episode 
            x = train_input[i:i+batch_size] #get batch from training episode 
            y = train_target[i:i+batch_size] #get batch of correcsponding label
            
            # get the loss/grad
            learner_w_grad.copy_flat_params(cI) # copy parameter of meta learner to learner model

            output = learner_w_grad(x) #Take x as the input dataset
            loss = learner_w_grad.criterion(output, y) # train using cross entropy loss
            acc = accuracy(output, y) #Check for prediction error
            learner_w_grad.zero_grad() #zero gradient before back propagation
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / batch_size for p in learner_w_grad.parameters()], 0)

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            #print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

    return cI



def main():

    # args, unparsed = FLAGS.parse_known_args()
    # if len(unparsed) != 0:
    #     raise NameError("Argument {} not recognized".format(unparsed))

    
    
    # random seed to ensure the reproducibility and consistency of result 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check device cpu or gpu
    if cpu:
        dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        # produce the same output for a given set of input and parameters
        torch.backends.cudnn.deterministic = True
        # automatically tune its algorithms to the specific GPU hardware
        torch.backends.cudnn.benchmark = False
        dev = torch.device('cuda')

    # instantiate GOATLogger from utils file
    logger = GOATLogger(seed, mode, save, log_freq)
    

    # Get preprocessed data from dataloader file 
    train_loader, val_loader, test_loader = prepare_data(data_root, n_shot, n_eval, image_size, n_workers,pin_mem, episode, episode_val, n_class)
    
    # Set up learner
    learner_w_grad = Learner(image_size, bn_eps, bn_momentum, n_class).to(dev)
    # copy learner_w_grad and its computation graph without gradients tp learner_wo_grad
    # This is needed for evaluation the test set 
    learner_wo_grad = copy.deepcopy(learner_w_grad)
    
    #Setting up meta-learner
    metalearner = MetaLearner(input_size, hidden_size, learner_w_grad.get_flat_params().size(0)).to(dev)
    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())

    # Set up loss, optimizer, learning rate scheduler
    optim = torch.optim.Adam(metalearner.parameters(), lr)
    
    # resumes training of a meta-learning algorithm from a saved checkpoint. 
    
    if resume:
        logger.loginfo("Initialized from: {}".format(resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim,resume, dev)

    #to making inference
    if mode == 'test':
        _ = meta_test(last_eps, test_loader, learner_w_grad, learner_wo_grad, metalearner,logger, dev,n_shot, n_class, n_eval)
        return

    best_acc = 0.0
    logger.loginfo("Start training")
    # Meta-training
    for eps, (episode_x, episode_y) in enumerate(train_loader):
        # episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED

        # get train and test data and labels
        train_input = episode_x[:, :n_shot].reshape(-1, *episode_x.shape[-3:]).to(dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(n_class), n_shot)).to(dev) # [n_class * n_shot]
        test_input = episode_x[:, n_shot:].reshape(-1, *episode_x.shape[-3:]).to(dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(n_class), n_eval)).to(dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        # set to training mode
        learner_w_grad.train()
        learner_wo_grad.train()
        # train learner_w_grad using the metalearner
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, epoch, batch_size)

        # Train meta-learner with validation loss
        # Initialize updated weights for network evaluation
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        # Prediction of learner on test_input
        output = learner_wo_grad(test_input)
        # Cross entropy loss 
        loss = learner_wo_grad.criterion(output, test_target)
        # Accuracy function from utils. To get accuracy of the network on the test data
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        # backpropagate loss through the network 
        loss.backward()
        # clipped to prevent exploding gradient
        nn.utils.clip_grad_norm_(metalearner.parameters(), grad_clip)
        #update the metalearner network
        optim.step()
        #print('eps is',eps)
        # record accuracy and loss 
        logger.batch_info(eps=eps, totaleps=episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        # Evaluation of meta-learner on validation set after every 1000 episodes

        if eps % val_freq == 0 and eps != 0:
            # save model for this episode
            save_ckpt(eps, metalearner, optim, logger.saven())
            acc = meta_test(eps, val_loader, learner_w_grad, learner_wo_grad, metalearner,logger,dev,n_shot, n_class, n_eval)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")
    plotgraph(logger.saven())
    logger.loginfo("Done")


if __name__ == '__main__':
    main()

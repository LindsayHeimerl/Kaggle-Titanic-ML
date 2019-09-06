import torch
import torch.nn as nn
import numpy as np
import argparse
import sys

device = torch.device('cpu')

# Parses all needed arguments, validating certain arguments as it does
def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("train", type=str)
    return parser.parse_args()
    
def main():
    torch.manual_seed(0)
    args = parse_args()
    data_train = torch.from_numpy(np.genfromtxt(args.train,delimiter=',').astype(np.float32))
    #creating a matrix of the training set without the first two colomns and then a vector
    # of just the "Survived" feature to use for training 
    train_feat = data_train[:,2:]
    train_targ = data_train[:,1].type(torch.LongTensor)
    
    #N = num features, D = feature dimensions
    N = list(train_feat.shape)[0]
    D = list(train_feat.shape)[1]
    
    model = dnn(D, args)
    train(model, train_feat, train_targ, N, D, args)

## Pytorch
class dnn(torch.nn.Module):
    def __init__(self, D, args):
        super(dnn, self).__init__()

        # Begin Embeddings
        self.embeddings = torch.nn.ModuleList()                 #Index, Category
        self.embeddings.append(nn.Embedding(3,3))               #0  PClass embedding
        self.embeddings.append(nn.Embedding(2,2))               #1  Gender embedding
        self.embeddings.append(nn.Embedding(7,4))              #4  Parch embedding
        self.embeddings.append(nn.Embedding(4,3))               #7  Gender embedding
        self.embeddings.append(nn.Embedding(3,3))               #8  Gender embedding

        #Begin Layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(nn.Linear(19, 256))
        self.layers.append(nn.Linear(256, 32))
        self.hidden = nn.functional.relu
        self.output = nn.Linear(32,2)

    def forward(self, x):
        embedding = self.embeddings[0](x[:,0].long())
        embedding = torch.cat((embedding, self.embeddings[1](x[:,1].long())), dim=1)
        embedding = torch.cat((embedding, self.embeddings[2](x[:,4].long())), dim=1)
        embedding = torch.cat((embedding, self.embeddings[3](x[:,7].long())), dim=1)
        embedding = torch.cat((embedding, self.embeddings[4](x[:,8].long())), dim=1)
        supreme = torch.cat((embedding,x[:,2:4].long()), dim=1)
        electric_boogaloo = torch.cat((supreme,x[:,5:7].long()), dim=1)
        x =electric_boogaloo

        for layer in self.layers:
            x = layer(x)
            x = self.hidden(x)
        
        x = self.output(x)

        return x

def train(model,train_x,train_y,N,D,args):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optim = torch.optim.Adam(model.parameters(),lr=0.1)

    epochs = 50
    
    for epoch in range(1,epochs+1):
        pred = model(train_x)
        loss = criterion(pred,train_y)

        optim.zero_grad() # reset the gradient values
        loss.backward()       # compute the gradient values
        optim.step()      # apply gradients

        _,train_y_pred_i = torch.max(pred,1)

        train_acc = torch.sum(torch.eq(train_y_pred_i, train_y), dtype=torch.float32)/N

        print("%03.d: train accuracy %.3f" % (epoch, train_acc))

'''
## Pytorch
class dnn(torch.nn.Module):
    def __init__(self, D, args):
        super(dnn_with_embeddings, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        self.embeddings.append(nn.Embedding(12568, 64)) #Location Embedding (geo-level 3)

        #Begin Category Two Embedding
        self.embeddings.append(nn.Embedding(3,3))       #Floor Count Embedding; PyTorch shits the bed on this one
        self.embeddings.append(nn.Embedding(3, 9))      #Roof Embedding
        self.embeddings.append(nn.Embedding(5, 15))     #Ground Floor Embedding
        self.embeddings.append(nn.Embedding(4, 12))     #Other Floor Embedding
        self.embeddings.append(nn.Embedding(10, 30))   #Building Plan Embedding

        self.layers = torch.nn.ModuleList()

        
        
        
        self.hidden = nn.functional.relu
        self.output = nn.Linear(64,3)

    # We can define our forward function in terms of torch operations and auto-grad will
    # still be able to handle it. One thing to try here is using the sparse=True option
    # on embeddings and SparseAdam optimizer (or whatever it is called)
    def forward(self, x):
        dense = True
        embedding = self.embeddings[0](x[:,0])
        #embedding = torch.cat((embedding, x[:,1].type(torch.FloatTensor)), dim=1) Don't even bother with floor count for now
        embedding = torch.cat((embedding, self.embeddings[2](x[:,2])), dim=1)
        embedding = torch.cat((embedding, self.embeddings[3](x[:,3])), dim=1)
        embedding = torch.cat((embedding, self.embeddings[4](x[:,4])), dim=1)
        embedding = torch.cat((embedding, self.embeddings[5](x[:,5])), dim=1)

        x = embedding
        #print(x.shape)

        for layer in self.layers:
            x = layer(x)
            x = self.hidden(x)
            
            if dense:
                x = self.hidden(x)
                dense = False
            else:
                dense = True
            

        x = self.output(x)

        return x

# Nothing different needs to happen here
def train(model,train_x,train_y,dev_x,dev_y,N,D,args):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr) #Can be used with cpu and sparse embeddings
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt == "sadam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr) #Can be used with CUDA or cpu on sparse embeddings only
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    devN = dev_x.shape[0]

    for epoch in range(args.epochs):
        # shuffle data once per epoch
        idx = np.random.permutation(N)
        train_x = train_x[idx]
        train_y = train_y[idx]

        for update in range(int(np.floor(N/args.mb))):
            mb_x = train_x[(update*args.mb):((update+1)*args.mb)].to(device)
            mb_y = train_y[(update*args.mb):((update+1)*args.mb)].to(device)

            mb_y_pred = model(mb_x) # evaluate model forward function

            loss      = criterion(mb_y_pred, mb_y) # compute loss
            optimizer.zero_grad() # reset the gradient values
            loss.backward()       # compute the gradient values
            optimizer.step()      # apply gradients

            if (update % args.report_freq) == 0:
                # eval on dev once per epoch
                dev_y_pred     = model(dev_x.to(device))
                _,dev_y_pred_i = torch.max(dev_y_pred,1)
                train_y_pred = model(train_x.to(device))
                _,train_y_pred_i = torch.max(train_y_pred,1)

                dev_acc = torch.sum(torch.eq(dev_y_pred_i, dev_y.to(device)),dtype=torch.float32)/devN
                train_acc = torch.sum(torch.eq(train_y_pred_i, train_y.to(device)),dtype=torch.float32)/N

                print("%03.d.%04d: dev %.3f train %.3f" % (epoch, update, dev_acc, train_acc))
'''

if __name__ == "__main__":
    main()
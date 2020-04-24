import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from distutils.util import strtobool
from torch.nn.parameter import Parameter
import math

def flip(x, dim):

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]

    return x.view(xsize)


def act_fun(act_type):

    if act_type=="relu":
       return nn.ReLU()

    if act_type=="tanh":
       return nn.Tanh()

    if act_type=="sigmoid":
       return nn.Sigmoid()

    if act_type=="leaky_relu":
       return nn.LeakyReLU(0.2)

    if act_type=="elu":
       return nn.ELU()

    if act_type=="softmax":
       return nn.LogSoftmax(dim=1)

    if act_type=="linear":
       return nn.LeakyReLU(1) # initializzed like this, but not used in forward!





class MLP(nn.Module):
    def __init__(self, options,inp_dim):
        super(MLP, self).__init__()

        self.input_dim=inp_dim
        self.dnn_lay=list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop=list(map(float, options['dnn_drop'].split(',')))
        self.dnn_use_batchnorm=list(map(strtobool, options['dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm=list(map(strtobool, options['dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp=strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp=strtobool(options['dnn_use_batchnorm_inp'])
        self.dnn_act=options['dnn_act'].split(',')


        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])


        # input layer normalization
        if self.dnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d(self.input_dim,momentum=0.05)


        self.N_dnn_lay=len(self.dnn_lay)

        current_input=self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

             # dropout
             self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

             # activation
             self.act.append(act_fun(self.dnn_act[i]))


             add_bias=True

             # layer norm initialization
             self.ln.append(LayerNorm(self.dnn_lay[i]))
             self.bn.append(nn.BatchNorm1d(self.dnn_lay[i],momentum=0.05))

             if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                 add_bias=False


             # Linear operations
             self.wx.append(nn.Linear(current_input, self.dnn_lay[i],bias=add_bias))

             # weight initialization
             self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.dnn_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.dnn_lay[i])),np.sqrt(0.01/(current_input+self.dnn_lay[i]))))
             self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

             current_input=self.dnn_lay[i]

        self.out_dim=current_input

    def forward(self, x):

      # Applying Layer/Batch Norm
      if bool(self.dnn_use_laynorm_inp):
        x=self.ln0((x))

      if bool(self.dnn_use_batchnorm_inp):

        x=self.bn0((x))

      for i in range(self.N_dnn_lay):

          if self.dnn_use_laynorm[i] and not(self.dnn_use_batchnorm[i]):
           x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

          if self.dnn_use_batchnorm[i] and not(self.dnn_use_laynorm[i]):
           x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

          if self.dnn_use_batchnorm[i]==True and self.dnn_use_laynorm[i]==True:
           x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

          if self.dnn_use_batchnorm[i]==False and self.dnn_use_laynorm[i]==False:
           x = self.drop[i](self.act[i](self.wx[i](x)))


      return x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class LSTM(nn.Module):

    def __init__(self, options,inp_dim):
        super(LSTM, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.lstm_lay=list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop=list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm=list(map(strtobool, options['lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm=list(map(strtobool, options['lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp=strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp=strtobool(options['lstm_use_batchnorm_inp'])
        self.lstm_act=options['lstm_act'].split(',')
        self.lstm_orthinit=strtobool(options['lstm_orthinit'])

        self.bidir=strtobool(options['lstm_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.to_do=options['to_do']

        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True


        # List initialization
        self.wfx  = nn.ModuleList([]) # Forget
        self.ufh  = nn.ModuleList([]) # Forget

        self.wix  = nn.ModuleList([]) # Input
        self.uih  = nn.ModuleList([]) # Input

        self.wox  = nn.ModuleList([]) # Output
        self.uoh  = nn.ModuleList([]) # Output

        self.wcx  = nn.ModuleList([]) # Cell state
        self.uch = nn.ModuleList([])  # Cell state

        self.ln  = nn.ModuleList([]) # Layer Norm
        self.bn_wfx  = nn.ModuleList([]) # Batch Norm
        self.bn_wix  = nn.ModuleList([]) # Batch Norm
        self.bn_wox  = nn.ModuleList([]) # Batch Norm
        self.bn_wcx = nn.ModuleList([]) # Batch Norm

        self.act  = nn.ModuleList([]) # Activations


        # Input layer normalization
        if self.lstm_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)

        # Input batch normalization
        if self.lstm_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_lstm_lay=len(self.lstm_lay)

        current_input=self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_lstm_lay):

             # Activations
             self.act.append(act_fun(self.lstm_act[i]))

             add_bias=True


             if self.lstm_use_laynorm[i] or self.lstm_use_batchnorm[i]:
                 add_bias=False


             # Feed-forward connections
             self.wfx.append(nn.Linear(current_input, self.lstm_lay[i],bias=add_bias))
             self.wix.append(nn.Linear(current_input, self.lstm_lay[i],bias=add_bias))
             self.wox.append(nn.Linear(current_input, self.lstm_lay[i],bias=add_bias))
             self.wcx.append(nn.Linear(current_input, self.lstm_lay[i],bias=add_bias))

             # Recurrent connections
             self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],bias=False))
             self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],bias=False))
             self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],bias=False))
             self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],bias=False))

             if self.lstm_orthinit:
                nn.init.orthogonal_(self.ufh[i].weight)
                nn.init.orthogonal_(self.uih[i].weight)
                nn.init.orthogonal_(self.uoh[i].weight)
                nn.init.orthogonal_(self.uch[i].weight)


             # batch norm initialization
             self.bn_wfx.append(nn.BatchNorm1d(self.lstm_lay[i],momentum=0.05))
             self.bn_wix.append(nn.BatchNorm1d(self.lstm_lay[i],momentum=0.05))
             self.bn_wox.append(nn.BatchNorm1d(self.lstm_lay[i],momentum=0.05))
             self.bn_wcx.append(nn.BatchNorm1d(self.lstm_lay[i],momentum=0.05))

             self.ln.append(LayerNorm(self.lstm_lay[i]))

             if self.bidir:
                 current_input=2*self.lstm_lay[i]
             else:
                 current_input=self.lstm_lay[i]

        self.out_dim=self.lstm_lay[i]+self.bidir*self.lstm_lay[i]



    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.lstm_use_laynorm_inp):
            x=self.ln0((x))

        if bool(self.lstm_use_batchnorm_inp):
            x_bn=self.bn0(x.view(x.shape[0]*x.shape[1],x.shape[2]))
            x=x_bn.view(x.shape[0],x.shape[1],x.shape[2])


        for i in range(self.N_lstm_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.lstm_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.lstm_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.lstm_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.lstm_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wfx_out=self.wfx[i](x)
            wix_out=self.wix[i](x)
            wox_out=self.wox[i](x)
            wcx_out=self.wcx[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.lstm_use_batchnorm[i]:

                wfx_out_bn=self.bn_wfx[i](wfx_out.view(wfx_out.shape[0]*wfx_out.shape[1],wfx_out.shape[2]))
                wfx_out=wfx_out_bn.view(wfx_out.shape[0],wfx_out.shape[1],wfx_out.shape[2])

                wix_out_bn=self.bn_wix[i](wix_out.view(wix_out.shape[0]*wix_out.shape[1],wix_out.shape[2]))
                wix_out=wix_out_bn.view(wix_out.shape[0],wix_out.shape[1],wix_out.shape[2])

                wox_out_bn=self.bn_wox[i](wox_out.view(wox_out.shape[0]*wox_out.shape[1],wox_out.shape[2]))
                wox_out=wox_out_bn.view(wox_out.shape[0],wox_out.shape[1],wox_out.shape[2])

                wcx_out_bn=self.bn_wcx[i](wcx_out.view(wcx_out.shape[0]*wcx_out.shape[1],wcx_out.shape[2]))
                wcx_out=wcx_out_bn.view(wcx_out.shape[0],wcx_out.shape[1],wcx_out.shape[2])


            # Processing time steps
            hiddens = []
            ct=h_init
            ht=h_init

            for k in range(x.shape[0]):

                # LSTM equations
                ft=torch.sigmoid(wfx_out[k]+self.ufh[i](ht))
                it=torch.sigmoid(wix_out[k]+self.uih[i](ht))
                ot=torch.sigmoid(wox_out[k]+self.uoh[i](ht))
                ct=it*self.act[i](wcx_out[k]+self.uch[i](ht))*drop_mask+ft*ct
                ht=ot*self.act[i](ct)

                if self.lstm_use_laynorm[i]:
                    ht=self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h=torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f=h[:,0:int(x.shape[1]/2)]
                h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
                h=torch.cat([h_f,h_b],2)

            # Setup x for the next hidden layer
            x=h


        return x

class GRU(nn.Module):

    def __init__(self, options,inp_dim):
        super(GRU, self).__init__()

        # Reading parameters
        self.input_dim=inp_dim
        self.gru_lay=list(map(int, options['gru_lay'].split(',')))
        self.gru_drop=list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm=list(map(strtobool, options['gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm=list(map(strtobool, options['gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp=strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp=strtobool(options['gru_use_batchnorm_inp'])
        self.gru_orthinit=strtobool(options['gru_orthinit'])
        self.gru_act=options['gru_act'].split(',')
        self.bidir=strtobool(options['gru_bidir'])
        self.use_cuda=strtobool(options['use_cuda'])
        self.to_do=options['to_do']

        if self.to_do=='train':
            self.test_flag=False
        else:
            self.test_flag=True


        # List initialization
        self.wh  = nn.ModuleList([])
        self.uh  = nn.ModuleList([])

        self.wz  = nn.ModuleList([]) # Update Gate
        self.uz  = nn.ModuleList([]) # Update Gate

        self.wr  = nn.ModuleList([]) # Reset Gate
        self.ur  = nn.ModuleList([]) # Reset Gate


        self.ln  = nn.ModuleList([]) # Layer Norm
        self.bn_wh  = nn.ModuleList([]) # Batch Norm
        self.bn_wz  = nn.ModuleList([]) # Batch Norm
        self.bn_wr  = nn.ModuleList([]) # Batch Norm


        self.act  = nn.ModuleList([]) # Activations


        # Input layer normalization
        if self.gru_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)

        # Input batch normalization
        if self.gru_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d(self.input_dim,momentum=0.05)

        self.N_gru_lay=len(self.gru_lay)

        current_input=self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_gru_lay):

             # Activations
             self.act.append(act_fun(self.gru_act[i]))

             add_bias=True


             if self.gru_use_laynorm[i] or self.gru_use_batchnorm[i]:
                 add_bias=False


             # Feed-forward connections
             self.wh.append(nn.Linear(current_input, self.gru_lay[i],bias=add_bias))
             self.wz.append(nn.Linear(current_input, self.gru_lay[i],bias=add_bias))
             self.wr.append(nn.Linear(current_input, self.gru_lay[i],bias=add_bias))


             # Recurrent connections
             self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i],bias=False))
             self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i],bias=False))
             self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i],bias=False))

             if self.gru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
                nn.init.orthogonal_(self.ur[i].weight)


             # batch norm initialization
             self.bn_wh.append(nn.BatchNorm1d(self.gru_lay[i],momentum=0.05))
             self.bn_wz.append(nn.BatchNorm1d(self.gru_lay[i],momentum=0.05))
             self.bn_wr.append(nn.BatchNorm1d(self.gru_lay[i],momentum=0.05))


             self.ln.append(LayerNorm(self.gru_lay[i]))

             if self.bidir:
                 current_input=2*self.gru_lay[i]
             else:
                 current_input=self.gru_lay[i]

        self.out_dim=self.gru_lay[i]+self.bidir*self.gru_lay[i]



    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.gru_use_laynorm_inp):
            x=self.ln0((x))

        if bool(self.gru_use_batchnorm_inp):
            x_bn=self.bn0(x.view(x.shape[0]*x.shape[1],x.shape[2]))
            x=x_bn.view(x.shape[0],x.shape[1],x.shape[2])


        for i in range(self.N_gru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2*x.shape[1], self.gru_lay[i])
                x=torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.gru_lay[i])


            # Drop mask initilization (same mask for all time steps)
            if self.test_flag==False:
                drop_mask=torch.bernoulli(torch.Tensor(h_init.shape[0],h_init.shape[1]).fill_(1-self.gru_drop[i]))
            else:
                drop_mask=torch.FloatTensor([1-self.gru_drop[i]])

            if self.use_cuda:
               h_init=h_init.cuda()
               drop_mask=drop_mask.cuda()


            # Feed-forward affine transformations (all steps in parallel)
            wh_out=self.wh[i](x)
            wz_out=self.wz[i](x)
            wr_out=self.wr[i](x)


            # Apply batch norm if needed (all steos in parallel)
            if self.gru_use_batchnorm[i]:

                wh_out_bn=self.bn_wh[i](wh_out.view(wh_out.shape[0]*wh_out.shape[1],wh_out.shape[2]))
                wh_out=wh_out_bn.view(wh_out.shape[0],wh_out.shape[1],wh_out.shape[2])

                wz_out_bn=self.bn_wz[i](wz_out.view(wz_out.shape[0]*wz_out.shape[1],wz_out.shape[2]))
                wz_out=wz_out_bn.view(wz_out.shape[0],wz_out.shape[1],wz_out.shape[2])

                wr_out_bn=self.bn_wr[i](wr_out.view(wr_out.shape[0]*wr_out.shape[1],wr_out.shape[2]))
                wr_out=wr_out_bn.view(wr_out.shape[0],wr_out.shape[1],wr_out.shape[2])


            # Processing time steps
            hiddens = []
            ht=h_init

            for k in range(x.shape[0]):

                # gru equation
                zt=torch.sigmoid(wz_out[k]+self.uz[i](ht))
                rt=torch.sigmoid(wr_out[k]+self.ur[i](ht))
                at=wh_out[k]+self.uh[i](rt*ht)
                hcand=self.act[i](at)*drop_mask
                ht=(zt*ht+(1-zt)*hcand)


                if self.gru_use_laynorm[i]:
                    ht=self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h=torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f=h[:,0:int(x.shape[1]/2)]
                h_b=flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0)
                h=torch.cat([h_f,h_b],2)

            # Setup x for the next hidden layer
            x=h


        return x







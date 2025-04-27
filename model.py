import torch
from torch import nn
import math
import numpy as np

class Input_Embeddings(nn.Module):

    def __init__(self,vocab_size:int,d_model:int):
        super().__init__() # pour l'heritage avec la classe au dessus
        self.vocab_size= vocab_size
        self.d_model= d_model
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class Positional_encoder(nn.Module):
    def __init__(self,d_model:int,seq_length:int,dropout:float)-> None:
        super().__init__()
        self.d_model=d_model
        self.seq_length=seq_length
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_length,d_model)
        position = torch.arange(0,seq_length,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)# remplit toutes le colonnes paires 
        pe[:,1::2]=torch.cos(position*div_term)# remplit colonne impaire commence par 1 et par de 2 en 2

        pe=pe.unsqueeze(0)
        self=self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class Layer_Normalization(nn.Module):
    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) #Multiplicatif
        self.bias=nn.Parameter(torch.zeros(1)) #biais additif
    
    def forward(self,x):
        mean=x.mean(dim =-1,keepdim=True)
        sd=x.std(dim=-1,keepdim=True)
        return (x-mean)*self.alpha/(sd+self.eps)+ self.bias
    


class Feed_forward(nn.Module):
    def __init__(self,d_model:int,dff:int,dropout:float)->None:
        super().__init__()
        self.linear_1=nn.Linear(d_model,dff) #W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(dff,d_model)#W2 and B2


    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, "d_model is not divisible by h"
        self.d_k=d_model//h
        self.W_Q=nn.Linear(d_model,d_model) # Dim= D_model*D_model
        self.W_K=nn.Linear(d_model,d_model) # Dim= D_model*D_model
        self.W_V=nn.Linear(d_model,d_model) # Dim= D_model*D_model

        self.W_O=nn.Linear(d_model,d_model) # Dim= D_model*D_model
        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9) # permet de transformer tous les mask en -inf pour apres le softmax (padding words)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores


    def forward(self,q,k,v,mask):
        query=self.W_Q(q) #(Batch, seq_len,d_model)->(Batch,seq_len,d_model) same 
        key=self.W_K(k) #//
        value=self.W_V(v)#//

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
        return self.W_O(x)


class ResidualConnection(nn.Module):
    def __init__(self,dropout:float)->None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=Layer_Normalization()

    def forward(self,x,sublayer):
        return x+sublayer.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:Feed_forward,dropout:float)-> None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward_block)
        return x
    


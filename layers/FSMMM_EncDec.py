import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.Global_manba import global_mamba
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from mosa import MoSA 
from layers.FSMattention import FSMattention,PositionLinearAttention
class FsmMambaEncoder(nn.Module):
    def __init__(self, configs):
        super(FsmMambaEncoder, self).__init__()

        self.layers = nn.ModuleList([FsmMambaBlock(configs) for _ in range(configs.e_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x


class FsmMambaBlock(nn.Module):
    def __init__(self, configs):
        super(FsmMambaBlock, self).__init__()
        #Mamba canshu
        self.even = Mamba(
            int(configs.d_model/2),  # Model dimension d_model
            configs.d_state,  # SSM state expansion factor
            configs.d_conv,  # Local convolution width
            configs.expand,  # Block expansion factor)
        )
        self.odd = Mamba(
            int(configs.d_model/2),  # Model dimension d_model
            configs.d_state,  # SSM state expansion factor
            configs.d_conv,  # Local convolution width
            configs.expand,  # Block expansion factor)
        )
        self.mean_fusion = Mamba(
            int(configs.d_model/2),  # Model dimension d_model
            configs.d_state,  # SSM state expansion factor
            configs.d_conv,  # Local convolution width
            configs.expand,  # Block expansion factor)
        )
        self.mamba_output=nn.Linear(configs.d_model,configs.d_model)
        self.channel_att_norm=nn.BatchNorm1d(configs.enc_in)
        self.LayerNorm=nn.LayerNorm(configs.d_model)
        self.ff = nn.Sequential(
                                nn.Linear(configs.d_model, int(configs.d_model*2)),
                                nn.GELU(),
                                nn.Dropout(configs.dropout),
                                nn.Linear(int(configs.d_model*2), configs.d_model),
                                )
        self.position_linear_attention=PositionLinearAttention(configs)
        self.channel_att_norm1=nn.BatchNorm1d(configs.enc_in)
        self.LayerNorm1=nn.LayerNorm(configs.d_model)
        self.ff1 = nn.Sequential(
                                nn.Linear(configs.d_model, int(configs.d_model*2)),
                                nn.GELU(),
                                nn.Dropout(configs.dropout),
                                nn.Linear(int(configs.d_model*2), configs.d_model),
                                )
    def deception(self,enc_out):
        even = enc_out[..., ::2]  
        odd  = enc_out[..., 1::2] 
        mean_fusion = (even + odd) / 2
        return even,odd,mean_fusion
    def forward(self, x):
      even,odd,mean_fusion=self.deception(x)
      even = self.even(even)
      odd = self.odd(odd)
      mean_fusion = self.mean_fusion(mean_fusion)


      enc=mean_fusion-(even+odd)/2
      odd=odd+enc#torch.Size([32, 7, 256])
      even=even+enc#torch.Size([32, 7, 256])
      manbaout = torch.cat([even, odd], dim=-1)
      res_2=self.channel_att_norm(manbaout+x)
      mamba_output=self.LayerNorm(self.ff(res_2)+res_2)#torch.Size([32, 7, 512])
      x1=mamba_output

      
      mamba_output=self.position_linear_attention(mamba_output)
      globla_2=self.channel_att_norm1(x1+mamba_output)
      globla_output=self.LayerNorm1(self.ff1(globla_2)+globla_2)#torch.Size([32, 7, 512])
      x=globla_output


      return x


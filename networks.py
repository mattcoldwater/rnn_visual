import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTM_Simple(nn.Module):
    def __init__(self, num_classes=120):
      super(LSTM_Simple, self).__init__()
      # self.layer1 = nn.LSTMCell(input_size=150, hidden_size=10) # input_size: feature
      self.lstm1 = nn.LSTM(input_size=150, hidden_size=100, num_layers=3, batch_first=True)
      # self.lstm2 = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
      self.layer2 = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, x_pack, hidden):
      # print(x.shape) # [batch_size, seq_len=129, features=150]

      # x_pack = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
      # print(x_pack.data.shape) # (seq_len=273, features=150)


      out, hidden = self.lstm1(x_pack, hidden)
      # print(out.data.shape) # (seq_len=273, num_hiddens=10)
      # print(h1.data.shape) # [num_layers*num_directions, batch_size,  num_hiddens]
      # print(c1.data.shape) # [num_layers*num_directions, batch_size,  num_hiddens]

      # out, (h1, c1) = self.lstm2(out)
      # print(out.data.shape) # (seq_len=273, num_hiddens=20)

      out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
      # print(out_pad.shape) # [batch_size, seq_len=129, num_hiddens=10]
      
      feat = torch.mean(out_pad, dim=1)
      # print(feat.shape) # (batch_size, num_hiddens=10)

      outs = self.layer2(feat)
      # print(outs.shape) # (batch_size, num_classes=120)

      return outs, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(3, batch_size, 100).zero_().cuda(),
                      weight.new(3, batch_size, 100).zero_().cuda())
        return hidden


class GRU_Simple(nn.Module):
    def __init__(self, num_classes=120, layers=1, hidden_size=2, input_size=150, packing=True):
        super(GRU_Simple, self).__init__()
        self.packing = packing
        self.layers = layers
        self.hidden_size = hidden_size
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True)
        self.layer2 = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

    def last_timestep(self, unpacked, x_len):
        # Index of the last output for each sequence.
        idx = (x_len - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        # print(idx)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, inputs, x_len, hidden):
        # For debug, set batch_size = 3, hidden_size=2
        if self.packing:
            x_pack = rnn_utils.pack_padded_sequence(inputs, list(x_len.data), batch_first=True, enforce_sorted=False)
            x_pack = x_pack.cuda()
            # print(x_pack.data.shape) # (seq_len=233, features=150)
            # print(hidden.shape) # layers, batch_size, hidden_size --> https://www.jianshu.com/p/95d5c461924c

            out, hidden = self.gru1(x_pack, hidden)
            # print(out.data.shape) # (seq_len=233, num_hiddens=100)
            # print(hidden.shape) # [num_layers, batch_size, num_hiddens]

            # assert torch.equal(x_pack.batch_sizes,out.batch_sizes)
            # assert torch.equal(x_pack.sorted_indices,out.sorted_indices)
            # assert torch.equal(x_pack.unsorted_indices,out.unsorted_indices)

            out_unpacked, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
            # print(out_unpacked.shape) # [batch_size=3, seq_len=99, num_hiddens=100]
            
            # feat = torch.mean(out_unpacked, dim=1)
            # print(x_len) [55, 99, 47]
            feat = self.last_timestep(out_unpacked, x_len.cuda()) # [54 54, 98 98, 46 46]
            # print(feat.shape) # (batch_size=3, num_hiddens=100)
        else:
            if inputs.shape[0] != hidden.shape[1]: # batch=3, seq_len=93, features=150
                hidden = hidden[:, :inputs.shape[0], :].contiguous()
            inputs = inputs.cuda()
            # print(x_len) 
            # print(torch.sum(inputs[0, (list(x_len.data)[0]-1):, :], dim=1))
            # print(torch.sum(inputs[1, (list(x_len.data)[1]-1):, :], dim=1))
            # print(torch.sum(inputs[2, (list(x_len.data)[2]-1):, :], dim=1))
            out, hidden = self.gru1(inputs, hidden)
            # print(out.shape, hidden.shape) # 3,93,100; 3,6,100
            feat = self.last_timestep(out, x_len.cuda())
            # print(feat.shape) # (batch_size=3, num_hiddens=100)

        outs = self.layer2(feat)
        # print(outs.shape) # (batch_size, num_classes=120)

        return outs, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.layers, batch_size, self.hidden_size).zero_().cuda()
        return hidden

#############################################

class GRU_Att_Layer(nn.Module):
    def __init__(self, input_size, hidden_size, atten=False, batch_first=True):
        super(GRU_Att_Layer, self).__init__()
        """
        x = batch * seq_len * input_size
        hid = batch * hidden_size
        cell input at T = batch * input_size
        cell output at T = batch * hidden_size
        """
        if not batch_first:
            return 'Batch First, Please.'
        self.atten = atten

        self.grucell = nn.GRUCell(input_size, hidden_size)
        if self.atten:
            self.att_layer = nn.Sequential(
                nn.Linear(input_size + hidden_size, input_size, bias=True),
                nn.Sigmoid()
            )

    def forward(self, inputs, hid, visual=False):
        seq_len = inputs.shape[1]
        outs = [0,] * seq_len
        if visual:
            attentions = [0,] * seq_len

        for seq in range(seq_len):
            x = inputs[:,seq,:]

            if self.atten:
                x_h = torch.cat((x, hid), dim=1)
                # print(x_h.shape) # 4, 250
                a = self.att_layer(x_h)
                # print(x.shape, a.shape) # 4, 150;  batch(4), input_size(150)
                x = x * a
                # print(x.shape) # 4, 150

                if visual:
                    attentions[seq] = a

            hid = self.grucell(x, hid)
            outs[seq] = hid
        
        outs = torch.stack(outs, dim=1) # batch * seq_len * hidden_size

        if visual:
            attentions = torch.stack(attentions, dim=1) # batch * seq_len * hidden_size
            return outs, attentions

        return outs


class GRU_Att_Layers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, atten=[False, False, False, False], batch_first=True, dropout=0):
        super(GRU_Att_Layers, self).__init__()
        """
        inputs = batch * seq_len * features
        hidden = layers * batch * hidden_size
        out_x = batch * seq_len * features
        """
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        layers = [GRU_Att_Layer(input_size, hidden_size, atten[0], batch_first),]
        for i in range(1, self.num_layers):
            layers.append(GRU_Att_Layer(hidden_size, hidden_size, atten[i], batch_first))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, hidden, batch_idx, visual=False):
        x = inputs
        next_hidden = [0,] * self.num_layers
        for i in range(self.num_layers):
            hid = hidden[i, :, :]
            if visual and i == 0:
                x, attentions = self.layers[i](x, hid, visual=True)
            else:
                x = self.layers[i](x, hid)
            # get last timestamp of hidden state
            next_hidden[i] = x.gather(1, batch_idx).squeeze() 
            # print(next_hidden[i].shape) # batch(4) * hidden_size(100)
            if i + 1 < self.num_layers:
                x = self.dropout(x)

        out = next_hidden[-1]
        next_hidden = torch.stack(next_hidden, dim=0) 
        # print(out.shape, next_hidden.shape) # batch(4) * hidden_size(100);  layers(3) * batch(4) * hidden_size(100)

        if visual:
            return out, next_hidden, attentions
        return out, next_hidden


class GRU_Att(nn.Module):
    def __init__(self, num_classes=120, layers=1, hidden_size=2, input_size=150, atten=[False, False, False, False], batch_first=True, dropout=0):
        super(GRU_Att, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.atten = atten
        self.gru1 = GRU_Att_Layers(input_size=input_size, hidden_size=hidden_size, num_layers=layers, atten=atten, batch_first=batch_first, dropout=dropout)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def last_timestep(self, unpacked, x_len):
        # Index of the last output for each sequence.
        # unpacked:  batch=3, seq_len=93, features=150
        idx = (x_len - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        # print(idx)
        return unpacked.gather(1, idx).squeeze()
    
    def forward(self, inputs, x_len, hidden, visual=False):
        # For debug, set batch_size = 4, hidden_size=2
        if inputs.shape[0] != hidden.shape[1]:
            hidden = hidden[:, :inputs.shape[0], :].contiguous()
        # print(inputs.shape) # batch(4), seq_len(134), features(150)
        
        batch_idx = (x_len.cuda() - 1).view(-1, 1).expand(inputs.shape[0], self.hidden_size).unsqueeze(1)

        if visual:
            feat, hidden, attentions = self.gru1(inputs.cuda(), hidden, batch_idx, visual=True)
        else:
            feat, hidden = self.gru1(inputs.cuda(), hidden, batch_idx)
        # print(feat.shape, hidden.shape) # batch_size(4), num_hiddens(100); layers(3), batch_size(4), num_hiddens(100); 

        outs = self.layer2(feat)
        # print(outs.shape) # batch_size(4), num_classes(120)

        if visual:
            return outs, hidden, attentions

        return outs, hidden

    def init_hidden(self, batch_size, best_hidden=None):
        if best_hidden != None:
            return best_hidden.cuda()
            
        weight = next(self.parameters()).data
        hidden = weight.new(self.layers, batch_size, self.hidden_size).zero_().cuda()
        return hidden


##########################################

class GRUModel_Template(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel_Template, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    def forward(self, x):
        # Initialize hidden state with zeros
        #print(x.shape,"x.shape")100, 28, 28
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())

        outs = []
        hn = h0[0,:,:]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out
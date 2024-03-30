import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb


class BahdanauAttention(nn.Module):
    def __init__(self, decoder_dim, encoder_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(decoder_dim, decoder_dim)
        self.W2 = nn.Linear(encoder_dim, decoder_dim)
        self.Va = nn.Linear(decoder_dim, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.W1(query) + self.W2(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttentionMultiHead(nn.Module):

    def __init__(self, input_size, out_size, nr_heads):
        super(AttentionMultiHead, self).__init__()
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttention(input_size, out_size)
                          for idx_head in range(nr_heads)])
        self.linear_out = nn.Linear(nr_heads*out_size, out_size)
        return

    def forward(self, input_vector):
        all_heads = []
        all_atts = []
        for head in self.heads:
            out, att = head(input_vector)
            all_heads.append(out)
            all_atts.append(att)
        z_out_concat = torch.cat(all_heads, dim=2)
        z_out_out = F.relu(self.linear_out(z_out_concat))
        return z_out_out, all_atts


class SelfAttention(nn.Module):

    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(
            in_features=input_size, out_features=out_size)
        self.key_linear = nn.Linear(
            in_features=input_size, out_features=out_size)
        self.value_linear = nn.Linear(
            in_features=input_size, out_features=out_size)
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))

        value_out = F.relu(self.value_linear(input_vector))
        att = torch.bmm(query_out, key_out.transpose(1, 2))

        out_q_k = torch.div(att, math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine, att

    def plot_attention(self, attentions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Save the figure to a wandb artifact
        wandb.log({"attention_matrix": wandb.Image(fig)})

        # Close the figure to prevent it from being displayed in the notebook
        plt.close(fig)

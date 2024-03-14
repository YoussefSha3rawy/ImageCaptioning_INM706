import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.W1(query) + self.W2(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class LuongAttn(nn.Module):
    def __init__(self, method, hidden_size):
        super(LuongAttn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2), None

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2), None

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies, att = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies, att = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies, att = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1), att


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
        self.softmax = nn.Softmax(dim=-1)
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

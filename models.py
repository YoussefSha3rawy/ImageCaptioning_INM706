import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
import math
from attention_models import BahdanauAttention, SelfAttention, AttentionMultiHead


class ImageEncoderRNN(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, backbone: str = None):
        super(ImageEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        x = x.unsqueeze(0)

        x = F.dropout(x, 0.2)

        return None, x


class ImageEncoderSelfAttentionRNN(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, nr_heads=1, backbone_name: str = None):
        super(ImageEncoderSelfAttentionRNN, self).__init__()
        self.hidden_size = hidden_size

        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-2])

        self.attention = AttentionMultiHead(
            input_size=2048, out_size=hidden_size, nr_heads=nr_heads)

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)
        # self.conv = nn.Conv2d(2048, 20, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        x = x.permute(0, 2, 3, 1)

        x = x.view(x.shape[0], -1, x.shape[-1])
        out, att = self.attention(x)

        out = F.dropout(out, p=0.3)

        return out, x


class DecoderRNN(nn.Module):
    SOS_token = 0
    EOS_token = 1

    def __init__(self, embedding_size, hidden_size, output_size, max_length=10, device=torch.device('cpu')):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(embedding_size, output_size)
        self.device = device
        self.max_length = max_length

    def forward(self, encoder_output, encoder_hidden, target_tensor=None):
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden.clone()
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(
                    1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # We return `None` for consistency in the training loop
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class AttnDecoderGRU(nn.Module):
    max_length = 10
    SOS_token = 0
    EOS_token = 1

    def __init__(self, hidden_size, output_size, embedding_size, max_length, dropout_p=0.1, device=torch.device('cpu')):
        super(AttnDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = BahdanauAttention(hidden_size)
        self.attention_function = self.forward_step_bahdanau
        self.gru = nn.GRU(embedding_size + hidden_size,
                          hidden_size, batch_first=True)
        self.device = device
        self.max_length = max_length

        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):

            decoder_output, decoder_hidden, attn_weights = self.attention_function(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(
                    1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        if attentions[0] is not None:
            attentions = torch.cat(attentions, dim=1)
        else:
            attentions = None

        return decoder_outputs, decoder_hidden, attentions

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from attention_models import BahdanauAttention, AttentionMultiHead, LuongAttn
from enum import Enum


class ImageEncoderFC(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, backbone: str = None):
        super(ImageEncoderFC, self).__init__()
        self.hidden_size = hidden_size

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)
        self.out_features = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = F.dropout(x, 0.2)

        return x


class ImageEncoderAttention(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, nr_heads=1, backbone_name: str = None):
        super(ImageEncoderAttention, self).__init__()
        self.hidden_size = hidden_size

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-2])

        # self.attention = AttentionMultiHead(
        #     input_size=2048, out_size=hidden_size, nr_heads=nr_heads)

        if hasattr(backbone, 'fc'):
            self.out_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            self.out_features = backbone.classifier.in_features

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)
        # self.conv = nn.Conv2d(2048, 20, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        return x


class AttentionTypes(Enum):
    NONE = 'None'
    BAHDANAU = 'Bahdanau'
    LUONG = 'Luong'


class DecoderWithAttention(nn.Module):
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2

    def __init__(self, embedding_size, hidden_size, encoder_dim, output_size, max_length=10, beam_size=1, attention_type=AttentionTypes.NONE, dropout_p=0.1, device=torch.device('cpu')):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = device
        self.attention_type = attention_type
        self.encoder_dim = encoder_dim

        print(attention_type)
        if attention_type == AttentionTypes.BAHDANAU.value:
            self.attention_module = BahdanauAttention(
                hidden_size, self.encoder_dim)
            self.attention_function = self.forward_step_bahdanau
            self.gru_input_size = embedding_size + self.encoder_dim
        elif attention_type == AttentionTypes.LUONG.value:
            self.attention = LuongAttn('general', hidden_size)
            self.attention_function = self.forward_step_luong
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.gru_input_size = embedding_size
        self.gru = nn.GRU(self.gru_input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.init_hidden = nn.Conv2d(self.encoder_dim, hidden_size, 7)

    def initiate_hidden(self, encoder_output):
        x = self.init_hidden(encoder_output)
        x = x.squeeze().unsqueeze(0)
        return x

    def forward(self, encoder_outputs, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full(
            (batch_size, 1), self.SOS_token, dtype=torch.long, device=self.device)
        decoder_hidden = self.initiate_hidden(encoder_outputs)
        encoder_outputs = encoder_outputs.permute(0, 2, 3, 1)
        encoder_outputs = encoder_outputs.view(
            batch_size, -1, encoder_outputs.size(-1))
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            if self.attention_type == AttentionTypes.NONE:
                decoder_output, decoder_hidden = self.forward_step_no_attention(
                    decoder_input, decoder_hidden
                )
            else:
                decoder_output, decoder_hidden, attn_weights = self.attention_function(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                attentions.append(attn_weights)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(
            attentions, dim=1) if self.attention_type == 'Bahdanau' else None

        return decoder_outputs, decoder_hidden, attentions

    def forward_step_no_attention(self, input, hidden):
        output = self.embedding(input)
        output = self.dropout(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention_module(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

    def forward_step_luong(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        attn_weights, _ = self.attention(output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)

        rnn_output = output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output.unsqueeze(1), hidden, None


class SelfAttnDecoderRNN(nn.Module):
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2

    def __init__(self, embedding_size, hidden_size, output_size, max_length=10, device=torch.device('cpu')):
        super(SelfAttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(embedding_size, output_size)
        self.device = device
        self.max_length = max_length

    def forward(self, encoder_output, encoder_hidden, target_tensor=None):
        batch_size = encoder_output.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_output.mean(dim=1)
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, encoder_output, decoder_hidden)
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

    def forward_step(self, input, encoder_output, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        gru_input = torch.cat((output, encoder_output), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output)
        return output, hidden

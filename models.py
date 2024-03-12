import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class ImageEncoderRNN(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, backbone: str = None):
        super(ImageEncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        x = x.unsqueeze(0)

        return x


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

    def forward(self, encoder_output, target_tensor=None):
        batch_size = encoder_output.size(1)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_output.clone()
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

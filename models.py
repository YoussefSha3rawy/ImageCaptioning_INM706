import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from attention_models import BahdanauAttention, AttentionMultiHead
from enum import Enum
from timm import create_model


class ImageEncoderFC(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, backbone: str = None):
        super(ImageEncoderFC, self).__init__()
        self.hidden_size = hidden_size

        self.cnn = resnet101(weights=ResNet101_Weights.DEFAULT)

        for param in self.cnn.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            for c in list(self.cnn.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_size)
        self.out_features = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = F.dropout(x, 0.2)

        return x

    def __str__(self) -> str:
        return self.__class__.__name__


class ImageEncoderAttention(nn.Module):
    def __init__(self, hidden_size: int, freeze_backbone=False, backbone_name: str = None, dropout_p=0.2, nr_heads=0):
        super(ImageEncoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.nr_heads = nr_heads

        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)

        self.cnn = nn.Sequential(*list(backbone.children())[:-2])

        self.dropout = nn.Dropout(dropout_p)

        if hasattr(backbone, 'fc'):
            self.out_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            self.out_features = backbone.classifier.in_features

        for param in self.cnn.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            for c in list(self.cnn.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

        if nr_heads > 0:
            self.self_attention = AttentionMultiHead(
                input_size=self.out_features, out_size=self.out_features, nr_heads=nr_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        if self.nr_heads > 0:
            x = x.permute(0, 2, 3, 1)
            batch, i, j, features = x.size()
            x = x.view(batch, -1, features)
            x, _ = self.self_attention(x)
            x = x.view(batch, i, j, features)
            x = x.permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x

    def __str__(self) -> str:
        return self.__class__.__name__ + (f'_{self.nr_heads}hAttn' if self.nr_heads > 0 else '')


class AttentionTypes(Enum):
    NONE = 'None'
    BAHDANAU = 'Bahdanau'


class DecoderWithAttention(nn.Module):
    SOS_token = 1
    EOS_token = 2

    def __init__(self, embedding_size, hidden_size, encoder_dim, output_size, max_length=10, attention_type=AttentionTypes.NONE, dropout_p=0.1, device=torch.device('cpu')):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.device = device
        self.attention_type = attention_type
        self.encoder_dim = encoder_dim

        assert attention_type in [
            member.value for member in AttentionTypes], 'Attention type not supported'

        if attention_type == AttentionTypes.BAHDANAU.value:
            self.attention_module = BahdanauAttention(
                hidden_size, self.encoder_dim)
            self.attention_function = self.forward_step_bahdanau
            self.gru_input_size = embedding_size + self.encoder_dim
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
            if self.attention_type == AttentionTypes.NONE.value:
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
        output = self.out(self.dropout(output))
        return output, hidden

    def forward_step_bahdanau(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention_module(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(self.dropout(output))
        return output, hidden, attn_weights

    def __str__(self) -> str:
        return f'{self.__class__.__name__}{"_" + self.attention_type if self.attention_type!= AttentionTypes.NONE.value else ""}'


class VanillaDecoderRNN(nn.Module):
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2

    def __init__(self, embedding_size, hidden_size, output_size, max_length=10, dropout_p=0.2, device=torch.device('cpu')):
        super(VanillaDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.max_length = max_length

    def forward(self, encoded_image, target_tensor=None):
        batch_size = encoded_image.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoded_image.unsqueeze(0)
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
        output = self.dropout(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(self.dropout(output))
        return output, hidden

    def __str__(self) -> str:
        return self.__class__.__name__


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, encoder_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(
            1, 500, hidden_size))  # Adjust 500 based on max length

        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt_emb = self.embedding(
            tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt_emb = self.dropout(tgt_emb)

        output = self.transformer_decoder(tgt_emb, memory)
        output = self.fc(output)
        return output


class ViTImageEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTImageEncoder, self).__init__()
        self.vit = create_model(model_name, pretrained=pretrained)
        self.vit.reset_classifier(0)  # Remove the classification head
        self.hidden_size = self.vit.embed_dim

    def forward(self, x):
        # [batch_size, num_patches, hidden_dim]
        features = self.vit.forward_features(x)
        return features

    def __str__(self) -> str:
        return self.__class__.__name__


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, max_length, dropout_p=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(
            1, int(max_length), int(hidden_size)))  # Adjust 500 based on max length

        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout_p, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tgt, memory):
        tgt_emb = self.embedding(
            tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt_emb = self.dropout(tgt_emb)

        output = self.transformer_decoder(tgt_emb, memory)
        output = self.fc(output)
        return output

    def __str__(self) -> str:
        return self.__class__.__name__


class ImageCaptioningModel(nn.Module):
    SOS_token = 1
    EOS_token = 2

    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, max_length, dropout_p=0.1, device=torch.device('cpu'), beam_width=3):
        super(ImageCaptioningModel, self).__init__()
        self.decoder = TransformerDecoder(
            vocab_size, hidden_size, num_heads, num_layers, max_length, dropout_p)
        self.max_length = max_length
        self.device = device
        self.beam_width = beam_width

    def forward(self, encoder_outputs, target_captions=None):
        # [batch_size, num_patches, hidden_dim]
        batch_size = encoder_outputs.size(0)

        if target_captions is not None:
            target_captions = target_captions.to(self.device)
            outputs = self.decoder(target_captions, encoder_outputs)
            return outputs, None, None
        else:
            # Beam search or greedy decoding
            generated_captions = torch.full(
                (batch_size, 1), self.SOS_token, dtype=torch.long, device=self.device)
            for _ in range(self.max_length):
                outputs = self.decoder(generated_captions, encoder_outputs)
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_captions = torch.cat(
                    (generated_captions, next_token), dim=1)
                if torch.all(next_token == self.EOS_token):
                    break
            return generated_captions, None, None

    def __str__(self) -> str:
        return self.__class__.__name__

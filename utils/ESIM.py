import torch
import torch.nn as nn
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(ESIM, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        # LSTM layers for input encoding
        self.lstm_input_encoder = nn.LSTM(input_size=hidden_size,
                                          hidden_size=hidden_size // 2,
                                          num_layers=1,
                                          batch_first=True,
                                          bidirectional=True)

        # Projection layer for input encoding
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 输入维度是 hidden_size
            nn.ReLU()
        )

        # LSTM layer for inference composition
        self.lstm_inference_composition = nn.LSTM(input_size=4 * hidden_size,
                                                  hidden_size=hidden_size // 2,
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True)

        # Prediction layer
        self.prediction = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input1, input2):
        # input1 and input2: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = input1.size()

        # Encode input sequences using LSTM
        input1_encoded, _ = self.lstm_input_encoder(input1)
        input2_encoded, _ = self.lstm_input_encoder(input2)

        # Project encoded sequences to a lower dimension
        input1_projected = self.projection(input1_encoded)
        input2_projected = self.projection(input2_encoded)

        # Compute attention weights
        attention_weights = torch.matmul(input1_projected, input2_projected.transpose(1, 2))
        attention_weights1 = F.softmax(attention_weights, dim=2)
        attention_weights2 = F.softmax(attention_weights.transpose(1, 2), dim=2)

        # Compute soft alignments
        input1_aligned = torch.matmul(attention_weights1, input2_projected)
        input2_aligned = torch.matmul(attention_weights2, input1_projected)

        # Compute enhanced representations
        input1_enhanced = torch.cat([input1_projected, input1_aligned,
                                     input1_projected - input1_aligned,
                                     input1_projected * input1_aligned], dim=2)
        input2_enhanced = torch.cat([input2_projected, input2_aligned,
                                     input2_projected - input2_aligned,
                                     input2_projected * input2_aligned], dim=2)

        # Inference composition
        input1_composed, _ = self.lstm_inference_composition(input1_enhanced)
        input2_composed, _ = self.lstm_inference_composition(input2_enhanced)

        # Pooling
        input1_pooled = torch.max(input1_composed, dim=1)[0]
        input2_pooled = torch.max(input2_composed, dim=1)[0]

        # Concatenate pooled representations
        combined_representation = torch.cat([input1_pooled, input2_pooled,
                                             input1_pooled - input2_pooled,
                                             input1_pooled * input2_pooled], dim=1)

        # Prediction
        logits = self.prediction(combined_representation)
        return logits
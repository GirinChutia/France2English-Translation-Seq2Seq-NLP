import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        """
        Initialize the EncoderRNN module.

        Args:
            input_size (int): Size of the input vocabulary.
            hidden_size (int): Hidden size of the RNN.
            dropout_p (float): Dropout probability.
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=hidden_size
        )

        self.gru = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        """
        Forward pass of the EncoderRNN.

        Args:
            input (Tensor): Input tensor to the encoder.

        Returns:
            output (Tensor): Output tensor.
            hidden (Tensor): Hidden state tensor.
        """
        embedded = self.dropout(self.embedding(input))  # Get input embedding
        output, hidden = self.gru(embedded)  # Pass input to GRU
        return output, hidden


class DecoderRNN(nn.Module):
    """
    DecoderRNN module for sequence-to-sequence translation.
    """

    def __init__(self, hidden_size, output_size, **kwargs):
        """
        Initialize the DecoderRNN module.

        Args:
            hidden_size (int): Hidden size of the RNN.
            output_size (int): Size of the output vocabulary.
            **kwargs: Additional keyword arguments.
        """
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.kwargs = kwargs

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """
        Forward pass of the DecoderRNN.

        Args:
            encoder_outputs (Tensor): Output tensor from the encoder.
            encoder_hidden (Tensor): Hidden state tensor from the encoder.
            target_tensor (Tensor, optional): Target tensor for teacher forcing.

        Returns:
            decoder_outputs (Tensor): Output tensor.
            decoder_hidden (Tensor): Hidden state tensor.
            None: Placeholder for consistency in the training loop.
        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.kwargs.get("device", "cpu")
        ).fill_(self.kwargs.get("SOS_token"))
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.kwargs.get("MAX_LENGTH")):
            # Forward pass for a single time step
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        )  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        """
        Forward pass for a single time step.

        Args:
            input (Tensor): Input tensor for a single time step.
            hidden (Tensor): Hidden state tensor.

        Returns:
            output (Tensor): Output tensor for a single time step.
            hidden (Tensor): Hidden state tensor.
        """
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    """
    Implements the Bahdanau attention mechanism.

    Args:
        hidden_size (int): The size of the hidden state of the RNN.
    """

    def __init__(self, hidden_size):
        """
        Initialize the BahdanauAttention module.

        Args:
            hidden_size (int): The size of the hidden state of the RNN.
        """
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # Linear transformation for query
        self.Ua = nn.Linear(hidden_size, hidden_size)  # Linear transformation for keys
        self.Va = nn.Linear(hidden_size, 1)  # Linear transformation for scores

    def forward(self, query, keys):
        """
        Forward pass of the Bahdanau attention mechanism.

        Args:
            query (Tensor): The query tensor of shape (batch_size, 1, hidden_size).
            keys (Tensor): The keys tensor of shape (batch_size, max_length, hidden_size).

        Returns:
            context (Tensor): The context tensor of shape (batch_size, 1, hidden_size).
            weights (Tensor): The attention weights tensor of shape (batch_size, 1, max_length).
        """
        # Compute scores
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)

        # Compute context tensor
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    """
    DecoderRNN module with attention.
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, **kwargs):
        """
        Initialize the AttnDecoderRNN module.

        Args:
            hidden_size (int): The size of the hidden state of the RNN.
            output_size (int): The size of the output vocabulary.
            dropout_p (float): The dropout probability.
            **kwargs: Additional keyword arguments.
        """
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.kwargs = kwargs

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """
        Forward pass of the AttnDecoderRNN.

        Args:
            encoder_outputs (Tensor): Output tensor from the encoder.
            encoder_hidden (Tensor): Hidden state tensor from the encoder.
            target_tensor (Tensor, optional): Target tensor for teacher forcing.

        Returns:
            decoder_outputs (Tensor): Output tensor.
            decoder_hidden (Tensor): Hidden state tensor.
            attentions (Tensor): Attention weights tensor.
        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.kwargs.get("device", "cpu")
        ).fill_(self.kwargs.get("SOS_token"))
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.kwargs.get("MAX_LENGTH")):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(
        self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ):
        """
        Forward pass for a single time step.

        Args:
            input (Tensor): Input tensor for a single time step.
            hidden (Tensor): Hidden state tensor.
            encoder_outputs (Tensor): Output tensor from the encoder.

        Returns:
            output (Tensor): Output tensor for a single time step.
            hidden (Tensor): Hidden state tensor.
            attentions (Tensor): Attention weights tensor.
        """
        embedded = self.dropout(self.embedding(input))

        # Compute attention weights
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)

        # Concatenate the input and context
        input_gru = torch.cat((embedded, context), dim=2)

        # Pass the input to the GRU
        output, hidden = self.gru(input_gru, hidden)

        # Pass the output to the output layer
        output = self.out(output)

        return output, hidden, attn_weights

import os
from typing import Literal, Optional
import numpy as np
import torch.nn as nn
import transformers
import torch
from torchviz import make_dot
import matplotlib.pyplot as plt

os.environ["GRAPHVIZ_DOT"] = "C:/Program Files/Graphviz/bin/dot.exe"

class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    
    def plot_architecture(self, example_inputs, file_name="model_architecture", format: Literal['png', 'pdf'] = 'png'):
        """
        Generates and saves a plot of the model's architecture (computational graph).
        
        :param example_inputs: A tuple of tensors that you would normally pass to forward(...).
        :param file_name: Base name of the file for saving the plot. By default is 'model_architecture'.
        :param format: The format in which to save the plot. Can be 'png' or 'pdf'. Default is 'png'.
        """
        self.eval()
        with torch.no_grad():
            output = self(*example_inputs)  # Forward pass with example inputs.
        
        # Create a dot object from the computational graph.
        # named_parameters() is turned into a dict for labeling.
        dot = make_dot(output, params=dict(self.named_parameters()))
        
        # Specify the format you want (e.g., 'png' or 'pdf')
        dot.format = format
        
        # Render the plot (file_name + .pdf, plus an additional .gv file if cleanup=False)
        dot.render(file_name, cleanup=True)
        
        print(f"Model architecture graph saved to {file_name}.{format}")
        
        # Optionally display inline if you want to see it in a notebook (for debugging).
        plt.imshow(plt.imread(f"{file_name}.{format}"))
        plt.axis('off')
        plt.show()

class GruBERT(AbstractModel):
    def __init__(self, embedding_size, gru_hidden_size, num_gru_layers, gru_dropout, model_name = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", text_enrichment = False, num_tags: Optional[int] = None):
        super(GruBERT, self).__init__()
        self.model_class = "grubert"
        self.num_tags = num_tags
        self.text_enrichment = text_enrichment
        
        # self.Bert = transformers.BertModel.from_pretrained(model_name)
        self.Bert = transformers.AutoModel.from_pretrained(model_name, token="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")

        self.gru = nn.GRU(
            input_size=embedding_size + num_tags if num_tags else embedding_size, 
            hidden_size = gru_hidden_size, 
            num_layers=num_gru_layers,
            dropout=gru_dropout,
            bidirectional=True,
            batch_first=True
            )

        fc_input_dim = gru_hidden_size * 2 if self.gru.bidirectional else gru_hidden_size
        
        if text_enrichment:
            fc_input_dim += 1

        self.fc = nn.Linear(fc_input_dim, 1)
        self.sigmoid = nn.Sigmoid() # TODO: try softmax with dual output

        # if text_enrichment:
        #     self.fc = nn.Linear(gru_hidden_size + 1, output_size)
        # else:
        #     self.fc = nn.Linear(gru_hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()  # Funzione di attivazione sigmoid

        for param in self.Bert.parameters():
            param.requires_grad = False
        # (32, 128, 1024)

    def forward(self, ids, mask, token_type_ids, probs=None, pos_tags=None):
        # adding a flag error that reminds me to implement pos tags here too
        # Check whether the two pos tags variables are compatible
        if (pos_tags is None) != (self.num_tags is None):
            pos_tags_is_none = "not " if pos_tags is None else ""
            raise ValueError(
                f"Model was initialized with num_tags = {self.num_tags}, "
                f"yet pos tags were {pos_tags_is_none}passed in the forward."
            )
        
        embedded = self.Bert(ids, 
                        attention_mask = mask, 
                        token_type_ids = token_type_ids, 
                        return_dict=True
                ).last_hidden_state
        
        # out, _ = self.gru(out) # da capire com'è fatto sto tensore
        # out = mean(out, dim=1) # (64, 256)

        if pos_tags is not None:
            # pos_tags = pos_tags.permute(1, 0, 2)
            embedded = torch.cat((embedded, pos_tags), dim=2) 

        # Pass through GRU
        _, hidden = self.gru(embedded)                              # Shape: (num_layers * num_directions, batch_size, gru_hidden_size)
        
        # Concatenate the final forward and backward hidden states if bidirectional
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)     # Shape: (batch_size, hidden_size * 2)
        else:
            hidden = hidden[-1]          

        if probs != None:
            hidden = torch.cat([hidden, probs.reshape((probs.shape[0], 1))], dim=1)  # da dare in pasto al lineare

        out = self.fc(hidden)  # Prendi l'output dell'ultimo timestep

        return self.sigmoid(out).squeeze(1) 
    
class Gru(AbstractModel):
    def __init__(self, embedding_matrix, scale_grad_by_freq, gru_hidden_size, num_gru_layers, gru_dropout, text_enrichment=False, num_tags: Optional[int] = None):
        super(Gru, self).__init__()
        self.model_class = "gru"

        self.num_tags = num_tags
        self.text_enrichment = text_enrichment

        # 0 - EMBEDDING LOOKUP LAYER
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=True,
            padding_idx=0, 
            scale_grad_by_freq=scale_grad_by_freq
            )
        
        # 1 - GRU MODULE
        self.gru = nn.GRU(
            input_size=embedding_matrix.shape[1] + num_tags if num_tags else embedding_matrix.shape[1], 
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            dropout=gru_dropout,
            bidirectional=True,
            batch_first=True
            )
        
        # 2 - LINEAR LAYER
        fc_input_dim = gru_hidden_size * 2 if self.gru.bidirectional else gru_hidden_size
        
        if text_enrichment:
            fc_input_dim += 1

        self.fc = nn.Linear(fc_input_dim, 1)
        self.sigm = nn.Sigmoid() # TODO: try softmax with dual output

    def forward(self, x, probs=None, pos_tags: Optional[np.ndarray] = None):
        # Check whether the two pos tags variables are compatible
        if (pos_tags is None) != (self.num_tags is None):
            pos_tags_is_none = "not " if pos_tags is None else ""
            raise ValueError(
                f"Model was initialized with num_tags = {self.num_tags}, "
                f"yet pos tags were {pos_tags_is_none}passed in the forward."
            )
        
        # x is expected to have shape (batch_size, sequence_length)
        
        # Pass input through the embedding layer
        embedded = self.embedding(x)                                # Shape: (batch_size, sequence_length, embedding_dim)
        # Optionally, permute to fit expected input shape for GRU
        # embedded = embedded.permute(1, 0, 2)                        # Shape: (sequence_length, batch_size, embedding_dim)
        
        if pos_tags is not None:
            # pos_tags = pos_tags.permute(1, 0, 2)
            embedded = torch.cat((embedded, pos_tags), dim=2)             # Shape: (sequence_length, batch_size, embedding_dim + num_tags)

        # Pass through GRU
        _, hidden = self.gru(embedded)                              # Shape: (num_layers * num_directions, batch_size, gru_hidden_size)
        
        # Concatenate the final forward and backward hidden states if bidirectional
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)     # Shape: (batch_size, hidden_size * 2)
        else:
            hidden = hidden[-1]                                     # Shape: (batch_size, hidden_size)
            

        if probs is not None:
            hidden = torch.cat([hidden, probs.reshape((probs.shape[0], 1))], dim=1)  # da dare in pasto al lineare

        # Pass through the fully connected layer and apply sigmoid activation
        output = self.fc(hidden)                                    # Shape: (batch_size, 1)
        output = self.sigm(output).squeeze(1)                        # Shape: (batch_size)
        return output

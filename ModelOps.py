import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from math import log
from typing import Literal, Union

from models import Gru, GruBERT

def load_weights(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def model_pred(model, data, device, model_class: Literal["gru", "grubert"]):
    assert model_class in ["gru", "grubert"], "model_class must be either 'gru' or 'grubert."
    # text enrichment
    probs = (
        data['probs'].to(device, dtype = torch.float) if model.text_enrichment and 'probs' in data.keys()
        else None
    )

    pos_tags = (
        data['pos_tags'].to(device, dtype=torch.long) if model.num_tags is not None and 'pos_tags' in data.keys()
        else None
    )

    # gru
    if model_class == "gru":
        inputs = data['inputs'].to(device, dtype=torch.long)
        try:
            return model(x = inputs, probs = probs, pos_tags = pos_tags)
        
        except Exception as e:
            print(f'ERROR: {e}')
            raise e
    
    # grubert
    ids = data['ids'].to(device, dtype=torch.long)
    mask = data['mask'].to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
    try:
        return model(ids, mask, token_type_ids, probs, pos_tags)
    except Exception as e:
        print(f'ERROR: {e}')
        raise e

        
def train_model(
        model,
        dataloader,
        val_loader, 
        device,
        loss_f,
        optimizer,
        scheduler,
        model_class: Literal["gru", "grubert"] = "grubert",
        # use_text_enrich: bool = False,
        # use_pos_tags: bool = False,
        verbose: bool = True,
    ):
    model.train() # set the model in training mode

    loss_history = []
    val_loss_history = []

    with tqdm(total=len(dataloader), desc="Batches", disable=False, leave=False) as pbar:
        for _, data in enumerate(dataloader, 0):

            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model_pred(
                model, data, device, model_class=model_class
            )
            
            loss = loss_f(outputs, targets)
            
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            pbar.update(1)

    if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()
        
    # evaluate validation loss

    model.eval()
    for _, data in enumerate(val_loader, 0):
        targets = data['targets'].to(device, dtype = torch.float)
        batch_pred = model_pred(
            model, data, device, model_class=model_class
        )
        loss = loss_f(batch_pred, targets)
        val_loss_history.append(loss.item())

    train_loss = sum(loss_history)/len(loss_history)
    validation_loss = sum(val_loss_history)/len(val_loss_history)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)

    if verbose:
        print(f"\nTRAIN LOSS: {train_loss} \nVALIDATION LOSS: {validation_loss}\n************************************")

    model.train()

    return {'train_loss':train_loss,
            'validation_loss': validation_loss}

def plot_predicted_true_distributions(true, predicted, params, columns = None):
    if columns is None:
        columns = np.arange(len(true[0]))
    print("Per class distribution of predicted and true labels")
    pred_test =  [(predicted[:, i]>=params[i]['thr']).int() for i in range(len(predicted[0]))]

    for _class in range(len(true[0])):
        print('*****',columns[_class],'*****')
        predicted_counter = {}
        true_counter = {}

        for i in np.array(pred_test[_class]):
            if i in predicted_counter:
                predicted_counter[i] = predicted_counter[i] + 1
            else:
                predicted_counter[i] = 1

        for i in np.array(true[:, _class]):
            if i in true_counter:
                true_counter[i] = true_counter[i] +1
            else:
                true_counter[i] = 1

        # print('predicted Counter: ', predicted_counter)
        # print('true Counter: ', true_counter)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].bar(predicted_counter.keys(), predicted_counter.values(), edgecolor='black')
        axs[0].set_title('predicted Counter')
        axs[0].set_xlabel('Class')
        axs[0].set_ylabel('Frequency')

        axs[1].bar(true_counter.keys(), true_counter.values(), edgecolor='black')
        axs[1].set_title('true Counter')
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Frequency')


        plt.tight_layout()
        plt.show()

def get_log_frequencies(train_set, labels):
    """
    Returns the negative log-frequency of the TPs of the given labels of the train set.

    ## Args:
        train_set (Pandas.DataFrame): the dataset.
        labels (list): the list of labels for which the frequencies are computed.
    
    ## Returns:
        class_frequencies (list): a list of negative log-frequencies in [0,1].

    ## Example usage:
        >>> train_data = pd.DataFrame({
            'label1': [1, 0, 1, 0, 1],
            'label2': [1, 0, 0, 0, 0],
            'label3': [0, 1, 1, 0, 0]
            })
        >>> labels_to_compute = ['label1', 'label3']
        >>> frequencies = get_log_frequencies(train_data, labels_to_compute)
        >>> print(frequencies)
        [0.4054651081081643, -0.4054651081081642]

    """
    TP = []
    TN = []
    pis = []
    class_frequencies = []
    for label in labels:
        TP.append(train_set[label].sum())
        TN.append(len(train_set)- TP[-1])
        pis.append(TP[-1]/(TP[-1] + TN[-1]))
        class_frequencies.append(-log((1-pis[-1])/(pis[-1])))
    return torch.tensor(class_frequencies)

def init_last_biases(model, specific_bias_values):
    """
    Initializes the biases of the last linear layer in the given model.

    ## Args:
        model (nn.Module): Your PyTorch model.
        specific_bias_values (list): List of specific bias values.
    
    ## Example usage:
        >>> model = nn.Sequential(
        >>>     nn.Linear(100, 50),  # Example layers
        >>>     nn.Linear(50, 5)    # Last linear layer
        >>> )
        >>> specific_bias_values = [0.1, -0.2, 0.3, 0.0, 0.5]
        >>> init_last_biases(model, specific_bias_values)
        Bias values set successfully!
    """
    last_linear_layer = None
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            last_linear_layer = layer

    if last_linear_layer is not None:
        if len(specific_bias_values) == last_linear_layer.out_features:
            last_linear_layer.bias.data = torch.tensor(specific_bias_values)
            print("Bias values set successfully!")
        else:
            print(f"Error: The provided bias values list should have {last_linear_layer.out_features} elements.")
            print(f"       Instead the list has {len(specific_bias_values)}.")

def test_predict(model: Union[Gru, GruBERT], weights_path: str, dataloader: DataLoader, device: Literal["cpu", "cuda"]):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    model.to(device)

    model.eval() # set model in evaluation mode

    all_outputs = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        with tqdm(total=len(dataloader), disable=False, leave=False, desc="Predictions") as pbar:
            for _, data in enumerate(dataloader, 0):

                labels = data['targets'].to(device, dtype = torch.float)
                outputs = model_pred(model, data, device, model_class=model.model_class)
            
                # Append predictions and true labels to the lists
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.update(1)

    return all_labels, all_outputs

def get_y_true(dataloader: DataLoader, device: str):
    '''Util to extract y_true from a dataloader'''
    all_labels = []
    for data in dataloader:
        labels = data['targets'].to(device, dtype=torch.float)
        all_labels.extend(labels.cpu().numpy())
    return all_labels


def get_lr_list(model, lr_i=1e-7, lr_f=1e-4):
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    #print(f'{idx}: {name}')

    trans_dict = {}
    trans_dict['before_trans'] = [layer for layer in layer_names if layer.split('.')[1] == 'embedding']
    for key in range(12):
        trans_dict[f'{key}_trans'] = [layer for layer in layer_names if f'{key}' in layer.split('.')]
    trans_dict['after_trans'] = [layer for layer in layer_names if layer not in trans_dict['before_trans'] and all(layer not in trans_dict[f'{key}_trans'] for key in range(12))]

    lr      = lr_i
    lr_mult = (lr_f/lr_i)**(1/(len(trans_dict.keys())-1))

    # placeholder
    parameters = []

    # store params & learning rates
    for idx, trans in enumerate(trans_dict):
    
        # display info
        #print(f'{trans}: lr = {lr:.6f}')
    
        for name in trans_dict[trans]:
            # append layer parameters
            parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':     lr}]
        
        # update learning rate
        lr *= lr_mult
    return parameters

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            alpha (float): Balancing factor for the class weights, typically between 0 and 1. 
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples, default is 2.
            weight (torch.Tensor): A tensor of shape [2] for the weights of each class, [weight for class 0, weight for class 1].
            reduction (str): Specifies the reduction to apply to the output, options are 'none', 'mean', 'sum'.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        # Compute the standard binary cross-entropy loss with logits
        bce_loss = self.bce(inputs, targets)
        
        # Calculate probabilities using the sigmoid function
        probs = torch.sigmoid(inputs)
        probs_t = probs * targets + (1 - probs) * (1 - targets)  # equivalent to pt

        # Apply focal loss modification
        focal_weight = (1 - probs_t) ** self.gamma
        
        # Apply alpha balancing factor
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final weighted focal loss
        loss = alpha_t * focal_weight * bce_loss
        
        # Reduce the loss if specified
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def save_loss_plot(train_loss, val_loss, file_path='loss_plot.png'):
    """
    Plots training and validation loss history and saves the plot to a file.
    
    Args:
        train_loss (list): A list of training loss values for each epoch.
        val_loss (list): A list of validation loss values for each epoch.
        file_path (str): The path where the plot image will be saved.
    """
    # Number of epochs based on length of training history
    epochs = range(1, len(train_loss) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    
    # Add title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig(file_path)
    plt.close()  # Close the plot to free memory

def bce(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets)

def wfl(outputs, targets):
    return WeightedFocalLoss().forward(outputs, targets)
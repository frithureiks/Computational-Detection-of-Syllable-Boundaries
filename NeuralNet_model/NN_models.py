import torch
from torch import nn
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CommonModel(nn.Module):
  '''
  This defines the fit, evaluate, and predict functions that are shared between all models.
  It is not instantiated directly but inherited by all other model classes.

  Methods:
    fit: Train model on train dataloader and evaluate on validation dataloader.
         Prints training losses and validation losses. No return.

    evaluate: Calculates losses on evaluation dataloader, prints and returns average loss across whole dataset.

    predict: Returns predictions and true labels from test dataloader.
             Predictions are list of word length with probabilities of each position being syllable break.
             True labels are binary list of word length, with 1 for syllable break and 0 otherwise.
  '''
  def __init__(self):
    super().__init__()

  def fit(self, train_dl, val_dl, epochs, lstm=False):
    self.loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(self.parameters())
    for epoch in range(epochs):
      running_loss = 0.0
      for i, data in enumerate(train_dl):
        optimizer.zero_grad()
        surprisal, syll, lang, word = data
        if lstm:
          surprisal = surprisal.unsqueeze(-1).to(torch.float32).to(device)
          syll = syll.unsqueeze(-1).to(torch.float32).to(device)
        else:
          surprisal = surprisal.unsqueeze(1).to(torch.float32).to(device)
          syll = syll.unsqueeze(1).to(torch.float32).to(device)
        prediction = self.forward(surprisal)
        loss = self.loss_fn(prediction, syll)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0 and i != 0:
          print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')
      print()

      val_loss = self.evaluate(val_dl, lstm=lstm)
      print('Validation Loss')
      print(val_loss)
      print()

    return running_loss / (i+1)

  def evaluate(self, val_dl, lstm=False):
    losses = []
    with torch.no_grad():
      for data in val_dl:
        surprisal, syll, lang, word = data
        if lstm:
          surprisal = surprisal.unsqueeze(-1).to(torch.float32).to(device)
          syll = syll.unsqueeze(-1).to(torch.float32).to(device)
        else:
          surprisal = surprisal.unsqueeze(1).to(torch.float32).to(device)
          syll = syll.unsqueeze(1).to(torch.float32).to(device)
        prediction = self.forward(surprisal)
        loss = self.loss_fn(prediction, syll)
        losses.append(loss)

    return torch.Tensor(losses).mean()

  def predict(self, test_dl, separate_words=False, lstm=False):
    with torch.no_grad():
      self.eval()
      predictions = []
      syllables = []
      activation = nn.Sigmoid()

      for data in test_dl:
        surprisal, syll, lang, word = data
        if lstm:
          surprisal = surprisal.unsqueeze(-1).to(torch.float32).to(device)
          syll = syll.unsqueeze(-1).to(torch.float32).to(device)
        else:
          surprisal = surprisal.unsqueeze(1).to(torch.float32).to(device)
          syll = syll.unsqueeze(1).to(torch.float32).to(device)
        pred = activation(self.forward(surprisal)).detach().cpu()

        if separate_words:
          if lstm:
            for i in range(syll.shape[0]):
              syllables.append(syll[i,:,0])
              predictions.append(pred[i,:,0])
          else:
            for i in range(syll.shape[0]):
              syllables.append(syll[i,0])
              predictions.append(pred[i,0])
        else:
          syll = syll.cpu().flatten().tolist()
          pred = activation(self.forward(surprisal)).flatten().tolist()
          syllables += syll
          predictions += pred

    return predictions, syllables



class NgramFNN(CommonModel):
  '''
  Feedforward Neural Network (FNN) that takes an ngram of size n_gram and feeds the surprisals through
  n hidden layers (n_layers) each of dimension d (d_hidden).

  *NOTE* This is implemented using Conv1d layers, however because we lock the kernel_size of all hidden
  layers and the output layer to 1, this is equivalent to a Feedforward Network operating over ngrams.

  Params:
    n_gram:   Size of ngram window to look over
    d_hidden: Dimensionality of hidden layers (equivalent to n_filters in NgramCNN)
    n_layers: Number of hidden layers (not including input and output layers)

  Returns:
    x: List of length of original word showing log-odds for each position to be a syllable break.
       This is converted to probabilities using sigmoid activation in "predict" method.

  Methods:
    See parent class "CommonModel"
  '''
  def __init__(self, n_gram, d_hidden, n_layers):
    super().__init__()

    self.n_gram = n_gram
    self.d_hidden = d_hidden
    self.n_layers = n_layers
    self.architecture = 'fnn'

    self.input_ = nn.Conv1d(in_channels=1, out_channels=d_hidden, kernel_size=n_gram, padding='same')
    self.output = nn.Conv1d(in_channels=d_hidden, out_channels=1, kernel_size=1)
    self.hidden = nn.ModuleList()
    for n in range(n_layers):
      self.hidden.append(nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=1))
      self.hidden.append(nn.GELU())
    self.activation = nn.GELU()

    self.size = sum(p.numel() for p in self.parameters())

  def forward(self, x):
    x = self.input_(x)
    x = self.activation(x)
    for hidden in self.hidden:
      x = hidden(x)
    x = self.output(x)

    return x



class NgramCNN(CommonModel):
  '''
  This is the model used in the main paper, a Convolutional Neural Network (CNN). Identical to the 
  NgramFNN, except that it applies the same window size (n_gram) to the hidden layers as it does to 
  the input layer, resulting in a kind of "recursive" ngram structure that incorporates information 
  from more and more neighboring ngrams the deeper the model goes.

  Params:
    n_gram:    Size of ngram window to look over
    n_filters: Number of filters to pass over each layer
    n_layers:  Number of hidden layers (not including input and output layers)

  Returns:
    x: List of length of original word showing log-odds for each position to be a syllable break.
       This is converted to probabilities using sigmoid activation in "predict" method.

  Methods:
    See parent class "CommonModel"
  '''
  def __init__(self, n_gram, n_filters, n_layers):
    super().__init__()

    self.n_gram = n_gram
    self.d_hidden = n_filters
    self.n_layers = n_layers
    self.architecture = 'cnn'

    self.input_ = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=n_gram, padding='same')
    self.output = nn.Conv1d(in_channels=n_filters, out_channels=1, kernel_size=1)
    self.hidden = nn.ModuleList()
    for n in range(n_layers):
      self.hidden.append(nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=n_gram, padding='same'))
      self.hidden.append(nn.GELU())
    self.activation = nn.GELU()

    self.size = sum(p.numel() for p in self.parameters())

  def forward(self, x):
    x = self.input_(x)
    x = self.activation(x)
    for hidden in self.hidden:
      x = hidden(x)
    x = self.output(x)

    return x



class NgramLSTM(CommonModel):
  '''
  Long Short-Term Memory (LSTM) recurrent neural network. Steps through the word one segment at a time
  to build an internal representation of the entire word. The probability of a segment being a syllable
  break is thus informed by all other segments in the word.

  Params:
    d_hidden:       Dimensionality of internal representation (comparable to n_filters in NgramCNN)
    n_layers:       Number of layers (not including output layer)
    bidirectional:  If True, the LSTM runs both forward and backward through the word; if False,
                    it only runs forward. The model used in the appendix sets bidrectional=True

  Returns:
    x: List of length of original word showing log-odds for each position to be a syllable break.
       This is converted to probabilities using sigmoid activation in "predict" method.

  Methods:
    See parent class "CommonModel"
  '''
  def __init__(self,
               d_hidden=5,
               n_layers=1,
               bidirectional=True):
    super().__init__()
    self.lstm = nn.LSTM(input_size=1,
                        d_hidden=d_hidden,
                        num_layers=n_layers,
                        bias=True,
                        batch_first=True,
                        dropout=0.0,
                        bidirectional=bidirectional,
                        proj_size=0,
                        device=None,
                        dtype=None)

    self.d_hidden = d_hidden
    self.n_layers = n_layers
    self.architecture = 'lstm'
    self.bidirectional=bidirectional

    if bidirectional:
      self.linear = nn.Linear(in_features= 2 * d_hidden, out_features=1)
    else:
      self.linear = nn.Linear(in_features=d_hidden, out_features=1)

    self.size = sum(p.numel() for p in self.parameters())

  def forward(self, x):
    x, hidden_out = self.lstm(x)
    x = self.linear(x)

    return x
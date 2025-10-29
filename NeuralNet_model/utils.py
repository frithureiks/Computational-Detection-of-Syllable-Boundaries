import numpy as np
import pandas as pd
from itertools import pairwise
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from nltk.probability import MLEProbDist, ConditionalFreqDist, ConditionalProbDist

def segment_surprisal(lexicon, syllables, language, sample_size=None, random_seed=None):
  '''
  Takes a lexicon and breaks it down into segments (phonemes or graphemes).
  Returns an N x 6 DataFrame, where N is the total number of segments, and
  the columns are:

    word_id: The index of the word the segment belongs to (starting at 1)
    position: The position of the segment within the word (starting at 1)
    segment: The segment itself (phoneme or grapheme)
    surprisal: The surprisal of that segment conditioned on all the preceding segments in the word.
    language: The language code of the lexicon.
    syllable: Binary marker for syllable breaks, 1 for the start of a new syllable, 0 otherwise.

  Params:
    lexicon (list(str)): List of vocab terms.
    syllables (list(int)): List of syllable breaks.
    language (str): Language code for lexicon.
    sample_size (int): Number of samples to take from lexicon.
                       If None, will take entire lexicon.
    random_seed (int): Random seed to make sampling reproducible.
                       If None, sampling will be non-reproducible.
                       If sample_size is None, this parameter is ignored.
  '''
  np.random.seed(seed=random_seed)

  # If sample_size is not None, draw that many samples from lexicon for vocab_list.
  if sample_size:
    if sample_size > len(lexicon):
      raise ValueError(f'Sample size of {sample_size} is larger than lexicon size of {len(lexicon)}')
    rand_idx = np.random.choice(len(lexicon), size=sample_size, replace=False)
    vocab_list = [lexicon[i] for i in rand_idx]
    syll_list = [syllables[i] for i in rand_idx]

  # Otherwise, use full lexicon.
  else:
    vocab_list = lexicon
    syll_list = syllables

  # Add ending character ">" to each word.
  vocab_list = [word+'>' for word in vocab_list]

  # Flatten syll_list so it is no longer separated by words
  syll_list = [syll for word in syll_list for syll in word]

  max_pos = max([len(word) for word in vocab_list])
  idx = list(range(len(vocab_list)))

  # Initialize empty lists for column values.
  word_id = []
  position = []
  segment = []
  surprisal = []

  for p in range(0, max_pos):
    # Create conditional probability distribution P(segment|context) up to position p.
    pos_list = [(word[0:p], word[p]) for word in vocab_list if len(word)>p]
    cond_freq_dist = ConditionalFreqDist(pos_list)
    cond_prob_dist = ConditionalProbDist(cond_freq_dist, MLEProbDist)

    # Add on to lists of word_ids, positions, segments, and surprisals up to position p.
    word_id += [i+1 for i in idx if len(vocab_list[i])>p]
    position += [p+1] * len(pos_list)
    segment += [ngram[1] for ngram in pos_list]
    surprisal += [-cond_prob_dist[ngram[0]].logprob(ngram[1]) for ngram in pos_list]

  # Add "lang" list, should match the length of other four lists.
  lang = [language] * len(word_id)

  # Create DataFrame with word_id, position, segment, surprisal, and lang columns.
  df = pd.DataFrame(list(zip(word_id, position, segment, surprisal, lang)),
                    columns=['word_id', 'position', 'segment', 'surprisal', 'lang'])

  # Remove rows with ending marker ">" from DataFrame
  df = df[df['segment'] != '>']


  # Make sure all surprisal values are >= 0, then change -0.0 to 0.0.
  if df['surprisal'].lt(0).any():
    raise ValueError('Surprisal cannot be < 0')
  df['surprisal'] = df['surprisal'].apply(abs)


  # Sort DataFrame first by word_id, then by position.
  df = df.sort_values(by=['word_id', 'position'], ignore_index=True)

  # Add syll_list to DataFrame
  df['syllable'] = syll_list

  return df



class WordDataset(Dataset):
  def __init__(self, dataframe):
    super().__init__()
    self.dataframe = dataframe

  def __len__(self):
    return self.dataframe['word_id'].max()

  def __getitem__(self, idx):
    word_id = idx + 1
    word_df = self.dataframe[self.dataframe['word_id']==word_id]

    surprisal = word_df['surprisal'].to_numpy()
    syll = word_df['syllable'].to_numpy()
    lang = word_df['lang'].iloc[0]
    word = ''.join(word_df['segment'].tolist())

    return surprisal, syll, lang, word



# EqualLengthsBatchSampler adapted from
# https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/27

class EqualLengthsBatchSampler(Sampler):
    '''
    Groups samples into batches based on sample-length, e.g. all 8-letter words appear only 
    with other 8-letter words. This eliminates the need for padding.

    It also makes the batch sizes as close to equal as possible while not overstepping the 
    specified batch_size. For example, if batch_size is 64 and there are 65 samples, rather 
    than create batches of size [64, 1], it will create batches of size [33, 32]. This also 
    means that, as long as there are more samples than the specified batch_size, the smallest 
    batch will always be at least batch_size // 2.

    Based on this, we delete any sample lengths that have fewer than batch_size // 2 samples. 
    This is because some sample-lengths have very few or even just one sample in the dataset. 
    If these small groups of samples are passed along as their own batches, they will be given 
    equal weight to groups of samples up to the full batch_size, e.g. a single sample of length 
    31 would be given the same weight as 64 samples of length 8, assuming batch_size = 64.

    Each of the 10 epochs will give uniquely shuffled batches, but if you run the code block 
    again it will give you the same 10 batches as the first time you ran it. If you don't want 
    this behavior, feed in `seed=None`.

    **Note:** To get deterministic behavior from the EqualLengthsBatchSampler, you need to 
    initialize it with a seed each time you initialize the corresponding DataLoader. If you 
    initialize a new DataLoader without re-initializing the Sampler, it will not reset the 
    RNG within the sampler and so you will get new samples for the new DataLoader. 
    Example code is show below:

    ```
    train_sampler = EqualLengthsBatchSampler(train_ds, batch_size, seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler)
    fnn_model = NgramFNN(n_gram=7, d_hidden=15, n_layers=5).to(device)
    fnn_model.fit(train_dl, val_dl, epochs=10, loss_fn='bce')
    ```
    '''

    def __init__(self, dataset, batch_size, seed):

        # Set random seed
        self.rng = np.random.default_rng(seed)

        # Remember batch size and number of samples
        self.batch_size = batch_size

        self.unique_lengths = set()
        self.samples = defaultdict(list)

        for i in range(0, len(dataset)):
            len_input = len(dataset[i][0])

            # Add length to set of all seen lengths
            self.unique_lengths.add(len_input)

            # For each length, keep track of which sample indices for this length
            # E.g.: self.lengths_to_sample = { 4: [3,5,11], 5: [1,2,9], ...}
            self.samples[len_input].append(i)

        # Delete lengths and corresponding samples if there are fewer than batch_size // 2
        # samples of that length.
        self.small_samples = set()
        for length in self.unique_lengths:
            if len(self.samples[length]) < batch_size // 2:
                self.small_samples.add(length)

        self.unique_lengths = self.unique_lengths - self.small_samples

        # Convert set of unique lengths to a list so we can shuffle it later
        self.unique_lengths = list(self.unique_lengths)

    def __len__(self):
        batches = 0
        for length in self.unique_lengths:
          batches += np.ceil(len(self.samples[length]) / self.batch_size).astype(int)
        return batches

    def __iter__(self):

        # Make list to store all batches of any length
        all_batches = []

        # Shuffle list of unique length pairs
        self.rng.shuffle(self.unique_lengths)

        # Iterate over all possible word lengths
        for length in self.unique_lengths:

            # Get indices of all samples for the current lengths
            # for example, all indices with a length of 8
            sequence_indices = self.samples[length]
            sequence_indices = np.array(sequence_indices)

            # Shuffle array of sequence indices
            self.rng.shuffle(sequence_indices)

            # Compute the number of batches
            num_batches = np.ceil(len(sequence_indices) / self.batch_size)

            # Loop over all possible batches of given length and add to list of all batches
            all_batches += [batch_indices for batch_indices in np.array_split(sequence_indices, num_batches)]

        # Shuffle list of all batches; this shuffles the order of batches but keeps their internal structure the same
        self.rng.shuffle(all_batches)
        for batch in all_batches:
          yield(np.asarray(batch))


    def show_batches(self):
      '''
      Print the different possible word lengths, the number of samples with each word length,
      the number of batches of size self.batch_size that can be made out of those samples,
      and the remainder, i.e. the number of samples in the final, smallest batch.
      (Note: if remainder is 0, that means the number of samples falls perfectly in n batches.)
      '''
      print(f'Length    # Samples    # Batches    Avg Batch Size')
      for length in self.unique_lengths:
          num_samples = len(self.samples[length])
          num_batches = np.ceil(num_samples / self.batch_size)
          average = num_samples / num_batches
          print(f'{length:>6} {num_samples:>12} {num_batches:>12.0f} {average:>12.1f}')

    def dropped_samples(self):
      '''
      Return dictionary of all words dropped from dataset for having too few samples of that length.
      '''
      small_samples = {}
      for length in self.small_samples:
        small_samples[length] = self.samples[length]
      return small_samples

    def show_dropped_samples(self):
      '''
      Print the word lengths that were dropped from the dataset, and the total number of words of each length.
      '''
      dropped_samples = self.dropped_samples()
      print('Dropped Samples \n')
      print('Length    # Samples')
      for key, value in dropped_samples.items():
        print(f'{key:>6} {len(value):>12}')



def format_k(value, tick_number):
    if value >= 1000:
        return f'{int(value / 1000)}k'
    else:
        return f'{int(value)}'
  


def syllabify(entry, syllables, segments):
    if entry[syllables] == 0 or entry['Positions'] == 1:
        return entry[segments]
    else:
        return '^' + entry[segments]



def match_syllables(row, sylls, ref_sylls):
    '''
    Compare a list of syllables (sylls) against a reference list of syllables (ref_sylls) and return
    a list with [True] for syllables that match their corresponding reference syllables and [False] for
    syllables that don't.
    '''
    sylls = row[sylls]
    ref_sylls = row[ref_sylls]
    breaks = [i for i, num in enumerate(sylls) if num==1]
    slices = [slice(pair[0],pair[1]+1) for pair in pairwise(breaks)]
    return [sylls[s]==ref_sylls[s] for s in slices] + [sylls[breaks[-1]:]==ref_sylls[breaks[-1]:]]
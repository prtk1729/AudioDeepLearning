import tensorflow as tf
import tensorflow.keras as keras 
# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import numpy as np


class MelodyPreprocessor:
    '''
    Prepares Dataset for training
        Input: Melody encoded as text
        Output: Returns tf datasets object used to training Seq-to-Seq models
    '''
    def __init__(self, dataset_path, batch_size=32):
        self.batch_size = batch_size
        self.path = dataset_path
        # some attributes as we process the input file
        self.max_melody_length = None
        self.number_of_tokens = None
        # init the Tokenizer to encode and tokenise melodies
        self.tokenizer = Tokenizer(filters="", 
                                    lower=False,
                                    split=",") # filters based on regex, split<separator>?, lower

    def prepare_training_data(self):
        # load the data as a dict   
        d = self._load_data()
        # print(d) # list of strings ['', '', '']
        # We get a list of melodies -> Each string is a melody
        # We further go over each melody and split into notes
        list_of_melodies = [ self._parse_melody(melody) for melody in d ]
        # print( list_of_melodies[0] ) # [C4-1.0] imagine 28 of these makes a melody
        # print( len(list_of_melodies[0]) ) # [ ['note1', 'note2'], ['', ''] ]
        # We then tokenise the list of melodies
        tokenised_list_of_melodies = self._tokenise_and_encode_melodies(list_of_melodies)
        # print( len(tokenised_list_of_melodies) )
        # print( len(tokenised_list_of_melodies[0]) ) # [2, 2, 5, 5, 10, 10, 8, 4, 4, 1, 1, 3, 3, 12, 5, 5, 4, 4, 1, 1, 7, 5, 5, 4, 4, 1, 1, 7]
        # encoded C4-1.0 to 2

        # some setters
        self.max_melody_length = self._set_max_length_melody(tokenised_list_of_melodies)
        self.number_of_tokens = len(self.tokenizer.word_index)

        # create input and target sequence pairs for each melody
        input_sequences, target_sequences = self._create_input_and_target_sequence_pairs(
                                                                                        tokenised_list_of_melodies
                                                                                      )
        # convert to tf.data.Datasets object
        shuffled_batched_dataset = self._convert_to_tf_dataset(input_sequences, target_sequences)

        print( type(shuffled_batched_dataset) )
        
    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def _create_input_and_target_sequence_pairs(self, tokenised_list_of_melodies):

        input_sequences, target_sequences = [], []
        for melody_as_list in tokenised_list_of_melodies:
            # [1, 2, 3, 4] -> ([1], [1, 2]), ([1, 2], [1, 2, 3]), ([1, 2, 3], [1,2,3,4])
            for i in range(1, len(melody_as_list)):
                input = melody_as_list[: i]
                target = melody_as_list[: i+1]
                # uniformise both (input, target) as diff melodies diff length
                input = self._pad_sequences(input)
                target = self._pad_sequences(target)
                input_sequences.append(input)
                target_sequences.append(target)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequences(self, sequence):
        ret = sequence
        to_be_padded = [0] * (self.max_melody_length - len(sequence))
        ret += to_be_padded
        return ret


    # def _get_shuffled_batched_dataset(self, input_sequences, target_sequences):
    #     tf_data = tf.data.Dataset.from_tensor_slices(
    #                                                     (input_sequences, target_sequences)
    #                                                 )
    #     # shuffled_data = tf_data.shuffle(buffer=1000)
    #     # batched_shuffled_data = shuffled_data.batch(batch_size=self.batch_size)
    #     # return batched_shuffled_data
    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a TensorFlow Dataset.

        Parameters:
            input_sequences (list): Input sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled
                TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset
    

    def _load_data(self):
        with open(self.path, "r") as fp:
            return json.load(fp)

    def _parse_melody(self, melody_as_string):
        return melody_as_string.split(", ") # returns list of notes within a given melody

    def _tokenise_and_encode_melodies(self, list_of_melodies):
        self.tokenizer.fit_on_texts(list_of_melodies)
        # once fit now it knows how to encode
        tokenised_melodies = self.tokenizer.texts_to_sequences(list_of_melodies)
        return tokenised_melodies
    
    def _set_max_length_melody(self, tokenised_melodies):
        return max([ len(melody) for melody in tokenised_melodies ])

if __name__ == "__main__":
    mp = MelodyPreprocessor(dataset_path="./data.json", batch_size=2)
    mp.prepare_training_data()
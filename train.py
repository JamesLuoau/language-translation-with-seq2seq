from collections import Counter

import helper
import problem_unittests as tests
import numpy as np


def print_info():
    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))
    print('Number of sentences: {}'.format(len(sentences)))
    print('Average number of words in a sentence: {}'.format(np.average(word_counts)))
    print()

    view_sentence_range = (0, 10)
    print('English sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
    print()
    print('French sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


def preprocess(text):
    words = text.replace("\"", "").split()
    print(words[:30])
    print("Total words: {}".format(len(words)))

    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    print("Unique words: {}".format(len(word_counts)))

    int_words = [vocab_to_int[word] for word in words]
    return words, vocab_to_int, int_to_vocab, int_words


def text_to_id(text, vocab_to_int):
    sentences = [word for word in text.split("\n")]
    return list(map(lambda sentence: [vocab_to_int[word] for word in sentence.split()], sentences))


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    def add_eos(ids):
        ids.append(target_vocab_to_int['<EOS>'])
        return ids

    return text_to_id(source_text, source_vocab_to_int), \
           [add_eos(ids) for ids in text_to_id(target_text, target_vocab_to_int)]


if __name__ == '__main__':
    source_path = 'data/small_vocab_en'
    target_path = 'data/small_vocab_fr'
    source_text = helper.load_data(source_path)
    target_text = helper.load_data(target_path)

    sentences = source_text.split('\n')
    word_counts = [len(sentence.split()) for sentence in sentences]

    tests.test_text_to_ids(text_to_ids)

    helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
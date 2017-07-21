from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
import helpers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class AwiModel(object):

    PAD = 0
    EOS = 1

    def __init__(self, vocab_size, num_units, embedding_size, debug=False):
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.embedding_size = embedding_size
        self.debug = debug

        self._make_graph()

    def _make_graph(self):
        if self.debug:
            self._init_debug_inputs()
        else:
            self._init_placeholder()
        self._init_decoder_train_connectors()
        self._init_embeddings()
        self._init_simple_encoder()
        self._init_decoder()
        self._init_optimizer()

        print('make graph done.')

    def _init_debug_inputs(self):
        """ Everything is time-major """

        print('Init debug inputs...')

        x = [[5, 6, 7],
             [7, 6, 0],
             [0, 7, 0]]
        xl = [2, 3, 1]

        self.encoder_inputs = tf.constant(
            x, dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.constant(
            xl, dtype=tf.int32, name='encoder_inputs_length')

        self.decoder_targets = tf.constant(
            x, dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.constant(
            xl, dtype=tf.int32, name='decoder_targets_length')

    def _init_placeholder(self):
        """ all is time-major """

        print('Init placeholders...')

        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        print('Init decoder train connectors...')

        with tf.name_scope('DecoderTrainFeeds'):
            print('self.decoder_targets.shape:',
                  tf.shape(self.decoder_targets))
            sequence_size, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))
            print('sequence_size:', sequence_size)
            print('batch_size:', batch_size)

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat(
                [EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], axis=0)
            self.decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        self.decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])

            # hacky way using one_hot to put EOS symbol at the end of target
            # sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        print('Init enbeddings...')

        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            import math
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        print('Init simple encoder...')

        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.num_units)
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
            )

    def _init_decoder(self):
        print('Init decoder...')

        with tf.variable_scope('decoder') as scope:
            _, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))
            self.decoder_cell = tf.contrib.rnn.LSTMCell(self.num_units)

            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            # attention_states: size [batch_size, max_time, num_units]
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

            train_helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_train_inputs_embedded, self.decoder_targets_length)
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_units, memory=attention_states,
                memory_sequence_length=self.decoder_targets_length)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, attention_mechanism, attention_layer_size=self.num_units)
            initial_state = self.decoder_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, train_helper, initial_state=initial_state)
            self.decoder_outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder, impute_finished=True)

            self.decoder_train_logits = output_fn(self.decoder_outputs)
            self.decoder_train_prediction = tf.argmax(
                self.decoder_logits_train, axis=-1)

    def _init_optimizer(self):
        print('Init optimizer...')

        logits = tf.transpose(self.decoder_train_logits, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets,
                                                     weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def make_seq2seq_model(**kwargs):
    args = dict(vocab_size=10, num_units=30,
                embedding_size=11, debug=False)
    args.update(kwargs)
    return AwiModel(**args)


def train_on_copy_task(session, model,
                       length_from=3,
                       length_to=8,
                       vocab_lower=2,
                       vocab_upper=10,
                       batch_size=100,
                       max_batches=5000,
                       batches_in_epoch=1000,
                       verbose=True):

    def make_train_inputs(model, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length = helpers.batch(target_seq)

        return{
            model.encoder_inputs: inputs_,
            model.encoder_inputs_length: inputs_length_,
            model.decoder_targets: targets_,
            model.decoder_targets_length: targets_length
        }

    batches = helpers.random_sequences(length_from=length_from,
                                       length_to=length_to,
                                       vocab_lower=vocab_lower,
                                       vocab_upper=vocab_upper,
                                       batch_size=batch_size)
    loss_track = []

    try:
        for batch in xrange(max_batches + 1):
            batch_data = next(batches)
            fd = make_train_inputs(model, batch_data, batch_data)
            _, l = session.run([model.train_op, model.loss], fd)
            loss_track.append(l)

            if verbose:
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('minibatch loss:{}'.format(
                        session.run(model.loss, fd)))
                    for i, (en_inp, de_pred) in \
                            enumerate(zip(fd[model.encoder_inputs].T,
                                          session.run(model.decoder_train_prediction, fd).T)):
                        print('sample {}:'.format(i + 1))
                        print('encoder input            > {}'.format(en_inp))
                        print('decoder train prediction > {}'.format(de_pred))
                        if i >= 2:
                            break
                    print()
    except KeyboardInterrupt:
        print('training interrupted!')

    return loss_track


if __name__ == '__main__':
    import sys

    if 'debug' in sys.argv:
        tf.reset_default_graph()
        with tf.Session() as session:
            model = make_seq2seq_model(debug=True)
            session.run(tf.global_variables_initializer())
            session.run(model.decoder_train_prediction)

    elif 'train' in sys.argv:
        batch_size = 100

        tf.reset_default_graph()
        with tf.Session() as session:
            model = make_seq2seq_model()
            session.run(tf.global_variables_initializer())
            loss_track = train_on_copy_task(
                session, model, batch_size=batch_size)

        import matplotlib.pyplot as plt
        plt.plot(loss_track)
        print('loss {:.4f} after {} examples (batch_size={})'.format(
            loss_track[-1], len(loss_track) * batch_size, batch_size))

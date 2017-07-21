#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function


import sys
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


class Config:

    EMBEDDING_SIZE = 128
    ENCODER_SEQ_LENGTH = 5
    ENCODER_NUM_STEPS = ENCODER_SEQ_LENGTH
    DECODER_SEQ_LENGTH = ENCODER_NUM_STEPS + 1  # plus 1 EOS
    DECODER_NUM_STEPS = DECODER_SEQ_LENGTH

    def __init__(self, file_):
        print('building volcabulary...')
        self.file_ = file_
        self._build_dic(file_)

    def _build_dic(self, file_):
        set_ = set()
        lnum = 0
        self.lines = []
        with open(file_, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip('\n')
                lnum += 1
                if not line:
                    continue
                self.lines.append(line)
                for cc in line:
                    # cc = cc.encode('utf-8')
                    set_.add(cc)

            print('built size of ', len(set_), ' dictionary', lnum)
        self.chars = []
        self.dic = {}

        index = 0
        for char in set_:
            self.chars.append(char)
            self.dic[char] = index
            index = index + 1
        self.chars.append('#EOS#')
        self.dic['#EOS#'] = index

        self.VOL_SIZE = len(self.chars)
        self.EOS = self.VOL_SIZE - 1

    def gen_triple(self, check=0):
        if check >= len(self.lines) - 1:
            check = 0
        source = self.lines[check]
        target = self.lines[check + 1]
        check = check + 2
        source = source.decode('utf-8')
        target = target.decode('utf-8')
        source_index = []
        for c in source:
            source_index.append(self.dic[c])
        target_index = []
        label_index = []
        for c in target:
            target_index.append(self.dic[c])
            label_index.append(self.dic[c])

        encoder_input = np.array(source_index)
        decoder_input = np.array(target_index)
        labels = np.array(label_index)

        # append EOS to decoder
        decoder_input = np.append(self.EOS, decoder_input)
        labels = np.append(labels, self.EOS)
        return check, encoder_input, decoder_input, labels

    def get_batch_data(self, batch_size):
        iter_ = 0
        while True:
            encoder_inputs = []
            decoder_inputs = []
            labels = []
            pad = (batch_size - 1) * 2
            check = iter_
            flag = False
            for bb in xrange(batch_size):
                check, a, b, c = self.gen_triple(check)
                if check == 2:
                    if iter_ > 0:
                        iter_ = -2
                        flag = True
                        break

                check += pad
                encoder_inputs.append(a)
                decoder_inputs.append(b)
                labels.append(c)
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = np.array(decoder_inputs)
            labels = np.array(labels)
            iter_ += 2
            if flag:
                yield [], [], []
            else:
                yield encoder_inputs, decoder_inputs, labels

    def recover(self, index):
        sentence = []
        for ii in index:
            sentence.append(self.chars[int(ii)])
        return ''.join(sentence)

if __name__ == '__main__':
    config = Config('../data/poem.txt')
    # with open('../../data/log.txt', 'w') as log_:
    batch_size = 32
    for a, b, c in config.get_batch_data(batch_size):
        for i in xrange(len(a)):
            print(config.recover(
                a[i]) + config.recover(b[i]) + config.recover(c[i]), len(a))
        print('===========')

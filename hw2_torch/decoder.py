import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel


class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(
            len(extractor.word_vocab), len(extractor.output_labels)
        )
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.output_labels.items()]
        )

    def parse_sentence(self, words, pos):

        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            feature = self.extractor.get_input_representation(words, pos, state)
            feature = torch.LongTensor(feature).unsqueeze(0)  # torch.Size([1, 6])

            log_probs = self.model(feature)  # torch.Size([1, 91])
            probs = torch.exp(log_probs).detach().numpy()[0]  # (91,)

            indices_sorted = np.argsort(-probs)
            # breakpoint()

            for idx in indices_sorted:
                (trans, label) = self.output_labels[idx]  # ('shift', None)
                if self.can_apply_transition(trans, state):
                    if trans == "shift":
                        state.shift()
                    elif trans == "left_arc":
                        state.left_arc(label)
                    elif trans == "right_arc":
                        state.right_arc(label)
                    break

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))

        return result

    def can_apply_transition(self, transition, state):
        if len(state.stack) == 0 and transition in ("left_arc", "right_arc"):
            return False

        if transition == "left_arc" and state.stack[-1] == 0:
            return False

        if transition == "shift":
            if len(state.buffer) == 1 and len(state.stack) > 0:
                return False

        return True


if __name__ == "__main__":

    WORD_VOCAB_FILE = "data/words.vocab"
    # POS_VOCAB_FILE = "data/pos.vocab"

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, "r")
        # pos_vocab_f = open(POS_VOCAB_FILE, "r")
    except FileNotFoundError:
        print(
            "Could not find vocabulary files {} and {}".format(
                WORD_VOCAB_FILE,  # POS_VOCAB_FILE
            )
        )
        sys.exit(1)

    # extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    extractor = FeatureExtractor(word_vocab_f)

    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], "r") as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

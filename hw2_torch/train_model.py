import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader

from extract_training_data import FeatureExtractor

SEED = 42

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


class DependencyDataset(Dataset):

    def __init__(self, input_filename, output_filename):
        self.inputs = np.load(input_filename)
        self.outputs = np.load(output_filename)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, k):
        return (self.inputs[k], self.outputs[k])


class DependencyModel(Module):
    """
    Input := [batch, 6]
    Emb := [batch, 6 * 128]
    Hidden := [batch, 128]
    relu activation
    out := [batch, 91]
    softmax
    """

    def __init__(self, word_types, outputs):
        super(DependencyModel, self).__init__()
        self.embedding = Embedding(
            embedding_dim=128,
            num_embeddings=word_types,
        )
        self.hidden = Linear(
            in_features=6 * 128,
            out_features=128,
        )
        self.output = Linear(
            in_features=128,
            out_features=outputs,
        )

    def forward(self, inputs) -> torch.Tensor:

        x = self.embedding(inputs)  # [batch_size, 6, 128]
        x = x.view(x.size(0), -1)  # [batch_size, 768]
        outputs = relu(self.hidden(x))  # [batch_size, 128]
        logits = self.output(outputs)  # [batch_size, outputs]
        return log_softmax(logits, dim=1)  # [batch_size, outputs]


def train(model, loader):

    loss_function = NLLLoss(reduction="mean")

    LEARNING_RATE = 0.01
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

    tr_loss = 0
    tr_steps = 0

    # put model in training mode
    model.train()

    correct = 0
    total = 0
    for idx, batch in enumerate(loader):

        inputs, targets = batch

        inputs = torch.LongTensor(inputs).to(DEVICE)
        targets_tensor = torch.FloatTensor(targets).to(DEVICE)
        targets_idx = torch.argmax(targets_tensor, dim=1)

        logits = model(torch.LongTensor(inputs))
        # loss = loss_function(logits, targets)
        loss = loss_function(logits, targets_idx)
        tr_loss += loss.item()

        # print("Batch loss: ", loss.item()) # Helpful for debugging, maybe

        tr_steps += 1

        if idx % 1000 == 0:
            curr_avg_loss = tr_loss / tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # To compute training accuracy for this epoch
        # correct += sum(torch.argmax(logits, dim=1) == torch.argmax(targets, dim=1))
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets_idx).sum().item()
        total += len(inputs)

        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / tr_steps
    acc = correct / total
    print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


def _set_dataloader_seed():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


if __name__ == "__main__":

    WORD_VOCAB_FILE = "data/words.vocab"

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, "r")
    except FileNotFoundError:
        print("Could not find vocabulary file {}.".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)

    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels)).to(
        DEVICE
    )

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(SEED),
        generator=_set_dataloader_seed(),
    )

    print("Done loading data")

    # Now train the model
    for i in range(5):
        train(model, loader)

    torch.save(model.state_dict(), sys.argv[3])

"""
Current average loss: 4.542828559875488
...
Current average loss: 0.6142518064033918
Training loss epoch: 0.6137977921514869,   Accuracy: 0.8141392636767518
...
Training loss epoch: 0.4571533832745658,   Accuracy: 0.8575913165385552
...
Training loss epoch: 0.3503251470411593,   Accuracy: 0.8892045828443937
"""

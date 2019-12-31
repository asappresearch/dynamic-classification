import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from flambe.dataset import TabularDataset
from flambe.field import TextField, LabelField
from tensorboardX import SummaryWriter

from sampler import BaseSampler, EpisodicSampler
from model import PrototypicalTextClassifier


def train(args):
    """Run Training """

    global_step = 0
    best_metric = None
    best_model: Dict[str, torch.Tensor] = dict()
    writer = SummaryWriter(log_dir=args.output_dir)

    # We use flambe to do the data preprocessing
    # More info at https://flambe.ai
    text_field = TextField(lower=args.lowercase, embeddings=args.embeddings)
    label_field = LabelField()
    transforms = {'text': text_field, 'label': label_field}
    dataset = TabularDataset.from_path(args.train_path,
                                       args.val_path,
                                       sep=',',
                                       transforms=transforms)

    # Create samplers
    train_sampler = EpisodicSampler(dataset.train,
                                    n_support=args.n_support,
                                    n_query=args.n_query,
                                    n_episodes=args.n_episodes,
                                    n_classes=args.n_classes)

    # The train_eval_sampler is used to computer prototypes over the full dataset
    train_eval_sampler = BaseSampler(dataset.train, batch_size=args.eval_batch_size)
    val_sampler = BaseSampler(dataset.val, batch_size=args.eval_batch_size)

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Build model, criterion and optimizers
    model = PrototypicalTextClassifier(vocab_size=dataset.text.vocab_size,
                                       distance=args.distance,
                                       embedding_dim=args.embedding_dim,
                                       pretrained_embeddings=dataset.text.embedding_matrix,
                                       rnn_type='sru',
                                       n_layers=args.n_layers,
                                       hidden_dim=args.hidden_dim,
                                       freeze_pretrained_embeddings=True)

    loss_fn = nn.CrossEntropyLoss()

    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    for epoch in range(args.num_epochs):

        ######################
        #       TRAIN        #
        ######################

        print(f'Epoch: {epoch}')

        model.train()

        with torch.enable_grad():
            for batch in train_sampler:
                # Zero the gradients and clear the accumulated loss
                optimizer.zero_grad()

                # Move to device
                batch = tuple(t.to(device) for t in batch)

                # Compute loss
                pred, target = model(*batch)
                loss = loss_fn(pred, target)
                loss.backward()

                # Clip gradients if necessary
                if args.max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)

                writer.add_scalar('Training/Loss', loss.item(), global_step)

                # Optimize
                optimizer.step()
                global_step += 1

            # Zero the gradients when exiting a train step
            optimizer.zero_grad()

        #########################
        #       EVALUATE        #
        #########################

        model.eval()

        with torch.no_grad():

            # First compute prototypes over the training data
            encodings, labels = [], []
            for text, label in train_eval_sampler:
                text_encoding = model.embedder(text)
                labels.append(label.cpu())
                encodings.append(text_encoding.cpu())
            # Compute prototypes
            encodings = torch.cat(encodings, dim=0)
            labels = torch.cat(labels, dim=0)
            prototypes = model.compute_prototypes(encodings, labels).to(device)

            _preds, _targets = [], []
            for batch in val_sampler:
                # Move to device
                batch = tuple(t.to(device) for t in batch)

                pred, target = model(*batch, prototypes=prototypes)
                _preds.append(pred.cpu())
                _targets.append(target.cpu())

            preds = torch.cat(_preds, dim=0)
            targets = torch.cat(_targets, dim=0)

            val_loss = loss_fn(preds, targets).item()
            val_metric = (pred.argmax(dim=1) == target).float().mean().item()

        # Update best model
        if best_metric is None or val_metric > best_metric:
            best_metric = val_metric
            best_model_state = model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            best_model = best_model_state

        # Log metrics
        print(f'Validation loss: {val_loss}')
        print(f'Validation accuracy: {val_metric}')
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_metric, epoch)

    # Save the best model
    with open(os.path.join(args.output_dir, 'model.pt'), 'w') as f:
        torch.save(best_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('train_path', type=str, required=True,
                        help="Path to training data. Should be a CSV with \
                              the text in the first column and label in the second")
    parser.add_argument('val_path', type=str, required=True,
                        help="Path to validation data. Should be a CSV with \
                              the text in the first column and label in the second")
    parser.add_argument('output_dir', type=str, required=True,
                        help="Path to output directory")

    # Optional
    parser.add_argument('distance', type=str, choices=['euclidean', 'hyperbolic'],
                        default='euclidean', help="Distance metric to use.")
    parser.add_argument('device', type=str, default=None, help="Device to use.")
    parser.add_argument('lowercase', type=bool, default=False, help="Whether to lowercase the text")
    parser.add_argument('embeddings', type=str, default='glove-wiki-gigaword-300',
                        help="Gensim embeddings to use.")
    parser.add_argument('n_layers', type=int, default=2, help="Number of layers in the RNN.")
    parser.add_argument('hidden_dim', type=int, default=128, help="Hidden dimension of the RNN.")
    parser.add_argument('embedding_dim', type=int, default=128, help="Dimension of the token embeddings.")

    parser.add_argument('n_support', type=int, default=1, help="Number of support points per class.")
    parser.add_argument('n_query', type=int, default=64, help="Total number of query points (not per class)")
    parser.add_argument('n_classes', type=int, default=None, help="Number of classes per episode")
    parser.add_argument('n_episodes', type=int, default=100, help="Number of episodes per 'epoch'")
    parser.add_argument('num_epochs', type=int, default=100, help="Number of training and evaluation steps.")
    parser.add_argument('eval_batch_size', type=int, default=128, help="Batch size used during evaluation.")
    parser.add_argument('learning_rate', type=float, default=0.001, help="The learning rate.")

    args = parser.parse_args()
    train(args)

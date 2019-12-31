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
    parser.add_argument('')

    args = parser.parse_args()
    train(args)

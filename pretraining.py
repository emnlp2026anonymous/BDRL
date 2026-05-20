import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from vocabulary import SMILESTokenizer, read_vocabulary
from dataset import Dataset
from model import GPT, GPTConfig
from utils import model_validity


def get_lr(it, total_it):
    warmup_iters = args.warmup * total_it
    if it < warmup_iters: # linear warmup        
        lr_mult = it / warmup_iters
    else: # cosine learning rate decay        
        decay_ratio  = (it - warmup_iters) / (total_it - warmup_iters)
        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
    return args.learning_rate * lr_mult


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gpt")
    parser.add_argument('--run_name', type=str, help="name for tensorboard run", default='')
    parser.add_argument('--dataset', type=str, help="name of dataset", default='chembl')
    parser.add_argument('--n_layer', type=int, default=12, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=384, help="embedding dimension", required=False)
    parser.add_argument('--num_epochs', type=int, default=20, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate", required=False)
    parser.add_argument('--lr_decay', type=bool, default=True, help="whether learning rate decays", required=False) 
    parser.add_argument('--warmup', type=float, default=0.01, help="warmup iters", required=False) 
    parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help="gradient normalization clip", required=False)
    parser.add_argument('--aug_prob', type=float, default=1.0, help="SMILES randomization prob", required=False)
    parser.add_argument('--max_length', type=int, default=128, help="max length of SMILES", required=False)
    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt", required=False)
    parser.add_argument('--ckpt_load_path', type=str, default=None, required=False)
    parser.add_argument('--ckpt_save_path', type=str, default="ckpt/", required=False)
    args = parser.parse_args()
    
    writer = SummaryWriter("logs/" + args.run_name)
    if not os.path.exists(args.ckpt_save_path + args.run_name):
        os.makedirs(args.ckpt_save_path + args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    if args.dataset == "chembl":
        data = pd.read_csv("data/ChEMBL_Smiles.csv")
        data = data['Smiles']
    else:
        Exception("Undefined dataset!")
    print(len(data))

    # Vocabulary
    if args.vocab_path != None:
        voc = read_vocabulary(args.vocab_path)
        print("Read vocabulary from: ", args.vocab_path)

    # Split train / val set
    train_data = data[:int(0.99 * len(data))]
    val_data = data[int(0.99 * len(data)):]
    train_dataset = Dataset(smiles_list=train_data, vocabulary=voc, tokenizer=SMILESTokenizer(), aug_prob=args.aug_prob)
    val_dataset = Dataset(smiles_list=val_data, vocabulary=voc, tokenizer=SMILESTokenizer(), aug_prob=args.aug_prob)
    print("Training size: ", train_dataset.__len__(), ", Validation size: ", val_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=Dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Model
    if args.model_type == "gpt":
        model_config = GPTConfig(voc.__len__(), n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.max_length)
        model = GPT(model_config).to("cuda")
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate, betas=(0.9, 0.95))

    if args.ckpt_load_path != None:
        model.load_state_dict(torch.load(args.ckpt_load_path), strict=True)

    scaler = torch.cuda.amp.GradScaler()
    model = torch.nn.DataParallel(model, device_ids=[0])

    num_batches = len(train_loader)
    for epoch in tqdm(range(args.num_epochs)):
        # training
        model.train()
        pbar = tqdm(enumerate(train_loader), total=num_batches, leave=False)
        for iter_num, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            lr = get_lr(iter_num + num_batches * epoch, num_batches * args.num_epochs) if args.lr_decay else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            writer.add_scalar('learning rate', lr, iter_num + num_batches * epoch)

            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(True):
                    logits, loss = model(x, y)
                    loss = loss.mean()
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f"epoch {epoch + 1}, iter {iter_num}: train loss {loss.item():.5f}, lr {lr:e}")
            writer.add_scalar('training loss', loss, iter_num + num_batches * epoch)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for iter_num, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                logits, loss = model(x, y)
                loss = loss.mean()
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        validity = model_validity(model, vocab_path=args.vocab_path, block_size=args.max_length)
        writer.add_scalar('validation loss', loss, epoch)
        writer.add_scalar('SMILES validity', validity, epoch)

        # save checkpoint
        torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/" + f"epoch{epoch}.pt")

    torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/" + f"final.pt")
        

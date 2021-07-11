import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import datetime


class Engine:
    def __init__(self):
        pass

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def train_fn(self, data_loader, model, optimizer, device, schedular):
        model.train()
        final_loss = 0
        t0_epoch = time.time()
        for data in tqdm(data_loader, total=len(data_loader)):
            b_input_ids = data['input_ids']
            b_attention_mask = data['attention_mask']
            b_token_type_ids = data['token_type_ids']
            b_target_pos = data['target_pos']
            b_target_tag = data['target_tag']

            # moving tensors to device
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            b_target_pos = b_target_pos.to(device)
            b_target_tag = b_target_tag.to(device)

            # optimizer.zero_grad()

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            o1, o2, loss = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                token_type_ids=b_token_type_ids,
                target_pos=b_target_pos,
                target_tag=b_target_tag
            )
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc
            optimizer.step()
            # Update the learning rate
            schedular.step()

            final_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_train_loss = final_loss / len(data_loader)
        # Measure how long this epoch took.
        training_time = self.format_time(time.time() - t0_epoch)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        return avg_train_loss

    def eval_fn(self,data_loader, model, device):
        print("Starting evaluation...\n")
        t0 = time.time()
        final_loss = 0
        model.eval()
        with torch.no_grad():
            for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                b_input_ids = data['input_ids']
                b_attention_mask = data['attention_mask']
                b_token_type_ids = data['token_type_ids']
                b_target_pos = data['target_pos']
                b_target_tag = data['target_tag']

                # moving tensors to device
                b_input_ids = b_input_ids.to(device)
                b_attention_mask = b_attention_mask.to(device)
                b_token_type_ids = b_token_type_ids.to(device)
                b_target_pos = b_target_pos.to(device)
                b_target_tag = b_target_tag.to(device)

                o1, o2, loss = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    token_type_ids=b_token_type_ids,
                    target_pos=b_target_pos,
                    target_tag=b_target_tag
                )

                final_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = final_loss / len(data_loader)
        # Measure how long this epoch took.
        training_time = self.format_time(time.time() - t0)

        print("")
        print("  Average validation loss: {0:.2f}".format(avg_val_loss))
        print("  Validation epoch took: {:}".format(training_time))

        return avg_val_loss

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

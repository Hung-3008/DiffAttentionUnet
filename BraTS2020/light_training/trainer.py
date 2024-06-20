# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from monai.data import DataLoader


import argparse
from .launch import launch_dist

from monai.utils import set_determinism
from .sampler import SequentialDistributedSampler, distributed_concat
from torch.utils.tensorboard import SummaryWriter

import os
import torch
import argparse
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, env_type,
                 max_epochs,
                 batch_size,
                 device="cpu",
                 val_every=1,
                 num_gpus=1,
                 logdir="./logs/",
                 master_ip='localhost',
                 master_port=17750,
                 training_script="train.py"):
        assert env_type in ["pytorch", "ddp", "DDP"], f"not support this env_type: {env_type}"
        self.env_type = env_type
        self.val_every = val_every
        self.max_epochs = max_epochs
        self.ddp = False
        self.num_gpus = num_gpus
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.batch_size = batch_size
        self.not_call_launch = True
        self.logdir = logdir
        self.scheduler = None 
        self.model = None
        self.auto_optim = True

        torch.backends.cudnn.enabled = True
        gpu_count = torch.cuda.device_count()
        if num_gpus > gpu_count:
            print("gpu数量不符")
            os._exit(0)

        if env_type == "DDP" or env_type == "ddp":
            self.ddp = True
            self.get_dist_args()
            if not self.not_call_launch:
                launch_dist(env_type=env_type,
                            num_nodes=1,
                            gpus_per_node=num_gpus,
                            master_addr=master_ip,
                            master_port=master_port,
                            training_script=training_script,
                            )
                os._exit(1)
            self.initialize_distributed()

    def initialize_distributed(self):
        if self.env_type == 'pytorch':
            self.print_rank_0('No need to initialize')
            return
        if self.env_type == 'DDP' or "deepspeed" in self.env_type:
            if self.local_rank is not None:
                device = self.local_rank
            torch.cuda.set_device(device)
            init_method = 'env://'
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=init_method)
            self.world_size = torch.distributed.get_world_size()
            print(f"world size is {self.world_size}")

    def get_dataloader(self, dataset, shuffle=False, batch_size=1, train=True):
        if dataset is None:
            return None
        if self.env_type == 'pytorch':
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=4)
        else:
            if not train:
                sampler = SequentialDistributedSampler(dataset, batch_size=batch_size)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              sampler=sampler,
                              drop_last=False)

    def get_dist_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0, help="local_rank")
        parser.add_argument('--not_call_launch',
                            action='store_true',
                            help="not call launch!")
        ds_args = parser.parse_args()
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = ds_args.local_rank
        self.not_call_launch = ds_args.not_call_launch
        self.device = self.local_rank

    def validation_single_gpu(self, val_dataset):
        if self.ddp:
            print(f"single gpu model not support the ddp")
            exit(0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.model.to(self.device)
        val_outputs = []
        self.model.eval()
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if isinstance(batch, dict):
                batch = {
                    x: batch[x].to(self.device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
            elif isinstance(batch, list):
                batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            else:
                print("not support data type")
                exit(0)
            with torch.no_grad():
                val_out = self.validation_step(batch)
                assert val_out is not None 
            return_list = False
            val_outputs.append(val_out)
        if isinstance(val_out, list) or isinstance(val_out, tuple):
            return_list = True
        val_outputs = torch.tensor(val_outputs)
        if not return_list:
            length = 0
            v_sum = 0.0
            for v in val_outputs:
                if not torch.isnan(v):
                    v_sum += v
                    length += 1
            if length == 0:
                v_sum = 0
            else:
                v_sum = v_sum / length             
        else:
            num_val = len(val_outputs[0])
            length = [0.0 for i in range(num_val)]
            v_sum = [0.0 for i in range(num_val)]
            for v in val_outputs:
                for i in range(num_val):
                    if not torch.isnan(v[i]):
                        v_sum[i] += v[i]
                        length[i] += 1
            for i in range(num_val):
                if length[i] == 0:
                    v_sum[i] = 0
                else:
                    v_sum[i] = v_sum[i] / length[i]
        return v_sum, val_outputs

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_mean_dice': self.best_mean_dice
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_mean_dice = checkpoint['best_mean_dice']
            print(f"Checkpoint loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")

    def train(self,
              train_dataset,
              optimizer=None,
              model=None,
              val_dataset=None,
              scheduler=None):
        if optimizer:
            self.optimizer = optimizer
        if model:
            self.model = model
        if scheduler:
            self.scheduler = scheduler

        set_determinism(1234 + self.local_rank)
        if self.model is not None:
            print(f"check model parameter: {next(self.model.parameters()).sum()}")
            para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
            if self.local_rank == 0:
                print(f"model parameters is {para * 4 / 1000 / 1000}M ")
        self.global_step = 0
        if self.env_type == "pytorch":
            if self.model is not None:
                self.model.to(self.device)
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)

        elif self.ddp:
            if self.local_rank == 0:
                os.makedirs(self.logdir, exist_ok=True)
                self.writer = SummaryWriter(self.logdir)
            else:
                self.writer = None
            if self.model is not None:
                self.model.cuda(self.local_rank)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                        device_ids=[self.local_rank],
                                                                        output_device=self.local_rank,
                                                                        find_unused_parameters=True)
        else:
            print("not support env_type")
            exit(0)

        train_loader = self.get_dataloader(train_dataset, shuffle=True, batch_size=self.batch_size)
        if val_dataset is not None:
            val_loader = self.get_dataloader(val_dataset, shuffle=False, batch_size=1, train=False)
        else:
            val_loader = None 

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            self.train_epoch(train_loader, epoch)
            
            val_outputs = []
            if (epoch+1) % self.val_every == 0 and val_loader is not None:
                if self.model is not None:
                    self.model.eval()
                if self.ddp:
                    torch.distributed.barrier()
                for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    else:
                        print("not support data type")
                        exit(0)
                    with torch.no_grad():
                        val_out = self.validation_step(batch)
                        assert val_out is not None 
                    return_list = False
                    val_outputs.append(val_out)
                    if isinstance(val_out, list) or isinstance(val_out, tuple):
                        return_list = True

                if self.ddp:
                    val_outputs = torch.tensor(val_outputs).cuda(self.local_rank)
                    torch.distributed.barrier()
                    val_outputs = distributed_concat(val_outputs, num_total_examples=len(val_loader.sampler.dataset))
                else:
                    val_outputs = torch.tensor(val_outputs)

                if self.local_rank == 0:
                    if not return_list:
                        length = 0
                        v_sum = 0.0
                        for v in val_outputs:
                            if not torch.isnan(v):
                                v_sum += v
                                length += 1
                        if length == 0:
                            v_sum = 0
                        else:
                            v_sum = v_sum / length 
                        self.validation_end(mean_val_outputs=v_sum)
                    else:
                        num_val = len(val_outputs[0])
                        length = [0.0 for i in range(num_val)]
                        v_sum = [0.0 for i in range(num_val)]
                        for v in val_outputs:
                            for i in range(num_val):
                                if not torch.isnan(v[i]):
                                    v_sum[i] += v[i]
                                    length[i] += 1
                        for i in range(num_val):
                            if length[i] == 0:
                                v_sum[i] = 0
                            else:
                                v_sum[i] = v_sum[i] / length[i]
                        self.validation_end(mean_val_outputs=v_sum)

            if self.scheduler is not None:
                self.scheduler.step()
            if self.model is not None:
                self.model.train()

    def train_epoch(self, loader, epoch):
        if self.model is not None:
            self.model.train()
        if self.local_rank == 0:
            with tqdm(total=len(loader)) as t:
                for idx, batch in enumerate(loader):
                    self.global_step += 1
                    t.set_description('Epoch %i' % epoch)
                    if isinstance(batch, dict):
                        batch = {
                            x: batch[x].contiguous().to(self.device)
                            for x in batch if isinstance(batch[x], torch.Tensor)
                        }
                    elif isinstance(batch, list):
                        batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                    elif isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    else:
                        print("not support data type")
                        exit(0)
                    if self.model is not None:
                        for param in self.model.parameters(): param.grad = None
                    loss = self.training_step(batch)
                    if self.auto_optim:
                        loss.backward()
                        self.optimizer.step()
                        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        t.set_postfix(loss=loss.item(), lr=lr)
                    t.update(1)
        else:
            for idx, batch in enumerate(loader):
                self.global_step += 1
                if isinstance(batch, dict):
                    batch = {
                        x: batch[x].contiguous().to(self.device)
                        for x in batch if isinstance(batch[x], torch.Tensor)
                    }
                elif isinstance(batch, list):
                    batch = [x.to(self.device) for x in batch if isinstance(x, torch.Tensor)]
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                else:
                    print("not support data type")
                    exit(0)
                for param in self.model.parameters(): param.grad = None
                loss = self.training_step(batch)
                if self.auto_optim:
                    loss.backward()
                    self.optimizer.step()
            for param in self.model.parameters(): param.grad = None

    def training_step(self, batch):
        raise NotImplementedError
    
    def validation_step(self, batch):
        raise NotImplementedError

    def validation_end(self, mean_val_outputs):
        pass

    def log(self, k, v, step):
        if self.env_type == "pytorch":
            self.writer.add_scalar(k, scalar_value=v, global_step=step)
        else:
            if self.local_rank == 0:
                self.writer.add_scalar(k, scalar_value=v, global_step=step)

    def load_state_dict(self, weight_path, strict=True):
        sd = torch.load(weight_path, map_location="cpu")
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        self.model.load_state_dict(new_sd, strict=strict)
        print(f"model parameters are loaded successfully.")

# Add these to your actual training and validation logic

# Example usage
# model = YourModel()  # Define your model
# optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
# scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=30, max_epochs=100)

# trainer = Trainer(env_type="pytorch", max_epochs=100, batch_size=32, device="cuda", num_gpus=1)
# trainer.model = model
# trainer.optimizer = optimizer
# trainer.scheduler = scheduler
# trainer.load_checkpoint("path/to/checkpoint.pt")

# train_dataset = YourTrainDataset()
# val_dataset = YourValDataset()
# trainer.train(train_dataset=train_dataset, val_dataset=val_dataset)

# For saving a checkpoint
# trainer.save_checkpoint("path/to/checkpoint.pt")



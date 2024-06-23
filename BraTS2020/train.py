import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last, save_model
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os

data_dir = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
logdir = "/kaggle/working/DiffAttentionUnet/BraTS2020/logs"

model_save_path = os.path.join(logdir, "model")

#env = "DDP" # or env = "pytorch" if you only have one gpu.
env = "pytorch"

max_epoch = 300
batch_size = 1
val_every = 10
num_gpus = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

number_modality = 4
number_targets = 3 ## WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])
        # return 4 layers <=> 4 embeddings
        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 128, 128, 128), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[128, 128, 128], sw_batch_size=1, overlap=0.25)
        self.model = DiffUNet()
        
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=30, max_epochs=max_epochs)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
    
    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_mean_dice = checkpoint['best_mean_dice']
            print(f"Checkpoint loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")

    def training_step(self, batch):
        image, label = self.get_input(batch)
        image, label = image.to(device), label.to(device)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
       
        label = label.float()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        image, label = image.to(device), label.to(device)
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        o = output[:, 1]
        t = target[:, 1] # ce
        wt = dice(o, t)
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        # active
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)
        
        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)
        self.log("mean_dice", (wt+tc+et)/3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")
            
            save_model(os.path.join(model_save_path, f"best_model_checkpoint_{mean_dice:.4f}.pt"), 
                        self.epoch, self.global_step, self.model, self.optimizer, self.scheduler, self.best_mean_dice)
            

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")
        save_model(os.path.join(model_save_path, f"final_model_checkpoint_{mean_dice:.4f}.pt"), 
                    self.epoch, self.global_step, self.model, self.optimizer, self.scheduler, self.best_mean_dice)

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")
    
    def resume_checkpoint(self, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        print(f"Resume from {checkpoint_path}")

if __name__ == "__main__":

    # Predefine default values
    default_batch_size = 1
    default_max_epoch = 100
    default_num_gpus = 2
    default_val_every = 10
    default_logdir = "/kaggle/working/DiffAttentionUnet/BraTS2020/logs"
    default_data_dir = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    default_env = "DDP"

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--max_epoch", type=int, default=default_max_epoch)
    parser.add_argument("--num_gpus", type=int, default=default_num_gpus)
    parser.add_argument("--val_every", type=int, default=default_val_every)
    parser.add_argument("--logdir", type=str, default=default_logdir)
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--env", type=str, default=default_env)

    args = parser.parse_args()

    # Update variables with the values from the command line
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    num_gpus = args.num_gpus
    val_every = args.val_every
    logdir = args.logdir
    data_dir = args.data_dir
    env = args.env

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)

    trainer = BraTSTrainer(env_type=env,
                                max_epochs=max_epoch,
                                batch_size=batch_size,
                                device=device,
                                logdir=logdir,
                                val_every=val_every,
                                num_gpus=num_gpus,
                                master_port=17751,
                                training_script=__file__)

    if args.resume:
        trainer.resume_checkpoint(args.checkpoint_dir)
        print(f"Resume from {args.checkpoint_dir}")

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


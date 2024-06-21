
import os 
import glob 
import torch 

def delete_last_model(model_dir, symbol):

    last_model = glob.glob(f"{model_dir}/{symbol}*.pt")
    if len(last_model) != 0:
        os.remove(last_model[0])


def save_new_model_and_delete_last(model, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    if delete_last_model is not None:
        delete_last_model(save_dir, delete_symbol)
    
    torch.save(model.state_dict(), save_path)

    print(f"model is saved in {save_path}")

def save_model(save_path, epoch, global_step, model, optimizer, scheduler, best_mean_dice):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_mean_dice': best_mean_dice
        }
    torch.save(checkpoint, save_path)
    print(f"model is saved in {save_path} from epoch {epoch} and global step {global_step}")

import torch
import re
import os

# Setup checkpointing
def save_checkpoint(model, optimizer, scheduler, epoch, filepath="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filepath, model, optimizer, scheduler):

    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch

# Function to find the most recent checkpoint
def get_latest_checkpoint(checkpoint_dir):
    # Get all files in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch')]
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        return None

    # Extract the epoch number from the filenames using a regular expression
    checkpoint_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(0)), reverse=True)
    
    # Get the latest checkpoint file
    latest_checkpoint = checkpoint_files[0]
    return os.path.join(checkpoint_dir, latest_checkpoint)
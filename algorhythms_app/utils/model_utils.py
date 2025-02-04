import torch
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def save_combined_model(music_gan, base_path):
    save_model(music_gan.generator, f'{base_path}_generator.pth')
    save_model(music_gan.discriminator, f'{base_path}_discriminator.pth')
    save_model(music_gan.lstm, f'{base_path}_lstm.pth')

def load_combined_model(music_gan, base_path):
    music_gan.generator = load_model(music_gan.generator, f'{base_path}_generator.pth')
    music_gan.discriminator = load_model(music_gan.discriminator, f'{base_path}_discriminator.pth')
    music_gan.lstm = load_model(music_gan.lstm, f'{base_path}_lstm.pth')
    return music_gan

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'] 
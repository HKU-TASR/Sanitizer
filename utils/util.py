import os
import torch


def save_model(model, model_name, dir_path='model'):
    # Check if file already exists.
    # If directory does not exist, create it.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    model_filepath = os.path.join(dir_path, f"{model_name}.pth")

    if os.path.exists(model_filepath):
        print(f"Warning: '{model_filepath}' already exists and will be overwritten.")

    # Save PyTorch model.
    torch.save(model.state_dict(), model_filepath)
    print(f'Model saved to {model_filepath}')


def load_model(model, model_name, dir_path='model'):
    model_filepath = os.path.join(dir_path, f"{model_name}.pth")

    # Check if file exists.
    if not os.path.exists(model_filepath):
        print(f"Error: '{model_filepath}' doesn't exist.")
        return model

    # Load PyTorch model.
    model.load_state_dict(torch.load(model_filepath))

    print(f'Model loaded from {model_filepath}')
    return model



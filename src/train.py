import argparse
import keras
from keras.models import load_model
from models import define_discriminator, define_generator, define_gan
from utils import generate_real_samples, generate_fake_samples, summarize_performance
from data_preparation import load_images, preprocess_data
import numpy as np
import os
import datetime

def save_model(
        d_model: keras.Model, 
        g_model: keras.Model,
        gan_model: keras.Model, 
        epoch: int, 
        d_loss1: float, 
        d_loss2: float, 
        g_loss: float, 
        save_dir: str = 'models') -> None:
    """
    Saves the models (discriminator, generator, and GAN) with a filename containing the current date and loss scores.
    
    :param d_model: Discriminator model
    :param g_model: Generator model
    :param gan_model: GAN model
    :param epoch: Current training epoch
    :param d_loss1: Discriminator loss on real samples
    :param d_loss2: Discriminator loss on fake samples
    :param g_loss: Generator loss
    :param save_dir: Directory where the models should be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"model_epoch{epoch+1}_d1{d_loss1:.3f}_d2{d_loss2:.3f}_g{g_loss:.3f}_{date_str}"
    
    d_model.save(os.path.join(save_dir, f'discriminator_{model_name}.h5'))
    g_model.save(os.path.join(save_dir, f'generator_{model_name}.h5'))
    gan_model.save(os.path.join(save_dir, f'gan_{model_name}.h5'))

    print(f"Models saved as: {model_name}")


def load_saved_model(model_path: str) -> keras.Model:
    """
    Load a model from the specified path.
    
    :param model_path: Path to the saved model
    :return: Loaded Keras model
    """
    return load_model(model_path)


def train(d_model: keras.Model, g_model: keras.Model, gan_model: keras.Model, dataset: tuple, n_epochs: int = 100, n_batch: int = 1) -> None:
    """
    Train the GAN model using the given dataset and model.
    
    :param d_model: Discriminator model
    :param g_model: Generator model
    :param gan_model: GAN model
    :param dataset: A tuple containing the source and target images (trainA, trainB)
    :param n_epochs: Number of training epochs
    :param n_batch: Batch size for training
    """
    trainA, trainB = dataset
    n_patch = d_model.output_shape[1]
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real, return_dict=False)
        if isinstance(d_loss1, list): d_loss1 = d_loss1[0]

        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake, return_dict=False)
        if isinstance(d_loss2, list): d_loss2 = d_loss2[0]

        g_loss_all = gan_model.train_on_batch(X_realA, [y_real, X_realB], return_dict=False)
        g_loss = g_loss_all[0] if isinstance(g_loss_all, list) else g_loss_all

        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        # Save models periodically
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
            save_model(d_model, g_model, gan_model, i, d_loss1, d_loss2, g_loss)

def main(args) -> None:
    """
    Main function to handle argument parsing, model initialization, and training.
    
    :param args: Parsed command-line arguments
    """
    # Load data
    [src_images, tar_images] = load_images(args.data_path)
    dataset = preprocess_data([src_images, tar_images])
    image_shape = src_images.shape[1:]

    # Define models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    # Load previously saved models if resuming training
    if args.resume:
        print("Loading previously saved models...")
        d_model = load_saved_model(args.d_model_path)
        g_model = load_saved_model(args.g_model_path)
        gan_model = load_saved_model(args.gan_model_path)

    # Train the models
    train(d_model, g_model, gan_model, dataset, n_epochs=args.epochs, n_batch=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Path to training dataset directory", default="data/maps_dataset/maps/train")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--resume', action='store_true', help="Resume training from saved models")
    parser.add_argument('--d_model_path', type=str, help="Path to saved discriminator model", default="")
    parser.add_argument('--g_model_path', type=str, help="Path to saved generator model", default="")
    parser.add_argument('--gan_model_path', type=str, help="Path to saved GAN model", default="")
    args = parser.parse_args()

    main(args)

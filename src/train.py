import argparse
from models import define_discriminator, define_generator, define_gan
from utils import generate_real_samples, generate_fake_samples, summarize_performance
from data_preparation import load_images, preprocess_data
import numpy as np

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    trainA, trainB = dataset
    n_patch = d_model.output_shape[1]
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to training dataset directory")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    args = parser.parse_args()

    # Load data
    [src_images, tar_images] = load_images(args.data_path)
    dataset = preprocess_data([src_images, tar_images])
    image_shape = src_images.shape[1:]

    # Define models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    # Train
    train(d_model, g_model, gan_model, dataset, n_epochs=args.epochs, n_batch=args.batch_size)

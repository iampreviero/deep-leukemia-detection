from model.autoencoder import AutoEncoder
from utils.load_data import load_images
from utils.train_model import training
from utils.visualise_outputs import visualise_reconstruction_loss, plot_training_loss

# Define parameters
directory_dict = {
    'DATA_DIR': '/Users/alessandropreviero/Downloads',
    'MODEL_DIR': '/Users/alessandropreviero/PycharmProjects/deep-leukemia-detection/src/saved/ae.pt',
    'RECON_DIR': "reconstructions.png",
    'TRAIN_LOSS_DIR': "training_loss.png"
}

max_epochs = 5
hyperparameters = {'lr': 1e-3, 'l1': 1e-3}
train_loader, validation_loader = load_images(directory_dict['DATA_DIR'], train_split=0.5, batch_size=256)
sparse_ae = AutoEncoder()


# Run example
def run_model(model, train, validation, epochs, hyper_params, directories):

    model_output, losses_vector = training(model, train, validation, epochs, hyper_params, directories, to_print=2)
    visualise_reconstruction_loss(model_output, directories['RECON_DIR'], max_epochs)
    plot_training_loss(losses_vector[0], directories['TRAIN_LOSS_DIR'])


if __name__ == "__main__":
    run_model(sparse_ae,
              train_loader,
              validation_loader,
              max_epochs,
              hyperparameters,
              directory_dict
              )

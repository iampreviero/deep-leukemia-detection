import matplotlib.pyplot as plt


def visualise_reconstruction_loss(model_output, filepath, max_epoch):
    fig = plt.figure(figsize=(9, 2))
    imgs = model_output[max_epoch - 1][1].detach().numpy()
    recon = model_output[max_epoch - 1][2].detach().numpy()

    for i, item in enumerate(imgs):
        if i >= 9:
            break
        plt.subplot(2, 9, i + 1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9:
            break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0])

    if filepath:
        fig.savefig(filepath)


def plot_training_loss(loss_vals, loss_directory):
    iters = range(len(loss_vals))
    fig = plt.figure()
    plt.plot(iters, loss_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    if loss_directory:
        fig.savefig(loss_directory)

import jax
import jax.numpy as jnp
import torch
import torchvision
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PX = 1/plt.rcParams['figure.dpi']



# WARNING: this code is from QSSM project and won't be updated 
def create_mnist_classification_dataset(bsz=128, root="./data", version="sequential"):
    print("[*] Generating MNIST Classification Dataset...")
    assert version in ["sequential", "row"], "Invalid version for MNIST dataset"

    # Constants
    if version == "sequential":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    elif version == "row":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 28, 10, 28
    tf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ]

    tf.append(transforms.Lambda(lambda x: x.view(SEQ_LENGTH, IN_DIM)))
    tf = transforms.Compose(tf)

    train = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=tf
    )

    # split the dataset into train and val 
    train, val = torch.utils.data.random_split(train, [50000, 10000])

    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        labels = np.array(transposed_data[1])
        images = np.array(transposed_data[0])

        return images, labels       


    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, collate_fn=custom_collate_fn, drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, collate_fn=custom_collate_fn, drop_last=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def g(x):
    return jnp.where(x > 0, x+0.5, jax.nn.sigmoid(x))

def plot_dynamics(model, params, batch_inputs, batch_labels, dataset_version='sequential',
                    id_sample=0, nb_inputs_to_plot=5, nb_components_to_plot=5, model_type='srn', variable_to_plot='h', zoom=True):
    
    model_LUT = {
        'lstm': {'h': 0, 'c': 1},
        'gru': {'h': 0, 'z': 1, 'r': 2},
        'mgu': {'h': 0, 'f': 1},
        'mingru': {'h': 0, 'z': 1, 'z_preact': 2},
        'mingru_heinsen': {'h': 0, 'z': 1, 'z_preact': 1, 'h_tilde': 2, 'h_tilde_preact': 2},
    }
    
    # check if params has the key 'params' or not
    if 'params' not in params:
        params = {'params': params}

    n_inputs_to_simulate = 5 if id_sample < 5 else id_sample+1

    if model_type == 'srn':
        state_hist, out_hist = model.apply(params, batch_inputs[:n_inputs_to_simulate])
        n_layers = len(state_hist)
    elif model_type in ['lstm', 'gru', 'mgu', 'mingru', 'mingru_heinsen']:
        state_hist, out_hist = model.apply(params, batch_inputs[:n_inputs_to_simulate])
        n_layers = len(state_hist)
        print(len(state_hist))
        print(len(state_hist[0]))
        print(state_hist[0][0].shape)
        if variable_to_plot == 'all':
            n_cols = len(model_LUT[model_type])
        else: 
            id_component = model_LUT[model_type][variable_to_plot]
            n_cols = 1

    t = jnp.arange(784) if dataset_version == "sequential" else jnp.arange(28)

    if dataset_version == "sequential":
        inputs_to_plot = batch_inputs[id_sample, :, 0]
        labels_input = f'id_sample={id_sample}'
    else:
        inputs_to_plot = batch_inputs[id_sample, :, 14-nb_inputs_to_plot//2:14+(nb_inputs_to_plot+1)//2]
        labels_input = [f'input_{i}' for i in range(nb_inputs_to_plot)]

    fig, axs = plt.subplots(len(state_hist)+2, n_cols, figsize=(((1200+(n_cols-1)*600)*PX, (600+200*n_layers)*PX)))

    legend_ncol = 5 if nb_components_to_plot > 5 else nb_components_to_plot
    if n_cols > 1: 
        for n in range(n_cols):
            axs[0,n].plot(t, inputs_to_plot, label=labels_input)
            axs[0,n].set_title(f'Input Sequence = {batch_labels[id_sample]}')
            axs[0,n].legend(ncol=14, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            axs[0,n].grid(True)
            for i in range(n_layers):
                variable_name = list(model_LUT[model_type].keys())[n]
                id_variable = model_LUT[model_type][variable_name]
                data_to_plot = state_hist[i][id_variable][id_sample, :, :nb_components_to_plot]
                if model_type == 'mingru_heinsen':
                    if variable_name == 'z': 
                        data_to_plot = jax.nn.sigmoid(data_to_plot)
                    elif variable_name == 'h_tilde':
                        data_to_plot = g(data_to_plot)
                axs[i+1,n].plot(t, data_to_plot, label=[f'state_{i}' for i in range(nb_components_to_plot)])
                axs[i+1,n].set_title(f'{variable_name} - Layer {i}')
                if nb_components_to_plot < 10:
                    axs[i+1,n].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))
                # axs[i+1].set_ylim(0, 1)
                # axs[i+1].set_yticks(np.arange(0, 1, 0.1))
                axs[i+1,n].grid(True)
                if zoom: 
                    if variable_to_plot in ['z', 'r', 'f']:
                        axs[i+1,n].set_ylim(0, 1)
                        axs[i+1,n].set_yticks(np.arange(0, 1, 0.1))
            axs[-1,n].plot(t, out_hist[id_sample, :, :], label=[f'out_{i}' for i in range(10)])
            axs[-1,n].set_title('Output History')
            axs[-1,n].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            axs[-1,n].grid(True)
    else:
        axs[0].plot(t, inputs_to_plot, label=labels_input)
        axs[0].set_title(f'Input Sequence = {batch_labels[id_sample]}')
        axs[0].legend(ncol=14, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        axs[0].grid(True)
        for i in range(n_layers):
            data_to_plot = state_hist[i][id_component][id_sample, :, :nb_components_to_plot]
            if model_type == 'mingru_heinsen':
                if variable_to_plot == 'z': 
                    data_to_plot = jax.nn.sigmoid(data_to_plot)
                elif variable_to_plot == 'h_tilde':
                    data_to_plot = g(data_to_plot)

            axs[i+1].plot(t, data_to_plot, label=[f'state_{i}' for i in range(nb_components_to_plot)])
            axs[i+1].set_title(f'{variable_to_plot} - Layer {i}')
            if nb_components_to_plot < 10:
                axs[i+1].legend(ncol=legend_ncol, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            # axs[i+1].set_ylim(0, 1)
            # axs[i+1].set_yticks(np.arange(0, 1, 0.1))
            axs[i+1].grid(True)
            if zoom: 
                if variable_to_plot in ['z', 'r', 'f']:
                    axs[i+1].set_ylim(0, 1)
                    axs[i+1].set_yticks(np.arange(0, 1, 0.1))

        axs[-1].plot(t, out_hist[id_sample, :, :], label=[f'out_{i}' for i in range(10)])
        axs[-1].set_title('Output History')
        axs[-1].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        axs[-1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_loss_and_acc(train_losses, val_losses, train_accuracies, val_accuracies):
    '''
    Plot the loss and accuracy of the model during training and validation
    # ax[0] entire loss - ax[0] entire accuracy
    # ax_in0 zoomed loss - ax_in0 zoomed accuracy
    '''

    t = np.arange(len(val_losses))
    offset = 20
    fig, ax = plt.subplots(1, 2, figsize=(1200*PX, 600*PX))

    ax[0].plot(t, train_losses, label='train')
    ax[0].plot(t, val_losses, label='val')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].set_yticks(np.arange(0, 4.0, 0.3))
    ax[0].set_xticks(np.arange(0, len(val_losses), 5))
    ax[0].grid()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Value')

    ax_in0 = inset_axes(ax[0], width="60%", height="60%", loc="upper right")
    ax_in0.plot(t[offset:], train_losses[offset:], label='train')
    ax_in0.plot(t[offset:], val_losses[offset:], label='val')
    ax_in0.legend()
    # ax_in0.set_yticks(np.arange(0, np.max(train_losses[offset:]).round(2), 0.1))
    ax_in0.set_xticks(np.arange(offset, len(val_losses), 5))
    # set ylim
    y_min = min(np.min(train_losses[offset:]), np.min(val_losses[offset:])) * 0.9
    y_max = max(np.max(train_losses[offset:]), np.max(val_losses[offset:])) * 1.1
    ax_in0.set_ylim(y_min, y_max)
    ax_in0.set_yticks(np.arange(round(y_min, 2), round(y_max, 2), 0.025))
    ax_in0.grid()


    ax[1].plot(t, train_accuracies, label='train')
    ax[1].plot(t, val_accuracies, label='val')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].set_yticks(np.arange(0, 1, 0.1))
    ax[1].set_xticks(np.arange(0, len(val_accuracies), 5))
    ax[1].grid()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')

    ax_in1 = inset_axes(ax[1], width="60%", height="50%", loc="lower right", borderpad=2)
    ax_in1.plot(t[offset:], train_accuracies[offset:], label='train')
    ax_in1.plot(t[offset:], val_accuracies[offset:], label='val')
    ax_in1.legend()
    ax_in1.set_xticks(np.arange(offset, len(val_accuracies), 5))
    # set ylim
    y_min = min(np.min(val_accuracies[offset:]), np.min(train_accuracies[offset:])) * 0.999
    y_max = max(np.max(val_accuracies[offset:]), np.max(train_accuracies[offset:])) * 1.001
    ax_in1.set_ylim(y_min, y_max)
    ax_in1.set_yticks(np.arange(round(y_min, 2), round(y_max, 2), 0.0025))
    ax_in1.grid()
            
    plt.tight_layout()
    plt.show()
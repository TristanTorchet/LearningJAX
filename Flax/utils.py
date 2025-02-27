import jax
import jax.numpy as jnp
import torch
import torchvision
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
px = 1/plt.rcParams['figure.dpi']



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



def plot_dynamics(model, params, batch_inputs, batch_labels, dataset_version='sequential',
                    id_sample=0, nb_inputs_to_plot=5, nb_components_to_plot=5, model_type='srn', variable_to_plot='h', zoom=True):
    
    model_LUT = {
        'lstm': {'h': 0, 'c': 1},
        'gru': {'h': 0, 'z': 1, 'r': 2},
        'mgu': {'h': 0, 'f': 1},
        'mingru': {'h': 0, 'z': 1, 'z_preact': 2},
    }
    
    # check if params has the key 'params' or not
    if 'params' not in params:
        params = {'params': params}

    n_inputs_to_simulate = 5 if id_sample < 5 else id_sample+1

    if model_type == 'srn':
        state_hist, out_hist = model.apply(params, batch_inputs[:n_inputs_to_simulate])
        n_layers = len(state_hist)
    elif model_type in ['lstm', 'gru', 'mgu', 'mingru']:
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

    fig, axs = plt.subplots(len(state_hist)+2, n_cols, figsize=(((1200+n_cols*600)*px, (600+200*n_layers)*px)))
    


    legend_ncol = 5 if nb_components_to_plot > 5 else nb_components_to_plot
    if n_cols > 1: 
        for n in range(n_cols):
            axs[0,n].plot(t, inputs_to_plot, label=labels_input)
            axs[0,n].set_title(f'Input Sequence = {batch_labels[id_sample]}')
            axs[0,n].legend(ncol=14, loc='upper center', bbox_to_anchor=(0.5, -0.1))
            axs[0,n].grid(True)
            for i in range(n_layers):
                axs[i+1,n].plot(t, state_hist[i][n][id_sample, :, :nb_components_to_plot], label=[f'state_{i}' for i in range(nb_components_to_plot)])
                axs[i+1,n].set_title(f'{list(model_LUT[model_type].keys())[n]} - Layer {i}')
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
            axs[i+1].plot(t, state_hist[i][id_component][id_sample, :, :nb_components_to_plot], label=[f'state_{i}' for i in range(nb_components_to_plot)])
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
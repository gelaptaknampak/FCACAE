#
# Federated Clustering via Adaptive Resonance Theory (ART)-based Clustering (FCAC)
#

import time
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from utility_fl import *
import sys
import os

from fcac import FCAC

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Evaluation metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

import copy
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler


# data_list = ["hillvalley", "ozone", "bioresponse", "phoneme", "texture", "optdigits", "pendigits", "mozilla4", "magic", "letter", "skin"]
data_name = "fmnist"



# experimental settings
n_trial = 2
niid = True  # True:non-iid, False:iid for federated learning
epsilon = -1  # privacy budget for \epsilon-differential privacy (-1: no noise)
max_iters = 1

# data split setting
if niid == True:
    balance = False  # Number of data points among clients. True:same, False:different
    partition = "dir"  # If set as "pat", then a train_dataset becomes pathological non-i.i.d.
    alpha = 0.5  # for Dirichlet distribution in separate_data()
else:
    balance = True  # Number of data points among clients. True:same, False:different
    partition = "pat"  # "dir", "pat"
    alpha = None  # for Dirichlet distribution in separate_data()

# for results
all_training_time = []
all_n_nodes = []
all_n_clusters = []
all_ari = []
all_ami = []
all_nmi = []
all_v_thres = []

def local_train_ae(global_model, client_data, device, epochs=1):

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    local_model = copy.deepcopy(global_model)

    optimizer = torch.optim.Adam(local_model.parameters(), lr=5e-4)

    loss_fn = torch.nn.MSELoss()

    tensor_data = torch.FloatTensor(client_data)

    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=64,
        shuffle=True,
        drop_last=True
    )

    local_model.train()

    for epoch in range(epochs):

        total_loss = 0

        for (batch,) in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            recon = local_model(batch)

            loss = loss_fn(
                recon,
                batch.view(-1, 784)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)

        if epoch == epochs - 1:
            print(
                f"   Final Loss: {avg_loss:.6f}"
            )

    return local_model.state_dict()

def fedavg(local_weights, train_data):

    avg_weights = OrderedDict()

    total_data = sum(
        len(client_data)
        for client_data in train_data
    )

    for key in local_weights[0].keys():

        avg_weights[key] = sum(
            (
                len(train_data[i]) / total_data
            ) * local_weights[i][key]
            for i in range(len(local_weights))
        )

    return avg_weights


print(data_name)
for i_trial in tqdm(range(n_trial), total=n_trial, desc='Trial for Averaging'):  # for averaging

    # load dataset
    DATA, TARGET, n_clients, n_classes = set_dataset(data_name, niid, i_trial)

    DATA = DATA.astype(np.float32) / 255.0

    # training data = test data
    train_DATA = DATA
    train_TARGET = TARGET
    test_data = DATA
    test_target = TARGET
    test_dataset = {"full_data": test_data, "true_label": test_target}

    # prepare for federated learning
    train_data, train_target, statistic = separate_data((train_DATA, train_TARGET), n_clients, n_classes, alpha, niid, balance, partition)

    # ==========================================
    # FEDERATED AUTOENCODER + FEDAVG
    # ==========================================

    import torch
    from pytorchAE.models.AE import Network

    class DummyArgs:
        def __init__(self):
            self.embedding_size = 64
            self.input_dim = 784
            self.cuda = torch.cuda.is_available()

    args_ae = DummyArgs()

    device = torch.device("cuda" if args_ae.cuda else "cpu")

    # Global AE model
    global_ae = Network(args_ae).to(device)

    # Federated training settings
    federated_rounds = 10
    local_epochs = 10

    print("Training Federated Autoencoder...")

    for rnd in range(federated_rounds):

        print(f"\nFederated Round {rnd+1}")

        local_weights = []

        for client_id in range(n_clients):

            print(f" Client {client_id}")

            client_data = train_data[client_id]

            weights = local_train_ae(
                global_ae,
                client_data,
                device,
                epochs=local_epochs
            )

            local_weights.append(weights)

        # FedAvg aggregation
        global_weights = fedavg(local_weights, train_data)

        global_ae.load_state_dict(global_weights)

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================

    global_ae.eval()

    embedded_train_data = []

    with torch.no_grad():

        for client_data in train_data:

            tensor_data = torch.FloatTensor(client_data).to(device)

            z = global_ae.encode(
                tensor_data.view(-1, 784)
            )

            embedded_train_data.append(
                z.cpu().numpy()
            )

    # Test embedding
    with torch.no_grad():

        tensor_test_data = torch.FloatTensor(test_data).to(device)

        embedded_test_data = global_ae.encode(
            tensor_test_data.view(-1, 784)
        ).cpu().numpy()

    # # ==========================================
    # # FEATURE EXTRACTION & NORMALIZATION
    # # ==========================================

    # global_ae.eval()
    # embedded_train_data = []

    # with torch.no_grad():
    #     for client_data in train_data:
    #         tensor_data = torch.FloatTensor(client_data).to(device)
    #         z = global_ae.encode(tensor_data.view(-1, 784))
    #         embedded_train_data.append(z.cpu().numpy())

    # # --- TAMBAHAN BARU: NORMALISASI RUANG LATEN ---
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    
    # # Gabungkan sementara untuk mencari nilai min-max global, lalu pisahkan lagi
    # concatenated_train = np.concatenate(embedded_train_data, axis=0)
    # scaler.fit(concatenated_train)
    
    # # Terapkan normalisasi ke masing-masing klien
    # normalized_train_data = [scaler.transform(client_z) for client_z in embedded_train_data]

    # # Test embedding (Jangan lupa di-scale juga!)
    # with torch.no_grad():
    #     tensor_test_data = torch.FloatTensor(test_data).to(device)
    #     embedded_test_data = global_ae.encode(tensor_test_data.view(-1, 784)).cpu().numpy()
    #     embedded_test_data = scaler.transform(embedded_test_data)
        
    # # Add Laplacian noise ke data yang SUDAH dinormalisasi
    # if epsilon == -1:  
    #     noised_train_data = normalized_train_data
    # else:
    #     noised_train_data = [add_laplace_noise(z, epsilon, seed=i_trial) for z in normalized_train_data]
        
    # Add Laplacian noise to a train_dataset
    if epsilon == -1:  # no noise setting
        noised_train_data = embedded_train_data
    else:
        noised_train_data = [add_laplace_noise(z, epsilon, seed=i_trial) for z in embedded_train_data]

    # training
    fcac = FCAC(n_clients_=n_clients, iter_server_=max_iters)
    start = time.time()
    params_server_fcac, params_clients_fcac = fcac.fit(noised_train_data)
    all_training_time.append(time.time() - start)

    # test
    server_assignments = params_server_fcac.predict(embedded_test_data)

    # evaluation
    all_ari.append(adjusted_rand_score(test_dataset['true_label'], server_assignments))
    all_ami.append(adjusted_mutual_info_score(test_dataset['true_label'], server_assignments))
    all_nmi.append(normalized_mutual_info_score(test_dataset['true_label'], server_assignments))
    all_n_nodes.append(params_server_fcac.G_.number_of_nodes())
    all_n_clusters.append(params_server_fcac.n_clusters_)
    all_v_thres.append(params_server_fcac.V_thres_)
    # Menghitung total ukuran memori (dalam bytes) dari seluruh data klien
    total_payload = sum(client_data.nbytes for client_data in noised_train_data)


# averaged results
print(data_name)
print('--------------- FCAC (mean result)')
print('Time:', '{:.5f}'.format(np.mean(all_training_time)), '[s]')
print(f"Payload Komunikasi: {total_payload} bytes")
print(' # of Nodes:', '{:.1f}'.format(np.mean(all_n_nodes)))
print(' # of Clusters:', '{:.1f}'.format(np.mean(all_n_clusters)))
print(' Similarity Threshold (V_thres):', '{:.5f}'.format(np.mean(all_v_thres)))
print(' ARI:', '{:.5f}'.format(np.mean(all_ari)))
print(' AMI:', '{:.5f}'.format(np.mean(all_ami)))
print(' NMI:', '{:.5f}'.format(np.mean(all_nmi)))

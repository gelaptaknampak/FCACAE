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


# data_list = ["hillvalley", "ozone", "bioresponse", "phoneme", "texture", "optdigits", "pendigits", "mozilla4", "magic", "letter", "skin"]
data_name = "fmnist"



# experimental settings
n_trial = 2
niid = True  # True:non-iid, False:iid for federated learning
epsilon = 50  # privacy budget for \epsilon-differential privacy (-1: no noise)
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
    # INTEGRASI AUTOENCODER (VERSI GITHUB ORIGINAL)
    # ==========================================
    import torch
    from pytorchAE.models.AE import Network # Import Network karena class AE di GitHub mencoba load dataset sendiri

    # 1. Setup Objek Args (Wajib karena Network(args) membutuhkannya)
    class DummyArgs:
        def __init__(self):
            self.embedding_size = 128  # Ukuran laten (bisa coba 16, 32, atau 64)
            self.input_dim = 784
            self.cuda = torch.cuda.is_available()

    args_ae = DummyArgs()
    device = torch.device("cuda" if args_ae.cuda else "cpu")

    # 2. Inisialisasi Model
    # Karena fmnist dan MNIST sama-sama 784 fitur, Network asli akan bekerja
    ae_model = Network(args_ae).to(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss() # Menggunakan MSE untuk rekonstruksi data kontinu

    # 3. Pre-training AE secara terpusat (PAKAI DATALOADER)
    from torch.utils.data import TensorDataset, DataLoader

    all_data_for_ae = np.vstack(train_data)

    tensor_all_data = torch.FloatTensor(all_data_for_ae)

    dataset_ae = TensorDataset(tensor_all_data)

    loader_ae = DataLoader(
        dataset_ae,
        batch_size=64,
        shuffle=True
    )

    ae_model.train()
    print("Training Autoencoder...")

    for epoch in range(60):

        total_loss = 0

        for (batch,) in loader_ae:

            batch = batch.to(device)

            optimizer.zero_grad()

            recon = ae_model(batch)

            loss = loss_fn(recon, batch.view(-1, 784))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader_ae)

        print(f"AE Epoch {epoch} Loss: {avg_loss:.6f}")

    # 4. Ekstraksi Fitur (Mengubah ke ruang laten)
    ae_model.eval()
    embedded_train_data = []
    with torch.no_grad():
        for client_data in train_data:
            tensor_data = torch.FloatTensor(client_data).to(device)
            # Menggunakan fungsi encode() langsung dari AE.py
            z = ae_model.encode(tensor_data.view(-1, 784))
            embedded_train_data.append(z.cpu().numpy())

    # 5. Transformasi data uji ke ruang laten
    with torch.no_grad():
        tensor_test_data = torch.FloatTensor(test_data).to(device)
        embedded_test_data = ae_model.encode(tensor_test_data.view(-1, 784)).cpu().numpy()
    # ==========================================

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
    # Menghitung total ukuran memori (dalam bytes) dari seluruh data klien
    total_payload = sum(client_data.nbytes for client_data in noised_train_data)


# averaged results
print(data_name)
print('--------------- FCAC (mean result)')
print('Time:', '{:.5f}'.format(np.mean(all_training_time)), '[s]')
print(f"Payload Komunikasi: {total_payload} bytes")
print(' # of Nodes:', '{:.1f}'.format(np.mean(all_n_nodes)))
print(' # of Clusters:', '{:.1f}'.format(np.mean(all_n_clusters)))
print(' ARI:', '{:.5f}'.format(np.mean(all_ari)))
print(' AMI:', '{:.5f}'.format(np.mean(all_ami)))
print(' NMI:', '{:.5f}'.format(np.mean(all_nmi)))

#
# Federated Clustering via Adaptive Resonance Theory (ART)-based Clustering (FCAC + PCA)
#

import time
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from utility_fl import *

from fcac import FCAC

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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

# PCA setting
pca_components = 64

# data split setting
if niid == True:
    balance = False
    partition = "dir"
    alpha = 0.5
else:
    balance = True
    partition = "pat"
    alpha = None

# for results
all_training_time = []
all_n_nodes = []
all_n_clusters = []
all_ari = []
all_ami = []
all_nmi = []

print(data_name)

for i_trial in tqdm(range(n_trial), total=n_trial, desc='Trial for Averaging'):

    # load dataset
    DATA, TARGET, n_clients, n_classes = set_dataset(data_name, niid, i_trial)

    # training data = test data
    train_DATA = DATA
    train_TARGET = TARGET

    test_data = DATA
    test_target = TARGET

    test_dataset = {
        "full_data": test_data,
        "true_label": test_target
    }

    # prepare federated learning
    train_data, train_target, statistic = separate_data(
        (train_DATA, train_TARGET),
        n_clients,
        n_classes,
        alpha,
        niid,
        balance,
        partition
    )

    # ==========================================
    # PCA DIMENSIONALITY REDUCTION
    # ==========================================

    # gabungkan semua client data untuk fit PCA global
    all_data_for_pca = np.vstack(train_data)

    # PCA
    pca = PCA(
        n_components=pca_components,
        random_state=i_trial
    )

    pca.fit(all_data_for_pca)

    # transform tiap client
    embedded_train_data = [
        pca.transform(client_data)
        for client_data in train_data
    ]

    # transform test data
    embedded_test_data = pca.transform(test_data)

    # ==========================================
    # NORMALIZATION
    # ==========================================

    scaler = MinMaxScaler()

    embedded_train_data = [
        scaler.fit_transform(client_data)
        for client_data in embedded_train_data
    ]

    embedded_test_data = scaler.fit_transform(embedded_test_data)

    # ==========================================
    # DIFFERENTIAL PRIVACY
    # ==========================================

    if epsilon == -1:
        noised_train_data = embedded_train_data
    else:
        noised_train_data = [
            add_laplace_noise(data, epsilon, seed=i_trial)
            for data in embedded_train_data
        ]

    # ==========================================
    # FCAC TRAINING
    # ==========================================

    fcac = FCAC(
        n_clients_=n_clients,
        iter_server_=max_iters
    )

    start = time.time()

    params_server_fcac, params_clients_fcac = fcac.fit(noised_train_data)

    all_training_time.append(time.time() - start)

    # ==========================================
    # TESTING
    # ==========================================

    server_assignments = params_server_fcac.predict(embedded_test_data)

    # ==========================================
    # EVALUATION
    # ==========================================

    all_ari.append(
        adjusted_rand_score(
            test_dataset['true_label'],
            server_assignments
        )
    )

    all_ami.append(
        adjusted_mutual_info_score(
            test_dataset['true_label'],
            server_assignments
        )
    )

    all_nmi.append(
        normalized_mutual_info_score(
            test_dataset['true_label'],
            server_assignments
        )
    )

    all_n_nodes.append(
        params_server_fcac.G_.number_of_nodes()
    )

    all_n_clusters.append(
        params_server_fcac.n_clusters_
    )

    # payload komunikasi
    total_payload = sum(
        client_data.nbytes
        for client_data in noised_train_data
    )


# averaged results
print(data_name)
print('--------------- FCAC + PCA (mean result)')
print('Time:', '{:.5f}'.format(np.mean(all_training_time)), '[s]')
print(f"Payload Komunikasi: {total_payload} bytes")
print(' # of Nodes:', '{:.1f}'.format(np.mean(all_n_nodes)))
print(' # of Clusters:', '{:.1f}'.format(np.mean(all_n_clusters)))
print(' ARI:', '{:.5f}'.format(np.mean(all_ari)))
print(' AMI:', '{:.5f}'.format(np.mean(all_ami)))
print(' NMI:', '{:.5f}'.format(np.mean(all_nmi)))
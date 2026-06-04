import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Mengimpor arsitektur Network bawaan dari paket pytorchAE milikmu
from pytorchAE.models.AE import Network

# 1. Fungsi Pelatihan Lokal di Klien
def local_train_ae(global_model, client_data, device, epochs=1):
    local_model = copy.deepcopy(global_model)
    # Weight decay tetap dipertahankan untuk menjaga stabilitas agregasi FedAvg
    optimizer = torch.optim.Adam(local_model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    if len(client_data) < 2:
        return local_model.state_dict(), 0.0

    tensor_data = torch.FloatTensor(client_data)
    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=64,
        shuffle=True,
        drop_last=False
    )

    local_model.train()
    avg_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = local_model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)

    return local_model.state_dict(), avg_loss

# 2. Fungsi Agregasi Server (FedAvg)
def fedavg(local_weights, local_sizes):
    avg_weights = OrderedDict()
    total_data = sum(local_sizes)
    for key in local_weights[0]:
        avg_weights[key] = sum(
            (local_sizes[i] / total_data) * local_weights[i][key] for i in range(len(local_weights))
        )
    return avg_weights

# 3. Orkestrator Utama (Menjalankan FL Loop & Ekstraksi Fitur Laten)
def run_federated_ae_and_extract(train_data, test_data, device, n_clients, federated_rounds=60, local_epochs=10, input_dim=64, embedding_size=16):
    
    # Membuat kelas internal untuk passing argument ke Network bawaan secara dinamis
    class AEArgs:
        def __init__(self):
            self.input_dim = input_dim
            self.embedding_size = embedding_size
            self.cuda = (device.type == 'cuda')

    args_ae = AEArgs()
    
    # Menginisialisasi model Network bawaan proyekmu
    global_ae = Network(args_ae).to(device)
    
    print("Training Federated Autoencoder (Bawaan)...")
    for rnd in range(federated_rounds):
        local_weights = []
        local_sizes = []
        round_total_loss = 0.0
        total_data_in_round = 0

        for client_id in range(n_clients):
            client_data = train_data[client_id]
            if len(client_data) < 2:
                continue

            weights, client_loss = local_train_ae(global_ae, client_data, device, epochs=local_epochs)
            
            local_weights.append(weights)
            local_sizes.append(len(client_data))
            round_total_loss += client_loss * len(client_data)
            total_data_in_round += len(client_data)

        # FedAvg Aggregation
        if total_data_in_round > 0:
            global_weights = fedavg(local_weights, local_sizes)
            global_ae.load_state_dict(global_weights)
            
            avg_round_loss = round_total_loss / total_data_in_round
            print(f"Federated Round {rnd+1} Total Loss ; {avg_round_loss:.6f}")

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    global_ae.eval()
    embedded_train_data = []

    with torch.no_grad():
        for client_data in train_data:
            tensor_data = torch.FloatTensor(client_data).to(device)
            # Memanggil fungsi .encode() dari model bawaan
            z = global_ae.encode(tensor_data)
            embedded_train_data.append(z.cpu().numpy())

    # Ekstraksi untuk data uji (test data) tanpa normalisasi
    with torch.no_grad():
        tensor_test_data = torch.FloatTensor(test_data).to(device)
        embedded_test_data = global_ae.encode(tensor_test_data).cpu().numpy()

    return embedded_train_data, embedded_test_data
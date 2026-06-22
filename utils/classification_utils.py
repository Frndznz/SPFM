import os
import torch
import numpy as np

from tqdm.auto import tqdm
from momentfm import MOMENTPipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

def get_moment_model(n_classes, n_channels, task="classification", device=None):
    if task == "classification":
        model_kwargs={
            "task_name": task,
            "n_channels": n_channels,
            "num_class": n_classes
        }
    elif task == "embedding":
        model_kwargs={
            "task_name": task,
            "n_channels": n_channels,
        }
    model = MOMENTPipeline.from_pretrained(
        './foundational_models/moment_large_embedding',
        model_kwargs={
            "task_name": task,
            "n_channels": n_channels,
            "num_class": n_classes
        },
    )
    model.init()
    model.to(device)
    if task == "embedding":
        model.eval()
    elif task == "classification":
        # Ensure the base encoder is frozen, and only the new classifier head requires gradients
        model.requires_grad_ = False
        model.head.requires_grad_ = True

    return model

def extract_moment_embeddings(dataloader, moment_model, device, is_test=False):
    """ Runs the data through MOMENT once and saves the feature vectors. """
    features, labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            # Handle Dicts vs Tuples
            if isinstance(batch, dict):
                x = batch['x'] if 'x' in batch else (batch['x_lf'] + batch['x_hf'])
                y = batch['y']
            else:
                x, y = batch

            x, y = x.float().to(device), y.long().to(device)

            # Test data needs slicing, Train data was already sliced during generation
            if is_test:
                x, y = prepare_moment_data(x, y, window_size=512, num_windows=2)

            # Mixed Precision Forward Pass (2x speedup)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = moment_model(x_enc=x)

            # Pool embeddings: MOMENT returns [batch, patches, d_model]
            # We mean-pool across the sequence length to get 1 vector per sample
            emb = outputs.embeddings

            features.append(emb)
            labels_list.append(y.cpu())

    return torch.cat(features, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).numpy()

def prepare_moment_data(samples, labels, window_size=512, num_windows=2):
    """
    Splits samples and replicates labels to fit MOMENT's 512 context.
    """
    N, C, L = samples.shape
    device = samples.device
    # Calculate windows
    stride = (L - window_size) // (num_windows - 1)
    windows = []
    for i in range(num_windows):
        start = i * stride
        # Adjust the last window to ensure it doesn't exceed L
        if i == num_windows - 1:
            start = L - window_size
        end = start + window_size
        windows.append(samples[:, :, start:end])

    # Concatenate windows: [N * num_windows, C, 512]
    new_samples = torch.cat(windows, dim=0)

    # Replicate labels: [N * num_windows]
    new_labels = torch.cat([labels] * num_windows, dim=0)

    return new_samples, new_labels

def train_and_eval_probe(train_loader, test_loader, n_classes, n_channels, n_epochs=50, lr=1e-3, device=None, return_probe=False):
    criterion = nn.CrossEntropyLoss()
    moment_model = get_moment_model(n_classes=n_classes, n_channels=n_channels, task="classification", device=device)
    # We only pass the parameters that require gradients (classifier head)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, moment_model.parameters()),
                            lr=lr, weight_decay=1e-4)
    moment_model.train()
    for epoch in tqdm(range(n_epochs), desc="Training classifier..."):
        for batch in train_loader:
            data, labels = batch['x'], batch['y']
            data, labels = data.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = moment_model(x_enc=data, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    moment_model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating classifier..."):
            test_data, test_labels = prepare_moment_data(batch['x'],  batch['y'])
            outputs = moment_model(x_enc=test_data.float().to(device), labels=test_labels.long().to(device))
            probs = F.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(logits.data, 1)

            all_labels.extend(test_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_Score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # One vs. rest for AUROC calculation
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auroc = float('nan')

    if return_probe:
        return {"Accuracy": acc, "F1": f1, "AUROC": auroc, "Recall": recall}, moment_model
    return {"Accuracy": acc, "F1": f1, "AUROC": auroc, "Recall": recall}

def train_and_eval_linear_probe(x_train, y_train, x_test, y_test, device=None):
    """ Trains a Scikit-Learn classifier instantly on the extracted features to do linear probing """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, multi_class='ovr', n_jobs=-1)
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    try:
        auroc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
    except ValueError:
        auroc = float('nan')

    return {"Accuracy": acc, "F1": f1, "AUROC": auroc, "Recall": recall}

def generate_classification_data(real_dataloader, model, prior, data_type, device, sampling_steps=[20, 30]):
    """Generates synthetic samples matching the train set's class distribution with the provided model.
    """
    if data_type == "ECG":
        C = 12
    elif data_type == "EEG":
        C = 2
    model.eval()

    fake_data, fake_labels = [], []
    real_data, real_labels = [], []
    for batch in tqdm(real_dataloader):
        x = batch['x'].to(device)
        real_data.append(x)
        labels = batch['y'].long().to(device)
        fake_labels.append(labels)
        real_labels.append(labels)
        x0 = prior.sample(x.shape, device)
        with torch.no_grad():
            fake_x, _ = model.sample(x0 = x0, sampling_timesteps=sampling_steps, y=labels)
        fake_data.append(fake_x)

    real_data, real_labels = prepare_moment_data(torch.cat(real_data), torch.cat(real_labels))
    syn_data, syn_labels = prepare_moment_data(torch.cat(fake_data), torch.cat(fake_labels))

    syn_dataset = torch.utils.data.TensorDataset(syn_data.cpu(), syn_labels.cpu())
    real_dataset = torch.utils.data.TensorDataset(real_data.cpu(), real_labels.cpu())

    return syn_dataset, real_dataset

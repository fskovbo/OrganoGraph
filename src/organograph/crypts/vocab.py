import numpy as np
from organograph.mesh.hks import compute_hks


def compute_vocabulary_encoding(bag_of_features, mesh):

    vocab = bag_of_features['vocab']
    scaler = bag_of_features['scaler'].item()
    sigma = bag_of_features['sigma']
    ts = bag_of_features['ts']

    hks = compute_hks(mesh, ts, coeffs=False)

    # normalise
    normalised_hks = (hks/np.mean(hks, axis=0, keepdims=True) - 1)
    normalised_hks = scaler.transform(normalised_hks)

    # convert to encoding
    dist = np.linalg.norm(normalised_hks[:, np.newaxis, :]-vocab[np.newaxis, :, :], axis=2)
    encoding = np.exp(-dist**2 / (2 * sigma**2))

    return encoding, hks, normalised_hks, ts

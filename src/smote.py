import numpy as np
from collections import Counter


class SMOTE:


    def __init__(self,
                 k_neighbours: int = 5,
                 random_state: int = 42,
                 sampling_strategy: str = "auto"):
        self.k = k_neighbours
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.rng = np.random.RandomState(random_state)

    def _euclidean_distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance from point x to all points in X."""
        return np.sqrt(np.sum((X - x) ** 2, axis=1))

    def _find_k_neighbours(self, sample: np.ndarray,
                           minority_samples: np.ndarray) -> np.ndarray:
        
        distances = self._euclidean_distances(sample, minority_samples)
        # Exclude the sample itself (distance = 0)
        distances_sorted_idx = np.argsort(distances)
        # Skip index 0 if it's the sample itself
        neighbours = []
        for idx in distances_sorted_idx:
            if distances[idx] == 0:
                continue
            neighbours.append(idx)
            if len(neighbours) == self.k:
                break
        return np.array(neighbours)

    def _generate_synthetic_sample(self,
                                   sample: np.ndarray,
                                   neighbours: np.ndarray,
                                   minority_samples: np.ndarray) -> np.ndarray:
        
        # Pick a random neighbour
        neighbour_idx = self.rng.choice(neighbours)
        neighbour = minority_samples[neighbour_idx]

        # Random interpolation factor
        lam = self.rng.uniform(0, 1)

        # Synthetic sample along the line between sample and neighbour
        synthetic = sample + lam * (neighbour - sample)
        return synthetic

    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray):
        
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)

        n_majority = class_counts[majority_class]
        n_minority = class_counts[minority_class]

        print(f"[SMOTE] Before resampling: {class_counts}")

        # Determine how many synthetic samples to generate
        if self.sampling_strategy == "auto":
            n_synthetic_needed = n_majority - n_minority
        else:
            target_minority = int(n_majority * self.sampling_strategy)
            n_synthetic_needed = max(0, target_minority - n_minority)

        if n_synthetic_needed == 0:
            print("[SMOTE] No oversampling needed.")
            return X, y

        # Get minority class samples
        minority_mask = (y == minority_class)
        minority_samples = X[minority_mask]

        # Generate synthetic samples
        synthetic_samples = []

        for i in range(n_synthetic_needed):
            # Pick a random minority sample as the base
            sample_idx = self.rng.randint(0, len(minority_samples))
            sample = minority_samples[sample_idx]

            # Find its k nearest neighbours within minority class
            k_actual = min(self.k, len(minority_samples) - 1)
            if k_actual == 0:
                synthetic_samples.append(sample)
                continue

            neighbours = self._find_k_neighbours(sample, minority_samples)
            if len(neighbours) == 0:
                synthetic_samples.append(sample)
                continue

            # Generate one synthetic sample
            synthetic = self._generate_synthetic_sample(
                sample, neighbours, minority_samples
            )
            synthetic_samples.append(synthetic)

        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(n_synthetic_needed, minority_class)

        # Combine original + synthetic
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate([y, synthetic_labels])

        new_counts = Counter(y_resampled)
        print(f"[SMOTE] After resampling:  {dict(new_counts)}")
        print(f"[SMOTE] Generated {n_synthetic_needed} synthetic minority samples")

        return X_resampled, y_resampled


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.RandomState(42)
    X_demo = rng.randn(200, 5)
    y_demo = np.array([0] * 160 + [1] * 40)  # 4:1 imbalance

    smote = SMOTE(k_neighbours=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_demo, y_demo)

    print(f"Original shape: {X_demo.shape}")
    print(f"Resampled shape: {X_res.shape}")
    assert len(X_res) == len(y_res), "Shape mismatch!"
    print("SMOTE sanity check passed.")

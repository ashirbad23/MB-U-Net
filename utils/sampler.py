import random
from torch.utils.data import Sampler


class GlacierBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()

        self.batch_size = batch_size

        self.pure_bg = []
        self.weak_fg = []
        self.strong_fg = []

        for i in range(len(dataset)):
            _, mask = dataset[i]

            if mask.ndim == 3:
                mask = mask.squeeze(0)

            fg_pixels = mask.sum().item()
            fg_ratio = fg_pixels / mask.numel()

            if fg_pixels == 0:
                self.pure_bg.append(i)
            elif fg_ratio < 0.05:
                self.weak_fg.append(i)
            else:
                self.strong_fg.append(i)

        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []

            n_strong = self.batch_size // 2
            n_weak = self.batch_size // 4
            n_bg = self.batch_size - n_strong - n_weak

            batch += random.sample(self.strong_fg, min(n_strong, len(self.strong_fg)))
            batch += random.sample(self.weak_fg, min(n_weak, len(self.weak_fg)))
            batch += random.sample(self.pure_bg, min(n_bg, len(self.pure_bg)))

            while len(batch) < self.batch_size:
                batch.append(random.choice(self.strong_fg))

            random.shuffle(batch)

            yield batch   # 🔥 IMPORTANT

    def __len__(self):
        return self.num_batches

from skimage import io
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import torch



def get_labels():
    return np.asarray(
            [
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [0, 255, 255],
                    [255, 127, 80],
                    [153, 0, 0],
                    ]
            )

mask = io.imread('/home/maui/software/ov_dset/AeroRIT/Aerial Data/Collection/image_labels.tif')[53:,7:,:]
mask = mask.astype(int)
label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
for ii, label in enumerate(get_labels()):
    label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
label_mask = label_mask.astype(int)
label_mask = label_mask.flatten()
# plt.imshow(label_mask)
# plt.show()

label_mask = label_mask[label_mask!=5]
print(label_mask.shape)


class_freq = Counter(label_mask.flatten())
total_pixels = sum(class_freq.values())
freq = {k: v / total_pixels for k, v in class_freq.items()}
med_freq = np.median(list(freq.values()))
class_weights = {k: med_freq / v for k, v in freq.items()}
# weights = torch.tensor(list(class_weights.values())).cuda()
ordered_list = []
for key in sorted(class_weights.keys()):
    ordered_list.append(class_weights[key])
weights = torch.tensor(list(ordered_list)).cuda()
print("primera forma", weights)


class_freq = Counter(label_mask)
total_pixels = sum(class_freq.values())
freq = {k: v / total_pixels for k, v in class_freq.items()}
med_freq = min(freq)
class_weights = {k: med_freq / v for k, v in freq.items()}
# weights = torch.tensor(list(class_weights.values())).cuda()
# weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]

ordered_list = []
for key in sorted(class_weights.keys()):
    ordered_list.append(class_weights[key])
weights = torch.tensor(list(ordered_list)).cuda()
print("segunda forma", weights)


from sklearn.utils import class_weight
classes = [0, 1, 2, 3, 4]
weights = class_weight.compute_class_weight(class_weight = 'balanced', classes=classes, y=np.array(label_mask))
print("sklearn", weights)
weights = weights / np.sum(weights)
print("sklearn norm", weights)
print(np.unique(label_mask, return_counts=1))

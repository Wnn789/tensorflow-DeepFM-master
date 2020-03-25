

import numpy as np

images_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
labels_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

state = np.random.get_state()
np.random.shuffle(images_list)
np.random.set_state(state)
np.random.shuffle(labels_list)

print(images_list)
print(labels_list)


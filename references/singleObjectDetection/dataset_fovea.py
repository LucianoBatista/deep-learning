from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


class AMDDataset(Dataset):
    def __init__(self, path2data, transform, trans_params) -> None:
        super(AMDDataset).__init__()
        path2labels = os.path.join(path2data, "Training400", "Fovea_location.xlsx")
        labels_df = pd.read_excel(path2labels, index_col="ID")
        self.labels = labels_df[["Fovea_X", "Fovea_Y"]].values
        self.imgName = labels_df["imgName"]
        self.ids = labels_df.index

        self.fullPath2img = []
        for id_ in self.ids:
            if self.imgName[id_][0] == "A":
                prefix = "AMD"
            else:
                prefix = "Non-AMD"

            self.fullPath2img.append(
                os.path.join(path2data, "Training400", prefix, self.imgName[id_])
            )

        self.transform = transform
        self.trans_params = trans_params

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.fullPath2img[idx])
        label = self.labels[idx]
        image, label = self.transform(image, label, self.trans_params)

        return image, label

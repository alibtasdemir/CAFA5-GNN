from torch.utils.data import Dataset


class CAFA5Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class CAFA5TestData(Dataset):
    def __init__(self, X_test_data):
        self.X_test_data = X_test_data

    def __getitem__(self, index):
        return self.X_test_data[index]

    def __len__(self):
        return len(self.X_test_data)
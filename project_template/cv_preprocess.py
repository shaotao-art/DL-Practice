from torchvision import transforms


class PreProcessor():
    def __init__(self):
        self.T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __call__(self, x):
        return self.T(x)

    def __str__(self):
        return str(self.T)


if __name__ == '__main__':
    obj = PreProcessor()
    print(obj)
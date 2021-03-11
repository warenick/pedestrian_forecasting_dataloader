import numpy as np


class Force_from_txt:
    def __init__(self, file='somefile.txt'):
        # k = []
        d = []
        # i=0
        with open(file, 'r') as f:
            for line in f:
                l = line.split(" ")
                d.append([int(l[0]),
                          float(l[1][1:-1]), float(l[2][:-1]),
                          float(l[3][1:-1]), float(l[4][:-1]),
                          float(l[5][1:-1]), float(l[6][:-2])])
        self.data = np.array(d)

    def get(self, i):
        if i not in self.data[:, 0]:
            return np.zeros((3, 2))
        arr = self.data[self.data[:, 0] == i, 1:][0]
        return np.array([[arr[0], arr[1]], [arr[2], arr[3]], [arr[4], arr[5]]])


if __name__ == "__main__":
    force_data = Force_from_txt()
    for i in range(1000):
        print(force_data.get(i))

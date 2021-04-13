import cv2
import numpy as np
import matplotlib.pyplot as plt


class Transformation():
    def __init__(self):
        pass

    def apply(self, image):
        raise NotImplementedError

    @property
    def transformation_parameters(self):

        raise NotImplementedError

    @property
    def transformation_matrix(self):
        # from old CS to NEW
        raise NotImplementedError


class Resize(Transformation):
    def __init__(self, resize_coef=(1, 1)):
        super().__init__()
        self.resize_coef = resize_coef

    def apply(self, image):
        small_img = cv2.resize(image, (int(image.shape[1]*self.resize_coef[1]), int(image.shape[0]*self.resize_coef[0])), interpolation=0)
        return small_img

    @property
    def transformation_parameters(self):
        return self.resize_coef

    @property
    def transformation_matrix(self):
        # from old CS to NEW
        tm = np.eye(3)
        tm[0,0] = self.resize_coef[1]
        tm[1, 1] = self.resize_coef[0]
        return tm


class Crop(Transformation):
    def __init__(self, point1=(1, 1), point2=()):
        super().__init__()
        self.point1 = point1
        self.point2 = point2

    def apply(self, image):
        small_img = image[self.point1[1]:self.point2[1], self.point1[0]:self.point2[0]]
        return small_img

    @property
    def transformation_parameters(self):
        return self.point1, self.point2

    @property
    def transformation_matrix(self):
        # from old CS to NEW
        tm = np.eye(3)
        tm[:2, 2] = -np.array([self.point1[0], self.point1[1]])
        return tm


class AddBorder(Transformation):
    def __init__(self, border_width=1):
        super().__init__()
        self.bw = border_width

    def apply(self, image):
        big_image = np.zeros((image.shape[0]+2*self.bw, image.shape[1]+2*self.bw, image.shape[2]), dtype=image.dtype)
        big_image[self.bw:-self.bw, self.bw:-self.bw] = 0+image
        return big_image

    @property
    def transformation_parameters(self):
        return self.bw

    @property
    def transformation_matrix(self):
        # from old CS to NEW
        tm = np.eye(3)
        tm[:2, 2] = np.array([self.bw, self.bw])
        return tm


class Rotate(Transformation):
    def __init__(self, angle=0, rot_center=(1, 1)):
        super().__init__()
        self.angle = angle
        self.rot_center = rot_center

    def apply(self, image):
        raise NotImplementedError


    @property
    def transformation_parameters(self):
        return self.angle, self.rot_center

    @property
    def transformation_matrix(self):
        # init_coord[:2]+=border_w
        #     print(init_coord)
        a = np.eye(3)
        # a[2,2] = 0
        # border = 200
        a[0, 2] = -self.rot_center[0]  # -border
        a[1, 2] = -self.rot_center[1]  # -border
        #     print("a", a)
        # b = np.eye(3)
        b = np.array([[np.cos(self.angle/180.*np.pi), np.sin(self.angle/180.*np.pi), 0],
                      [-np.sin(self.angle/180.*np.pi), np.cos(self.angle/180.*np.pi), 0],
                      [0, 0, 1]])

        c = np.eye(3)
        c[0, 2] = self.rot_center[0]  # -border
        c[1, 2] = self.rot_center[1]  # -border
        transf = c @ b @ a
        return transf


class ChangeOrigin(Transformation):
    def __init__(self, new_origin=(1, 1), rotation=np.ndarray((2, 2))):
        super().__init__()
        self.new_origin = new_origin
        self.rotation = rotation


    def apply(self, image):
        pass

    @property
    def transformation_parameters(self):
        return self.new_origin, self.rotation

    @property
    def transformation_matrix(self):
        tm = np.eye(3)
        tm[:2, 2] = -np.array([self.new_origin[0],  self.new_origin[1]])
        tm[:2, :2] *= self.rotation
        return tm


if __name__ == "__main__":

    image = np.zeros((600, 600, 3))
    image[:100, :100, :] = 255
    plt.imshow(image)
    plt.show()
    resize = Resize((0.5, 2))
    img = resize.apply(image)
    plt.imshow(img)
    plt.show()
    print()

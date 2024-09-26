import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage.morphology import skeletonize
from skimage.draw import polygon

# خواندن نقاط از فایل
def read_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            points.append((x, y))
    return np.array(points)

# تولید تصویر باینری از نقاط
def create_binary_image(points, img_size):
    img = np.zeros(img_size, dtype=np.uint8)
    rr, cc = polygon(points[:, 1], points[:, 0])
    img[rr, cc] = 1
    return img

# محاسبه محور میانی
def compute_medial_axis(binary_image):
    skeleton = skeletonize(binary_image)
    return skeleton

# بازسازی شکل با مثلث‌بندی دلونی
def reconstruct_shape_with_delaunay(points):
    tri = Delaunay(points)
    return tri

# نمایش نقاط اصلی
def plot_original_points(points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='red', s=1)
    plt.title('Original Points')
    plt.gca().invert_yaxis()
    plt.show()

# نمایش تصویر باینری
def plot_binary_image(binary_image):
    plt.figure()
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.show()

# نمایش محور میانی
def plot_medial_axis(medial_axis):
    plt.figure()
    plt.imshow(medial_axis, cmap='gray')
    plt.title('Medial Axis')
    plt.show()

# نمایش شکل بازسازی شده
def plot_reconstructed_shape(points, tri):
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.gca().invert_yaxis()
    plt.title('Reconstructed Shape')
    plt.show()

# مسیر فایل ورودی
file_path = 'horse-2D.txt'  # فایل خود را اینجا قرار دهید

# خواندن نقاط
points = read_points(file_path)

# اندازه تصویر
img_size = (int(points[:, 1].max()) + 1, int(points[:, 0].max()) + 1)

# نمایش نقاط اصلی
plot_original_points(points)

# تولید تصویر باینری
binary_image = create_binary_image(points, img_size)

# نمایش تصویر باینری
plot_binary_image(binary_image)

# محاسبه محور میانی
medial_axis = compute_medial_axis(binary_image)

# نمایش محور میانی
plot_medial_axis(medial_axis)

# بازسازی شکل با مثلث‌بندی دلونی
tri = reconstruct_shape_with_delaunay(points)

# نمایش شکل بازسازی شده
plot_reconstructed_shape(points, tri)

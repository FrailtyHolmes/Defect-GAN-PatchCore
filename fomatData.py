import os
import glob
import shutil
from abc import ABC, abstractmethod

# 将数据集路径规则改成项目中mvtec的样式

class AddNewClass(ABC):
    @abstractmethod
    def copy_to_mvtec(self, image_list, save_path):
        pass


class MagneticTile(AddNewClass):
    def __init__(self, path, classname):
        self.path = path
        self.classname = classname
        # 'mvtec'表示数据库的根路径, classname表示这个数据的名称
        self.train_folder = os.path.join('mvtec', classname, r'train/good')
        self.test_folder = os.path.join('mvtec', classname, 'test')
        self.gt_folder = os.path.join('mvtec', classname, 'ground_truth')
        self.image_dict = self.get_image_dict()

    def get_image_dict(self) -> dict:
        train_image = []
        test_image = []
        ground_truth = []

        jpg_files = glob.glob(os.path.join(self.path, '*\\Imgs\\*.jpg'))
        for file in jpg_files:
            path_list = file.split('\\')
            # train_image.append(file)
            if path_list[1] == 'MT_Free':
                train_image.append(file)
            else:
                test_image.append(file)

        png_files = glob.glob(os.path.join(self.path, '*\\Imgs\\*.png'))
        for file in png_files:
            ground_truth.append(file)

        image_dict = {'train': train_image, 'test': test_image, 'gt': ground_truth}
        return image_dict

    def copy_to_mvtec(self):
        main_folder_path = os.path.join('mvtec', self.classname)
        os.makedirs(main_folder_path, exist_ok=True)

        # 复制训练文件
        train_folder_path = os.path.join(main_folder_path, 'train', 'good')
        os.makedirs(train_folder_path, exist_ok=True)
        for file_path in self.image_dict['train']:
            path_list = file_path.split('\\')
            target_path = os.path.join(train_folder_path, path_list[-1])
            shutil.copy(file_path, target_path) # 复制粘贴

        # 复制测试文件
        for file_path in self.image_dict['test']:
            path_list = file_path.split('\\')
            test_folder_path = os.path.join(main_folder_path, 'test', path_list[1][3:])
            os.makedirs(test_folder_path, exist_ok=True)
            target_path = os.path.join(test_folder_path, path_list[-1])
            shutil.copy(file_path, target_path)

        # 复制ground_truth
        for file_path in self.image_dict['gt']:
            path_list = file_path.split('\\')
            gt_folder_path = os.path.join(main_folder_path, 'ground_truth', path_list[1][3:])
            os.makedirs(gt_folder_path, exist_ok=True)
            target_path = os.path.join(gt_folder_path, path_list[-1])
            shutil.copy(file_path, target_path)

#需要在src\datasets\mvtec.py的_CLASSNAMES中添加一个类——magnetic_tile
#需要在test数据集中加入good目录，用来存放无缺陷的图片
#运行时，参数部分直接输入 -d magnetic_tile

if __name__ == "__main__":
    mt = MagneticTile(r'Magnetic-Tile-Defect', 'magnetic_tile')
    mt.copy_to_mvtec()
    # print(mt.image_dict)

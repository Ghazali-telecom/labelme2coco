import os
import argparse
import json

from labelme import utils
import numpy as np
import glob
import PIL.Image


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(json_file)
            with open(json_file, "r") as fp:
                data = json.load(fp)
                # print('data',data)
                self.images.append(self.image(data, num))
                for labels in data["labels"]:
                    label = labels["values"][0].split(',')
                    # print(label, self.label)
                    for l in label:
                        if l not in self.label:
                            self.label.append(l)
                print('----------------------------------------------------------------')
                print('self.label',label)
                # print('data["shapes"]',len(data["shapes"]))
                for k,shapes in enumerate(data["shapes"]):
                    # # label = shapes["label"].split("_")
                    # label = shapes["label"]
                    # print('label',label)
                    # if label not in self.label:
                    #     self.label.append(label)
                    points = shapes["points"]
                    # print('points:',points)
                    self.annotations.append(self.annotation(points, label[k], num))
                    self.annID += 1
                print('self.annotations', self.annotations)
                print('----------------------------------------------------------------')

        # Sort all text labels so they are in the same order across data splits.
        print(self.label.sort())
        for k,label in enumerate(self.label):
            print('label_sort', label)
            self.categories.append(self.category(label))
        print('self.categories_sort', self.categories)
        # print('self.annotations', self.annotations)
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        # print(' height, width', height, width)
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        # print('category_label',label)
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)
        category["name"] = label
        print('category',category)
        return category

    def annotation(self, points, label, num):
        print('ann_label',label)
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        # print('annotation["segmentation"]',annotation["segmentation"])
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))
        print('annotation["bbox"]',annotation["bbox"])
        annotation["category_id"] =label # self.getcatid(label)
        print('annotation["category_id"] ',annotation["category_id"] )
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        # print('label _getcat', label)
        print('self.categories _getcat', self.categories)
        for category in self.categories:
            if label == category["name"]:
                # print('label == category["name"]',label,category["name"],category["id"])
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        # print('mask_getbbox',mask.sum())
        return self.mask2box(mask)

    def mask2box(self, mask):
        # print('mask2box',mask.sum())
        index = np.argwhere(mask == 1)
        # print('index',index)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        # print('img_shape[0]',img_shape[0],'img_shape[1]',img_shape[1])
        # img_shape = [img_shape[1], img_shape[0]]
        mask = np.zeros(img_shape, dtype=np.uint8)
        # print('mask_zero',mask.shape)
        mask = PIL.Image.fromarray(mask)
        
        xy = list(map(tuple, polygons))
        # print('xy',xy)
        # print('mask_fromarray',mask.show())
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        # print('polygons_to_mask',mask.sum())
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        # print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4,default=str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file."
    )
    parser.add_argument(
        "labelme_images",
        help="Directory to labelme images and annotation json files.",
        type=str,
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="valid_test.json"
    )
    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    
    labelme2coco(labelme_json, args.output)
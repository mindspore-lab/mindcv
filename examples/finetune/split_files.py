""" Extract images and generate ImageNet-style dataset directory """
import os
import shutil


# only for Aircraft dataset but not a general one
def extract_images(images_path, subset_name, annotation_file_path, copy=True):
    # read the annotation file to get the label of each image
    def annotations(annotation_file_path):
        image_label = {}
        with open(annotation_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                label = " ".join(line.split(" ")[1:]).replace("\n", "").replace("/", "_")
                if label not in image_label.keys():
                    image_label[label] = []
                    image_label[label].append(line.split(" ")[0])
                else:
                    image_label[label].append(line.split(" ")[0])
        return image_label

    # make a new folder for subset
    subset_path = images_path + subset_name
    os.mkdir(subset_path)

    # extract and copy/move images to the new folder
    image_label = annotations(annotation_file_path)
    for label in image_label.keys():
        label_folder = subset_path + "/" + label
        os.mkdir(label_folder)
        for image in image_label[label]:
            image_name = image + ".jpg"
            if copy:
                shutil.copy(images_path + image_name, label_folder + image_name)
            else:
                shutil.move(images_path + image_name, label_folder)


# take train set of aircraft dataset as an example
images_path = "./aircraft/data/images/"
subset_name = "trainval"
annotation_file_path = "./aircraft/data/images_variant_trainval.txt"
extract_images(images_path, subset_name, annotation_file_path)

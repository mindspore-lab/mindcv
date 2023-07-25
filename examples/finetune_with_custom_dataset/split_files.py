""" Extract images and generate ImageNet-style dataset directory """
import os
import shutil


def extract_images(images_path, subset_name, annotation_file_path):
    # read annotation file to get the label of each image
    def annotations(annotation_file_path):
        image_label = {}
        for i in open(annotation_file_path, "r"):
            label = " ".join(i.split(" ")[1:]).replace("\n", "").replace("/", "_")
            if label not in image_label.keys():
                image_label[label] = []
                image_label[label].append(i.split(" ")[0])
            else:
                image_label[label].append(i.split(" ")[0])
        return image_label

    # make a new folder for subset
    subset_path = images_path + subset_name
    os.mkdir(subset_path)

    # extract and copy images to the new folder above
    image_label = annotations(annotation_file_path)
    for label in image_label.keys():
        label_folder = subset_path + "/" + label
        os.mkdir(label_folder)
        for image in image_label[label]:
            image_name = image + ".jpg"
            shutil.copy(images_path + image_name, label_folder + image_name)


# take train set of aircraft dataset as an example
images_path = "./aircraft/data/images/"
subset_name = "trainval"
annotation_file_path = "./aircraft/data/images_variant_trainval.txt"
extract_images(images_path, subset_name, annotation_file_path)

"""
航拍图片分析覆盖度 classification 分析 2

Added crop image function

Selina Wang
Feb. 2022

"""
import os

import pickle
from glob import glob

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.formats.excel
from PIL import Image

# get base mean


@click.command()
@click.option("--image-valid-factor", type=float, default=1.0)

def main(image_valid_factor):

    # create folder to store results
    isExist = os.path.exists("Results")
    if not isExist:
        os.mkdir("Results")

    k = float(input("proportion of reduced image：")) # proportion of reduced image
    folder = input("folder name：")
    filename = input("calssification model: ")
    final_model = pickle.load(open(filename, "rb"))

    get_coverage_df(folder, final_model, k, image_valid_factor)


def get_base_means(k):
    # generate base means from a base image
    image = Image.open("./test_images/DJI_0206.JPG")
    image = reduce_im(image, k)
    image_array = np.array(image)
    image_array = image_array.astype("float32")

    image.close()

    base_means = image_array.mean(axis=(0, 1), dtype="float64")

    return base_means


def reduce_im(image, k):
    w, h = image.size
    w = int(k * w)
    h = int(k * h)
    image = image.resize((w, h))

    return image


def get_array(image):
    image_array = np.array(image)
    image_array = image_array.astype("float32")
    return image_array, image.size


def center_im(image_array, base_means):

    means = image_array.mean(axis=(0, 1), dtype="float64")

    # adjust according to base image
    adjust = np.diag(base_means / means)

    # center:
    image_array = image_array @ adjust

    # confirm it had the desired effect
    means = image_array.mean(axis=(0, 1), dtype="float64")
    # print('Means: %s' % means)
    # print('Mins: %s, Maxs: %s' % (image_array.min(axis=(0,1)), image_array.max(axis=(0,1))))

    return image_array


def flatten_im(image_array):
    image_flat = image_array.reshape((image_array.shape[0] * image_array.shape[1]), image_array.shape[2])
    return image_flat


def process_im(image, k, base_means):
    image = reduce_im(image, k)
    image_array, (w, h) = get_array(image)
    image_array = center_im(image_array, base_means)
    # plt.imshow(image_array.astype(int))

    image_array = flatten_im(image_array)

    return image_array, w, h


# functions for classification:


def classify_im(image_array, model):
    # apply prediction function
    clf_labels = model.predict(image_array)

    return clf_labels


def get_coverage(labels):
    # should create a class variable called labels :)
    total = np.shape(labels)[0]
    trees = np.sum(labels)

    return trees / total


def plot_im(labels, w, h, name):
    classified_image = labels.reshape(h, w)
    plt.imsave("Results/clf_{}".format(name), classified_image)


def crop_image(image, image_valid_factor):
    if image_valid_factor > 1.0:
        image_valid_factor = 1.0

    image_w, image_h = int(image.width), int(image.height)
    thumb_w = int(image_w * image_valid_factor)
    thumb_h = int(image_h * image_valid_factor)
    x = int((image_w - thumb_w) / 2)
    y = int((image_h - thumb_h) / 2)
    image = image.crop((x, y, x + thumb_w, y + thumb_h))
    # image.show()
    return image


def get_coverage_df(folder, final_model, k, image_valid_factor):
    image_names = glob("./{}/DJI*.JPG".format(folder))
    name_list = []
    coverage_list = []

    base_means = get_base_means(k)

    for filename in image_names:
        # get array from image file
        image = Image.open(filename)
        image = crop_image(image, image_valid_factor)
        processed_image, w, h = process_im(image, k, base_means)
        image.close()

        # strip name
        filename = os.path.basename(filename)

        # get classification labels
        labels = classify_im(processed_image, final_model)
        coverage = get_coverage(labels)

        # save classified image
        plot_im(labels, w, h, filename)

        coverage_list.append(coverage)
        name_list.append(filename)

    coverage_df = pd.DataFrame({"image_name": name_list, "coverage": coverage_list})
    # coverage_df['image_name'] = coverage_df['image_name'].str.split('/').str[2]
    coverage_df["folder"] = folder

    coverage_df.to_csv(f"Results/{folder}_coverage.csv")
    # output excel
    pandas.io.formats.excel.ExcelFormatter.header_style = None
    sheet_name = "Results"
    writer = pd.ExcelWriter(f"Results/{folder}_coverage.xlsx", engine="xlsxwriter")
    # setup excel style
    coverage_df.to_excel(writer, sheet_name=sheet_name, header=True)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    # Add column style
    worksheet.set_column(
        1,
        3,
        18,
        workbook.add_format(
            {
                "bold": False,
                "border": 1,
                "border_color": "#d4d4d4",
                "valign": "vcenter",
            }
        ),
    )
    worksheet.set_column(
        0,
        0,
        4,
        workbook.add_format(
            {
                "bold": False,
                "border": 1,
                "border_color": "#d4d4d4",
                "valign": "vcenter",
            }
        ),
    )
    # Add header style
    header_default_fmt = workbook.add_format(
        {
            "bold": True,
            "font_color": "black",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "border_color": "#d4d4d4",
            "font_size": 16,
        }
    )
    header_fmt = workbook.add_format(
        {
            "bg_color": "#DCB67A",
            "bold": True,
            "font_color": "black",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "border_color": "#d4d4d4",
            "font_size": 16,
        }
    )
    worksheet.set_default_row(24)
    worksheet.set_row(0, None, header_default_fmt)
    worksheet.conditional_format(
        "B1:D1",
        {
            "type": "no_blanks",
            "format": header_fmt,
        },
    )
    writer.save()

if __name__=="__main__":
	main()


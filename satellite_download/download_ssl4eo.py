#!/usr/bin/env python3

# Licensed under the MIT License.

""" Sample and download Satellite images with Google Earth Engine

#### run the script:

### Install and authenticate Google Earth Engine

### match and download pre-sampled locations
python download_ssl4eo.py \
        --save-path ./data \
        --dates 2021-12-21 \
        --radius 1320 \
        --dtype uint16 \
        --num-workers 16 \
        --match-file ./data/sampled_locations.csv \
        --indices-range 0 100000 \
        --output h5 \
        --batch-size 1000

### resume from interruption
python download_ssl4eo.py \
        --save-path ./data \
        --dates 2021-12-21 \
        --radius 1320 \
        --dtype uint16 \
        --num-workers 16 \
        --match-file ./data/sampled_locations.csv \
        --indices-range 0 100000 \
        --output h5 \
        --batch-size 1000 \
        --resume ./data/checked_locations.csv

"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import date, timedelta
from multiprocessing.dummy import Lock, Pool
from typing import Any, Optional
from tqdm import tqdm

import ee
import numpy as np
import rasterio
from rasterio.transform import Affine
import cv2
from shutil import rmtree
import h5py
import random
import string
from uuid import uuid4



def date2str(date: date) -> str:
    return date.strftime("%Y-%m-%d")


def get_period(date: date, days: int = 5):
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)
    date3 = date1 - timedelta(days=365)
    date4 = date2 - timedelta(days=365)
    return (
        date2str(date1),
        date2str(date2),
        date2str(date3),
        date2str(date4),
    )  # two-years buffer


"""get collection and remove clouds from ee"""


def mask_clouds(args: argparse.Namespace, image: ee.Image) -> ee.Image:
    qa = image.select(args.qa_band)
    cloudBitMask = 1 << args.qa_cloud_bit
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask)


def get_collection(
        collection_name: str, meta_cloud_name: str, cloud_pct: float
) -> ee.ImageCollection:
    collection = ee.ImageCollection(collection_name)
    collection = collection.filter(
        ee.Filter.And(
            ee.Filter.gte(meta_cloud_name, 0), ee.Filter.lte(meta_cloud_name, cloud_pct)
        )
    )
    # Uncomment the following line if you want to apply cloud masking.
    # collection = collection.map(mask_clouds, args)
    return collection


def filter_collection(
        collection: ee.ImageCollection,
        coords,
        period,
) -> ee.ImageCollection:
    filtered = collection
    if period is not None:
        # filtered = filtered.filterDate(*period)  # filter time, if there's one period
        filtered = filtered.filter(
            ee.Filter.Or(
                ee.Filter.date(period[0], period[1]),
                ee.Filter.date(period[2], period[3]),
            )
        )  # filter time, if there're two periods

    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f"ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}."
            # noqa: E501
        )
    return filtered


def center_crop(
        img: "np.typing.NDArray[np.float32]", out_size: int
) -> "np.typing.NDArray[np.float32]":
    image_height, image_width = img.shape[:2]
    crop_height = crop_width = out_size
    pad_height = max(crop_height - image_height, 0)
    pad_width = max(crop_width - image_width, 0)
    img = np.pad(img, ((pad_height, 0), (pad_width, 0), (0, 0)), mode="edge")
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top: crop_top + crop_height, crop_left: crop_left + crop_width]


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [
            coords[0][0] + ((xoff + new_size[1]) * xres),
            coords[0][1] - ((yoff + new_size[0]) * yres),
        ],
    ]


def get_patch(
        collection: ee.ImageCollection,
        center_coord,
        radius: float,
        bands,
        original_resolutions,
        new_resolutions,
        dtype: str = "float32",
        meta_cloud_name: str = "CLOUD_COVER",
        default_value: Optional[float] = None,
):
    image = collection.sort(meta_cloud_name).first()
    region = ee.Geometry.Point(center_coord).buffer(radius).bounds()

    # Group by original and new resolution
    band_groups = defaultdict(list)
    for i in range(len(bands)):
        band_groups[(original_resolutions[i], new_resolutions[i])].append((i, bands[i]))

    # Reproject (if necessary) and download all bands
    raster = {}
    for (orig_res, new_res), value in band_groups.items():
        indices, bands_group = zip(*value)
        patch = image.select(*bands_group)
        if orig_res != new_res:
            patch = patch.reproject(patch.projection().crs(), scale=new_res)
        patch = patch.sampleRectangle(region, default_value)
        features = patch.getInfo()
        for i, band in zip(indices, bands_group):
            x = features["properties"][band]
            x = np.atleast_3d(x)
            x = center_crop(x, out_size=int(2 * radius // new_res))
            raster[i] = x.astype(dtype)

    # Compute coordinates after cropping
    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    old_size = (len(features["properties"][band]), len(features["properties"][band][0]))
    new_size = raster[0].shape[:2]
    coords = adjust_coords(coords, old_size, new_size)

    return {"raster": raster, "coords": coords, "metadata": image.getInfo()}


def get_random_patches_match(
        idx: int,
        collection: ee.ImageCollection,
        bands,
        original_resolutions,
        new_resolutions,
        dtype: str,
        meta_cloud_name: str,
        default_value: Optional[float],
        dates,
        radius: float,
        debug: bool = False,
        match_coords={},
):
    # (lon,lat) of idx patch
    coords = match_coords[idx]

    # random +- 60 days of random days within 1 year from the reference dates
    periods = [get_period(date, days=60) for date in dates]

    try:
        filtered_collections = [
            filter_collection(collection, coords, p) for p in periods
        ]
        patches = [
            get_patch(
                c,
                coords,
                radius,
                bands,
                original_resolutions,
                new_resolutions,
                dtype,
                meta_cloud_name,
                default_value,
            )
            for c in filtered_collections
        ]
    except Exception as e:
        if debug:
            print(e)
        return [], coords

    return patches, coords


def save_geotiff(
        img: "np.typing.NDArray[np.float32]",
        coords,
        filename: str,
) -> None:
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": "None",
    }
    with rasterio.open(filename, "w", **profile) as f:
        # Transpose is necessary because rasterio uses band-major order,
        # meaning that the first axis is the band number and the last two are the image x and y.
        f.write(img.transpose(2, 0, 1))


def normalize(
        img: "np.typing.NDArray[np.float32]",
        mean,
        std,
):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_jpg(
        img: "np.typing.NDArray[np.float32]",
        filename: str,
):
    # Only use RGB bands
    bands = ['B4', 'B3', 'B2']
    # Mean and std for each band
    MEAN = {
        'B2': 1397.6,
        'B3': 1322.3,
        'B4': 1373.1}
    STD = {
        'B2': 854.3,
        'B3': 878.7,
        'B4': 1144.9}
    chs = []
    for i, band in enumerate(bands):
        ch = img[:, :, i]
        ch = cv2.resize(ch, dsize=(256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
        ch = normalize(ch, mean=MEAN[band], std=STD[band])
        chs.append(ch)
    img = np.stack(chs, axis=-1)

    height, width, channels = img.shape
    profile = {
        "driver": "JPEG",
        "width": width,
        "height": height,
        "count": channels,
        "dtype": img.dtype,
        "compress": "None",
    }
    with rasterio.open(filename, "w", **profile) as f:
        # Transpose is necessary because rasterio uses band-major order,
        # meaning that the first axis is the band number and the last two are the image x and y.
        f.write(img.transpose(2, 0, 1))


def process_img(
        img: "np.typing.NDArray[np.float32]",
):
    # Only use RGB bands
    bands = ['B4', 'B3', 'B2']
    # Mean and std for each band
    MEAN = {
        'B2': 1397.6,
        'B3': 1322.3,
        'B4': 1373.1}
    STD = {
        'B2': 854.3,
        'B3': 878.7,
        'B4': 1144.9}
    chs = []
    for i, band in enumerate(bands):
        ch = img[:, :, i]
        ch = cv2.resize(ch, dsize=(256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
        ch = normalize(ch, mean=MEAN[band], std=STD[band])
        chs.append(ch)
    img_out = np.stack(chs, axis=-1)
    return img_out


def save_patch(
        raster,
        coords,
        metadata,
        bands,
        new_resolutions,
        path: str,
        output_format: str,
        output_name,
):

    # JPG output
    if output_format == "jpg":
        image_path = os.path.join(path, "imgs")
        metadata_path = os.path.join(path, "metadata")
        # If there's only one resolution, save all bands in one file
        # Otherwise save each band in a separate file
        if len(set(new_resolutions)) == 1:
            img_all = np.concatenate([raster[i] for i in range(len(raster))], axis=2)
            save_jpg(img_all, os.path.join(image_path, f"{output_name}.jpg"))
        else:
            for i, band in enumerate(bands):
                img = raster[i]
                save_jpg(img, os.path.join(image_path, f"{output_name}_{band}.jpg"))

        # Save metadata in the same folder as the patch
        with open(os.path.join(metadata_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        return None

    # GeoTIFF output
    elif output_format == "geotiff":
        # Save into separate folder based on patch_id
        patch_id = metadata["properties"]["system:index"]
        patch_path = os.path.join(path, patch_id)
        os.makedirs(patch_path, exist_ok=True)

        # If there's only one resolution, save all bands in one file
        # Otherwise save each band in a separate file
        if len(set(new_resolutions)) == 1:
            img_all = np.concatenate([raster[i] for i in range(len(raster))], axis=2)
            save_geotiff(img_all, coords, os.path.join(patch_path, "all_bands.tif"))
        else:
            for i, band in enumerate(bands):
                img = raster[i]
                save_geotiff(img, coords, os.path.join(patch_path, f"{band}.tif"))

        # Save metadata in the same folder as the patch
        with open(os.path.join(patch_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        return None

    # H5 output
    elif output_format == "h5":
        if len(set(new_resolutions)) == 1:
            img_all = np.concatenate([raster[i] for i in range(len(raster))], axis=2)
            img_out = process_img(img_all)
            return img_out
        else:
            return None

    else:
        raise ValueError(f"Unknown output format: {output_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2", help="GEE collection name"
    )
    parser.add_argument("--qa-band", type=str, default="QA60", help="qa band name")
    parser.add_argument(
        "--qa-cloud-bit", type=int, default=10, help="qa band cloud bit"
    )
    parser.add_argument(
        "--meta-cloud-name",
        type=str,
        default="CLOUDY_PIXEL_PERCENTAGE",
        help="meta data cloud percentage name",
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=20, help="cloud percentage threshold"
    )
    # patch properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        # https://www.weather.gov/media/ind/seasons.pdf
        default=["2021-12-21", "2021-09-23", "2021-06-21", "2021-03-20"],
        help="reference dates",
    )
    parser.add_argument(
        "--radius", type=int, default=1320, help="patch radius in meters"
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=[
            "B4",
            "B3",
            "B2",
        ],
        help="bands to download (default are the RGB bands, B4, B3, B2)",
    )
    # Reprojection options
    #
    # If the original resolutions differ between bands, you can reproject them to
    # new resolutions. Crop dimensions are the size of each patch you want to crop
    # to after reprojection. All of these options should either be a single value
    # or the same length as the bands flag.
    parser.add_argument(
        "--original-resolutions",
        type=int,
        nargs="+",
        # B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
        # default=[60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20],
        # B4 B3 B2
        default=[10, 10, 10],
        help="original band resolutions in meters",
    )
    parser.add_argument(
        "--new-resolutions",
        type=int,
        nargs="+",
        default=[10],
        help="new band resolutions in meters",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="data type")
    # If None, don't download patches with nodata pixels
    parser.add_argument(
        "--default-value", type=float, default=None, help="default fill value"
    )
    # download settings
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--log-freq", type=int, default=1, help="print frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    # sampler options
    parser.add_argument(
        "--match-file",
        type=str,
        required=True,
        help="match pre-sampled coordinates and indexes",
    )
    # number of locations to download
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )
    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")

    # output
    parser.add_argument("--output", type=str, default="jpg", help="output format (jpg, geotiff, h5)")

    # batch size
    parser.add_argument("--batch-size", type=int, default=100, help="batch size (only works for h5 output)")

    # adaptive resolution
    parser.add_argument("--adaptive-resolution", action="store_true", help="adaptive resolution (needs special input file, containing the radii and resolutions)")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # initialize ee
    ee.Initialize()

    # get data collection (remove clouds)
    collection = get_collection(args.collection, args.meta_cloud_name, args.cloud_pct)

    dates = []
    for d in args.dates:
        dates.append(date.fromisoformat(d))

    bands = args.bands
    original_resolutions = args.original_resolutions
    new_resolutions = args.new_resolutions
    dtype = args.dtype

    # Validate inputs
    num_bands = len(bands)
    if len(original_resolutions) == 1:
        original_resolutions *= num_bands
    if len(new_resolutions) == 1:
        new_resolutions *= num_bands

    for values in [original_resolutions, new_resolutions]:
        assert len(values) == num_bands

    # if resume
    ext_coords = {}
    ext_flags = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = int(row[0])
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
                ext_flags[key] = int(row[3])  # success or not
    else:
        ext_path = os.path.join(args.save_path, "checked_locations.csv")

    # match from pre-sampled coords
    match_coords = {}
    if args.adaptive_resolution:
        match_sizes = {}
        match_resolutions = {}
    with open(args.match_file) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            key = int(row[0])
            val1 = float(row[1])
            val2 = float(row[2])
            if args.adaptive_resolution:
                match_sizes[key] = float(row[3])
                match_resolutions[key] = int(row[4])
            match_coords[key] = (val1, val2)  # lon, lat


    def worker(idx: int):
        if idx in ext_coords.keys():
            return None, None, None

        worker_start = time.time()
        # Extract a patch of a given coordinate for each date
        if args.adaptive_resolution:
            radius = match_sizes[idx]
            new_resolutions = [match_resolutions[idx]] * num_bands
        else:
            radius = args.radius
        patches, center_coord = get_random_patches_match(
            idx,
            collection,
            bands,
            original_resolutions,
            new_resolutions,
            dtype,
            args.meta_cloud_name,
            args.default_value,
            dates,
            radius=radius,
            debug=args.debug,
            match_coords=match_coords,
        )

        # If the extraction was successful, save the patches
        if patches:
            if args.output == "jpg":
                # Make a subfolder for images
                image_path = os.path.join(args.save_path, "imgs")
                os.makedirs(image_path, exist_ok=True)
                metadata_path = os.path.join(args.save_path, "metadata")
                os.makedirs(metadata_path, exist_ok=True)
                file_name = f"{idx:07d}"
                # Save only the first patch
                patch = patches[0]
                img = save_patch(
                        patch["raster"],
                        patch["coords"],
                        patch["metadata"],
                        bands,
                        new_resolutions,
                        args.save_path,
                        args.output,
                        file_name,
                    )

            elif args.output == "geotiff":
                # Make a subfolder with the location index
                location_path = os.path.join(args.save_path, "imgs", f"{idx:07d}")
                os.makedirs(location_path, exist_ok=True)
                # Save each patch at the location folder
                for patch in patches:
                    img = save_patch(
                            patch["raster"],
                            patch["coords"],
                            patch["metadata"],
                            bands,
                            new_resolutions,
                            location_path,
                            args.output,
                            None,
                            )

            elif args.output == "h5":
                for patch in patches:
                    img = save_patch(
                        patch["raster"],
                        patch["coords"],
                        patch["metadata"],
                        bands,
                        new_resolutions,
                        None,
                        args.output,
                        None,
                    )

            else:
                raise ValueError("Unknown output format: %s" % (args.output))

        else:
            img = None
            if args.debug:
                print("no suitable image for location %d." % (idx))

        # add to existing checked locations
        if not args.output == "h5":
            with open(ext_path, "a") as f:
                writer = csv.writer(f)
                if patches is not None:
                    success = 1
                else:
                    success = 0
                data = [idx, *center_coord, success]
                writer.writerow(data)

        # Throttle throughput to avoid exceeding GEE quota:
        # https://developers.google.com/earth-engine/guides/usage
        worker_end = time.time()
        elapsed = worker_end - worker_start
        num_workers = max(1, args.num_workers)
        time.sleep(max(0, num_workers / 100 - elapsed))

        return idx, center_coord, img


    # set indices
    indices = list(range(args.indices_range[0], args.indices_range[1]))

    # Save images to h5 batches
    if args.output == "h5":
        h5_location = os.path.join(args.save_path, "imgs_h5")
        os.makedirs(h5_location, exist_ok=True)
        imgs = []
        results = []
        batch_name = str(uuid4().hex)
        with Pool(processes=args.num_workers) as p:
            for idx,center_coord,img in tqdm(p.imap_unordered(worker, indices), total=len(indices)):
                # Skip if it's already processed
                if idx is None:
                    continue
                # Check for success
                if img is not None:
                    imgs.append(img)
                    success = 1
                    results.append([idx, *center_coord, success, batch_name])
                else:
                    success = 0
                    results.append([idx, *center_coord, success, 'None'])
                # Save image batch
                if len(imgs) == args.batch_size:
                    stack = np.stack(imgs)
                    with h5py.File(os.path.join(h5_location, f"{batch_name}.h5"), "w") as f:
                        f.create_dataset("imgs", data=stack)
                    # add to existing checked locations
                    with open(ext_path, "a") as f:
                        writer = csv.writer(f)
                        writer.writerows(results)
                    # Reset
                    imgs = []
                    results = []
                    batch_name = str(uuid4().hex)

    else:
        if args.num_workers == 0:
            for i in indices:
                _ = worker(i)
        else:
            # parallelism data
            with Pool(processes=args.num_workers) as p:
                _ = list(tqdm(p.imap(worker, indices), total=len(indices)))
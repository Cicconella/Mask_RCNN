"""
Mask R-CNN
Baloon -> Cells

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
import os
import sys
import datetime
import numpy as np
import skimage
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = '/home/cicconella/Mask_RCNN'
HOME_DIR = '/home/cicconella'
sys.path.append(HOME_DIR)
sys.path.append(ROOT_DIR)

# Import Mask RCNN
import tensorflow as tf
from config import Config
import utils
import model as modellib
import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
class DSBConfig(Config):
    NAME = "DSB"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 20
    EPOCHS = 4
    DETECTION_MIN_CONFIDENCE = 0.7

############################################################
#  Dataset
############################################################
class DSBDataset(utils.Dataset):

    def load_dsb(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("dsb", 1, "nucleo")

        # Train or validation dataset?
        if subset == "train" or subset == "val":
            dataset_dir = os.path.join(dataset_dir, "TRAINCLAHE/")
        elif subset == 'test':
            dataset_dir = os.path.join(dataset_dir, "TEST/")
        else:
            print("Invalid Subset",subset)
        #Listar quais exames tem
        exames = next(os.walk(dataset_dir))[1]

        if subset=="train":
            exames = exames[:600]
        elif subset == "val":
            exames = exames[600:]
        else:
            # exames = exames
            pass


        #Acessar a pasta exame/image
        for n, id_ in tqdm(enumerate(exames), total=len(exames)):
            path = dataset_dir + id_
            self.add_image(
                "dsb",
                image_id=id_,  # use file name as a unique image id
                path=path + '/images/' + id_ + '.png', dir=path,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "dsb":
            return super(self.__class__, self).load_mask(image_id)

        path = image_info["dir"]

        mascara = next(os.walk(path + '/masks/'))[2]
        masc = skimage.io.imread(path + '/masks/' + mascara[0])
        height, width = masc.shape

        mask = np.zeros((height, width, len(mascara)), dtype=np.uint8)

        for i, mask_file in enumerate(mascara):
            mask[:,:,i] = skimage.io.imread(path + '/masks/' + mask_file)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "dsb":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DSBDataset()
    dataset_train.load_dsb(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSBDataset()
    dataset_val.load_dsb(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs= config.EPOCHS,
                layers='5+') #heads

def test(model):
    """Train the model."""
    # Training dataset.
    dataset_test = DSBDataset()
    dataset_test.load_dsb(args.dataset, "test")
    dataset_test.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Testing on dataset")

    output = open("sub.csv","w")
    output.write("ImageId,EncodedPixels\n")
    for im in dataset_test.image_ids:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(im))
        # Read image
        im_path = dataset_test.image_info[im]["path"]
        im_name = dataset_test.image_info[im]["id"]
        image = skimage.io.imread(im_path)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        mask = remove_all_overlaps(r['masks'])
        s = mask.shape
        if s[0]==0:
            continue
        for nuc in range(s[2]):
            m = mask[:,:,nuc]
            l = ' '.join([ str(x) for x in rle_encoding(m)])
            strin = "%s,%s" % (im_name, l)
            output.write(strin+"\n")

    output.close()

def remove_all_overlaps(mask):
    n = mask.shape[2]
    for n1 in range(n):
        for n2 in range(n1+1,n):
            mask[:, :, n1],mask[:, :, n2] = remove_overlap(mask[:,:,n1], mask[:,:,n2])
    return mask

def remove_overlap(m1,m2):
    inter = m1*m2
    i =inter.flatten()
    if max(i) > 0:
        print("Overlap entre mascaras, tamanho =",len(i[i>0]))
    return m1-inter,m2

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        print("Image shape is ",image.shape)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        #boxes = visualize.display_top_masks(image, r['masks'], class_ids=[0], class_names=['nuclei'], limit=1)
        #print(boxes)
        print(r.keys())
        boxes = visualize.draw_boxes(image,boxes=r['boxes'], masks=r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        file_name = "boxes_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, boxes)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "test":
        assert args.dataset, \
               "Provide --dataset to test"
    print("Command:", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DSBConfig()
    else:
        class InferenceConfig(DSBConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "test":
        test(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train','splash' or 'test'".format(args.command))

# SignDet: Traffic sign detection suite
Collection of object detection methods, tips, and tricks we use at the [Intelligent Ground Vehicle Competition](http://igvc.org).


## Table of contents
* [Downloads](#downloads)
* [Dataset creation](#dataset-creation)
* [Model selection](#model-selection)
* [Training tips](#training-tips)
* [Further reading](#further-reading)

## Downloads <a name="downloads"></a>
------------
* [Pretrained YOLOv3 weights](https://files.drew.hu/igvc/igvc-yolov3_3000.weights) (240MB) using Alexey Bochkovskiy's [Darknet (YOLO) fork](https://github.com/AlexeyAB/darknet). Detects `{one_way_sign, person, road_closed_sign, stop_sign, traffic_drum, traffic_sign}`. The original Darknet/YOLO is by [Joseph Redmon](https://github.com/pjreddie) who developed YOLOv1, v2, and v3. Alexey made some improvements, aptly named [YOLOv4](https://arxiv.org/abs/2004.10934), after Joseph stopped work in object detection [due to ethical concerns](https://twitter.com/pjreddie/status/1230524770350817280).
* [IGVC Dataset](https://files.drew.hu/igvc/igvc-training-data.zip) (459MB) contains the training images and labels that we used to train our model. See the included `README.txt` for more details. Images came from a combination of Google Images ([google-images-download](https://github.com/hardikvasa/google-images-download)) and also just me walking around NYC meatpacking district with my smartphone camera.

## Dataset creation <a name="dataset-creation"></a>
-------------------

### Using existing datasets
There are a lot of autonomous driving related datasets available for free. Here's a brief list, but new ones are released regularly so look around for yourself. 
* [Cityscapes](https://www.cityscapes-dataset.com/) is one of the more popular autonomous driving datasets, and is often used as a benchmark in research papers.
* [KITTI]() is another popular dataset that's often used in research.
* [Waymo Open Dataset](https://waymo.com/open/data/) provides 2D (camera) and 3D (Lidar) object detection data.
* [nuScenes Dataset](https://www.nuscenes.org/)
* [Berkeley DeepDrive]()


For more, see Scale's [index of Open Datasets](https://scale.com/open-datasets).


There are also many general-purpose object detection datasets. You can download subsets of these datasets with whichever classes you need.
* [Common Objects in Context (COCO)](http://cocodataset.org/). The COCO dataset has over 80 object categories and is a standard dataset for benchmarking object detection methods. For our object detector, we used the `person` and `stop_sign` classes.

### Creating your own dataset
If the existing object detection datasets don't have the data you need, you can create your own dataset. It's not as hard as it seems!

For IGVC we needed to detect stop signs, people, traffic barrels, and one way signs. COCO only includes stop signs and people, so we downloaded images with [google-images-download](https://github.com/hardikvasa/google-images-download), and labeled them with [Multiclass-BBox-Label-Tool](https://github.com/andrewhu/Multiclass-BBox-Label-Tool) which is a fork of [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool) with a few UI improvements.

You only need 200-300 labeled images per class to get good results (by fine-tuning existing models), but more is always better!

### Synthetic datasets
One interesting idea we explored was generating synthetic datasets using cut+paste methods. Basically the idea is paste your object onto some random background and create data that way. 

For traffic signs and traffic drums, this method worked really well. Using just a single "template" image per class and pasting them onto images from the [MIT Places dataset](http://places2.csail.mit.edu/), we managed to get accuracy comparable to using actual images.

It should be noted that the reason this worked is likely because traffic signs and traffic drums have very little variance in their appearance (almost all the same color and shape), so it's a relatively easy task for the model to learn. If you were 

Here's a paper about the idea: [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/pdf/1708.01642.pdf)

### Dataset creation tips
* **"Money see, monkey do."** This means your model will learn the distribution of the data you feed it. So if you only train your object detector on images where stop signs take up a large portion of the space, then your model will have difficulty detecting small stop signs. 
* **Actually spend time *looking* at your data!** As computer scientists and machine learning practitioners, we fantasize the idea of computers doing all the work for us. However, spending the time & effort looking through your data, removing outliers, and getting an understanding of what it looks like is an important step. Your model will only be as good as the data you train it with.
 

## Model selection <a name="model-selection"></a>
------------------
New papers are constantly being published which improve the accuracy and speed of object detection models. However, unless you need to have SOTA accuracy/speed, there are a few well established models that will work well enough.
* [YOLOv4](https://arxiv.org/abs/2004.10934) is the latest iteration of the YOLO series of object detectors. It claims to have a better speed/accuracy ratio than competitors like EfficientDet. I can't confirm this, but I got good results with YOLOv3 so this is a safe bet. It was also relatively easy to train and use.

Check out [Papers with Code](https://paperswithcode.com/task/object-detection) for benchmarks and [awesome-object-detection](https://github.com/amusi/awesome-object-detection) for an up-to-date list of papers.

It's also a good idea to have a basic understanding of how the model you choose works. You don't need to know how to implement it from scratch, but sometimes some models will have a maximum number of objects that they can detect at a time, which is useful to know.

## Training tips <a name="training-tips"></a>

* Sometimes the object detector will fail for one or two consecutive frames, either predicting an object in the wrong place or not predicting an object at all. Your other software should be robust to this. 

* Consider using an ensemble of models. For example, when detecting one way signs, you might make two classes, `one_way_left` and `one_way_right`. However, the issue is that your model may have difficulty recognizing whether the sign is pointing right or left, and may not have enough confidence to pass the detection threshold or may misclassify it. Instead, try making a class for `one_way_sign` and having a separate binary classifier to decide whether the sign is pointing right or left.

## Further reading <a name="further-reading"></a>
* Andrej Karpathy's [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) provides a lot of good advice for training neural networks.

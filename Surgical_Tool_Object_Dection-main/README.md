# Surgical Tool Object Detection

This project focuses developping effective object detector to detect surgical tools. We specifically study on the capability of transformer bsaed architecture for surgical tool dataset.

The dataset is avaiable at here:

https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools

We provide an example of implementation for the dataset class from pytorch in ```surgical_tool_dataset.ipynb```.

## Instructions for running Deformable DETR:

1. You should download the dataset and save it in ```./Deformable-DETR-surgical_tool/data``` for properly run.

2. To run the Deformable DETR, remember to build up the requriements dependencies with using:

```
pip install -r requirements_surgical_tool.txt
```

Also, following the ```./Deformable-DETR-surgical_tool/README.md```. to properly run the ```./Deformable-DETR-surgical_tool/models/ops/make.sh``` file, to build up the deformable DETR into the system.

3. Install the pretrained deformable DETR checkpoint from:
   - https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view  for base deformable DETR

   - https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view for deformable DETR with box refinement architecture

   Save these checkpoint into  ```./Deformable-DETR-surgical_tool/pretrain_ckpt```

4. Now, you should be able to properly run:
	
	```bash ./Deformable-DETR-surgical_tool/train_surgical_tool.sh```

	You can remove the tag of ```--debug``` for a fully running.

5. For running the box refinement architecture:

	```bash ./Deformable-DETR-surgical_tool/train_with_box_refine.sh```

## TODOS:
1. we can make some other conv architecture to cast gray scale image input to the 3-channel input required by the DETR.

Maybe use some batch normalization over the initially casted 3-channel input

2. finish modifying the "evaluate.py" file.

3. maybe run YOLO or faster-rcnn to have some cross comparison

4. think about novelity modification. We can certainly choose to modify the backbone structure, where we will need to pre-train over the COCO dataset first and then transfer here.

5. We also need to add more image size augmentation process into the ```main_surgical_tool.py``` file. One thing to notice is that we need to follow the augmented image size used in the COCO training case.

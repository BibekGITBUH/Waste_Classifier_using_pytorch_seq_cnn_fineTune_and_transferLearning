# Smart Waste Image Classifier

Image classification project comparing a custom CNN (trained from scratch)
against transfer learning with ResNet50, across 5 waste categories:
**glass, metal, organic, paper, plastic**.

---

## 1. Project structure

```
project/
├── config.py              # all paths & hyperparameters live here
├── dataset.py              # builds train/val/test DataLoaders
├── prepare_data.py         # splits raw images into train/val/test folders
├── train.py                # trains a model, saves best checkpoint + history
├── evaluate.py              # evaluates a checkpoint on the test set
├── predict.py               # run inference on a single image or folder
├── compare_models.py        # compares multiple trained runs side by side
├── requirements.txt
├── models/
│   ├── custom_cnn.py        # from-scratch CNN baseline
│   └── pretrained.py        # ResNet50 transfer-learning wrapper
├── datasets/
│   └── final/                # <-- put your raw data here (see step 2)
├── checkpoints/              # best model weights get saved here
└── outputs/                  # metrics, history, confusion matrices, plots
```

---

## 2. Setup

### 2.1 Create environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, install the matching `torch`/`torchvision`
build from https://pytorch.org/get-started/locally/ instead of the plain
`pip install` above, for GPU acceleration.

### 2.2 Add your raw data

Place your images under `datasets/final/`, one subfolder per class,
matching the names in `config.CLASS_NAMES` exactly (case-sensitive,
alphabetical order):

```
datasets/final/
├── glass/     *.jpg
├── metal/     *.jpg
├── organic/   *.jpg
├── paper/     *.jpg
└── plastic/   *.jpg
```

### 2.3 Split into train/val/test

```bash
python prepare_data.py
```

This copies images into `datasets/split/{train,val,test}/<class>/` using
an 80/10/10 split (configurable in `config.py` via `TRAIN_RATIO`,
`VAL_RATIO`, `TEST_RATIO`). The split is seeded (`SPLIT_SEED`) so it's
reproducible.

Sanity check it worked:

```bash
python dataset.py
```

You should see batch counts and a sample batch shape printed with no errors.

---

## 3. Training

### 3.1 Custom CNN (baseline)

```bash
python train.py --model custom_cnn
```

### 3.2 ResNet50 — frozen backbone (fast, trains only the new head)

```bash
python train.py --model resnet50 --freeze-backbone
```

### 3.3 ResNet50 — full fine-tune (slower, usually more accurate)

```bash
python train.py --model resnet50 --no-freeze-backbone
```

Each run saves:
- `checkpoints/<run_name>_best.pt` — best weights by validation accuracy
- `outputs/<run_name>_history.json` — per-epoch train/val loss & accuracy

Where `<run_name>` is `custom_cnn`, `resnet50_frozen`, or `resnet50_finetuned`.

Training stops early if validation accuracy doesn't improve for
`EARLY_STOPPING_PATIENCE` epochs (default 5), and uses a
`ReduceLROnPlateau` scheduler. All of this is configurable in `config.py`.

**Note on fine-tuning LR:** when `--no-freeze-backbone` is used, the
learning rate is automatically scaled to 10% of `config.LEARNING_RATE`,
since large updates can destroy pretrained features. Edit `train.py`'s
`train_model()` if you want to control this explicitly instead.

---

## 4. Evaluation

Run on the held-out test set — reports accuracy, per-class precision/
recall/F1, macro & weighted averages, and saves a confusion matrix plot:

```bash
python evaluate.py --model custom_cnn
python evaluate.py --model resnet50 --freeze-backbone
python evaluate.py --model resnet50 --no-freeze-backbone
```

Each run saves:
- `outputs/<run_name>_classification_report.json`
- `outputs/<run_name>_confusion_matrix.png`

(You must train a model before evaluating it — `evaluate.py` will error
with a clear message if the checkpoint isn't found.)

---

## 5. Comparing models

Once you've trained + evaluated the runs you care about:

```bash
python compare_models.py
```

This prints a table of test accuracy / macro F1 / weighted F1 for each
run, and saves `outputs/model_comparison_val_acc.png` — a plot of
validation accuracy per epoch across all runs, for visualizing which
model converged fastest / generalized best.

To compare a custom subset of runs:

```bash
python compare_models.py --runs custom_cnn resnet50_finetuned
```

---

## 6. Inference (predicting new images)

Single image:

```bash
python predict.py --image path/to/image.jpg --model resnet50 --no-freeze-backbone
```

Folder of images (also writes a timestamped CSV of results into that folder):

```bash
python predict.py --folder path/to/images/ --model custom_cnn
```

If `--model` is omitted, it defaults to `custom_cnn`. If using
`resnet50`, remember `--freeze-backbone` / `--no-freeze-backbone` must
match whichever checkpoint you actually trained (default is
`--freeze-backbone`).

---

## 7. Typical end-to-end workflow

```bash
# one-time setup
pip install -r requirements.txt
python prepare_data.py

# train all three
python train.py --model custom_cnn
python train.py --model resnet50 --freeze-backbone
python train.py --model resnet50 --no-freeze-backbone

# evaluate all three
python evaluate.py --model custom_cnn
python evaluate.py --model resnet50 --freeze-backbone
python evaluate.py --model resnet50 --no-freeze-backbone

# compare
python compare_models.py

# predict on new images
python predict.py --folder path/to/new_images/ --model resnet50 --no-freeze-backbone
```

---

## 8. Notes

- `prepare_data.py` wasn't part of the original scripts you shared — I
  added it because `config.py` and `dataset.py` both assume a
  `datasets/split/{train,val,test}` layout already exists. If you already
  have your own splitting script, just drop your split data straight into
  `datasets/split/` and skip this one.
- `ResNet50_Weights.IMAGENET1K_V2` in `models/pretrained.py` requires a
  reasonably recent `torchvision` (≥0.15). If you're pinned to an older
  version and it errors on import, tell me your `torchvision` version and
  I'll adjust to the older weights API.
- MobileNetV2 was intentionally left as a placeholder hook (see the
  comment in `train.py`'s `build_model()`) in case you want to add it later.












```bash

### Terminal Log during Train / Test / Execute


D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python prepare_data.py
Splitting data: train=0.8, val=0.1, test=0.1 (seed=42)

glass      total=2237  train=1789  val=223   test=225  
metal      total=1340  train=1072  val=134   test=134  
organic    total=699   train=559   val=69    test=71   
paper      total=1930  train=1544  val=193   test=193  
plastic    total=2079  train=1663  val=207   test=209  

Done. Totals -> train=6627, val=826, test=832
Split data written to: datasets\split/

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python train.py --model custom_cnn
Using device: cpu
C:\Users\BIBEK\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py:1095: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
Epoch  1/25 | train_loss=1.3658 train_acc=0.4224 | val_loss=1.2757 val_acc=0.4697 | 592.8s
Epoch  2/25 | train_loss=1.2681 train_acc=0.4773 | val_loss=1.1993 val_acc=0.5073 | 536.7s
Epoch  3/25 | train_loss=1.2099 train_acc=0.5100 | val_loss=1.2694 val_acc=0.5012 | 558.3s
Epoch  4/25 | train_loss=1.2140 train_acc=0.4981 | val_loss=1.1847 val_acc=0.4976 | 554.6s
Epoch  5/25 | train_loss=1.1722 train_acc=0.5170 | val_loss=1.2864 val_acc=0.5061 | 549.6s
Epoch  6/25 | train_loss=1.0899 train_acc=0.5607 | val_loss=1.0445 val_acc=0.5944 | 591.0s
Epoch  7/25 | train_loss=1.0634 train_acc=0.5686 | val_loss=1.0148 val_acc=0.6029 | 587.3s
Epoch  8/25 | train_loss=1.0577 train_acc=0.5794 | val_loss=1.1664 val_acc=0.5472 | 586.0s
Epoch  9/25 | train_loss=1.0281 train_acc=0.5980 | val_loss=0.9956 val_acc=0.6077 | 585.5s
Epoch 10/25 | train_loss=0.9803 train_acc=0.6164 | val_loss=1.0631 val_acc=0.5969 | 586.1s
Epoch 11/25 | train_loss=0.9623 train_acc=0.6309 | val_loss=0.9052 val_acc=0.6622 | 584.8s
Epoch 12/25 | train_loss=0.9301 train_acc=0.6394 | val_loss=0.8596 val_acc=0.6695 | 584.8s
Epoch 13/25 | train_loss=0.9273 train_acc=0.6487 | val_loss=0.8165 val_acc=0.6828 | 583.4s
Epoch 14/25 | train_loss=0.8819 train_acc=0.6677 | val_loss=0.9861 val_acc=0.6162 | 583.7s
Epoch 15/25 | train_loss=0.8820 train_acc=0.6698 | val_loss=0.9102 val_acc=0.6501 | 586.4s
Epoch 16/25 | train_loss=0.8487 train_acc=0.6789 | val_loss=0.8583 val_acc=0.6683 | 584.7s
Epoch 17/25 | train_loss=0.7857 train_acc=0.7107 | val_loss=0.7526 val_acc=0.7094 | 584.0s
Epoch 18/25 | train_loss=0.7583 train_acc=0.7186 | val_loss=0.7002 val_acc=0.7312 | 584.0s
Epoch 19/25 | train_loss=0.7406 train_acc=0.7237 | val_loss=0.7996 val_acc=0.7167 | 584.9s
Epoch 20/25 | train_loss=0.7294 train_acc=0.7346 | val_loss=0.6927 val_acc=0.7458 | 584.2s
Epoch 21/25 | train_loss=0.7159 train_acc=0.7355 | val_loss=0.7019 val_acc=0.7373 | 585.2s
Epoch 22/25 | train_loss=0.7068 train_acc=0.7383 | val_loss=0.6017 val_acc=0.7893 | 584.4s
Epoch 23/25 | train_loss=0.6980 train_acc=0.7429 | val_loss=0.6383 val_acc=0.7651 | 584.3s
Epoch 24/25 | train_loss=0.6831 train_acc=0.7466 | val_loss=0.6696 val_acc=0.7482 | 583.7s
Epoch 25/25 | train_loss=0.6651 train_acc=0.7583 | val_loss=0.6416 val_acc=0.7446 | 585.0s

Best val accuracy: 0.7893
Saved best checkpoint to: checkpoints\custom_cnn_best.pt
Saved training history to: outputs\custom_cnn_history.json

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>
D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python train.py --model resnet50--freeze-backbone
usage: train.py [-h] [--model {custom_cnn,resnet50}] [--freeze-backbone] [--no-freeze-backbone]
train.py: error: argument --model: invalid choice: 'resnet50--freeze-backbone' (choose from 'custom_cnn', 'resnet50')

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python train.py --model resnet50 --freeze-backbone
Using device: cpu
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to C:\Users\BIBEK/.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:13<00:00, 7.41MB/s]
C:\Users\BIBEK\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py:1095: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
Epoch  1/25 | train_loss=0.7042 train_acc=0.7807 | val_loss=0.4640 val_acc=0.8692 | 422.0s
Epoch  2/25 | train_loss=0.4152 train_acc=0.8663 | val_loss=0.3619 val_acc=0.8838 | 421.9s
Epoch  3/25 | train_loss=0.3570 train_acc=0.8797 | val_loss=0.3179 val_acc=0.8995 | 412.9s
Epoch  4/25 | train_loss=0.3207 train_acc=0.8927 | val_loss=0.2937 val_acc=0.8971 | 409.7s
Epoch  5/25 | train_loss=0.2856 train_acc=0.9066 | val_loss=0.2754 val_acc=0.9153 | 407.8s
Epoch  6/25 | train_loss=0.2827 train_acc=0.9048 | val_loss=0.2758 val_acc=0.9092 | 413.2s
Epoch  7/25 | train_loss=0.2718 train_acc=0.9048 | val_loss=0.2565 val_acc=0.9165 | 409.4s
Epoch  8/25 | train_loss=0.2445 train_acc=0.9185 | val_loss=0.2526 val_acc=0.9128 | 434.0s
Epoch  9/25 | train_loss=0.2367 train_acc=0.9178 | val_loss=0.2582 val_acc=0.9177 | 433.1s
Epoch 10/25 | train_loss=0.2396 train_acc=0.9175 | val_loss=0.2469 val_acc=0.9201 | 447.2s
Epoch 11/25 | train_loss=0.2275 train_acc=0.9247 | val_loss=0.2425 val_acc=0.9213 | 445.5s
Epoch 12/25 | train_loss=0.2231 train_acc=0.9259 | val_loss=0.2384 val_acc=0.9225 | 446.8s
Epoch 13/25 | train_loss=0.2184 train_acc=0.9241 | val_loss=0.2400 val_acc=0.9213 | 431.4s
Epoch 14/25 | train_loss=0.2164 train_acc=0.9283 | val_loss=0.2288 val_acc=0.9262 | 423.1s
Epoch 15/25 | train_loss=0.2052 train_acc=0.9289 | val_loss=0.2364 val_acc=0.9165 | 422.2s
Epoch 16/25 | train_loss=0.2028 train_acc=0.9303 | val_loss=0.2352 val_acc=0.9213 | 422.7s
Epoch 17/25 | train_loss=0.2092 train_acc=0.9286 | val_loss=0.2294 val_acc=0.9225 | 422.2s
Epoch 18/25 | train_loss=0.1873 train_acc=0.9389 | val_loss=0.2286 val_acc=0.9213 | 422.8s
Epoch 19/25 | train_loss=0.1884 train_acc=0.9344 | val_loss=0.2302 val_acc=0.9213 | 421.5s
Early stopping at epoch 19 (no improvement for 5 epochs).

Best val accuracy: 0.9262
Saved best checkpoint to: checkpoints\resnet50_frozen_best.pt
Saved training history to: outputs\resnet50_frozen_history.json

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python train.py --model resnet50 --no-freeze-backbone
Using device: cuda
Epoch  1/25 | train_loss=0.4926 train_acc=0.8363 | val_loss=0.2249 val_acc=0.9213 | 70.1s
Epoch  2/25 | train_loss=0.1630 train_acc=0.9470 | val_loss=0.1495 val_acc=0.9564 | 68.3s
Epoch  3/25 | train_loss=0.1090 train_acc=0.9654 | val_loss=0.1466 val_acc=0.9540 | 67.7s
Epoch  4/25 | train_loss=0.0795 train_acc=0.9768 | val_loss=0.1376 val_acc=0.9576 | 68.0s
Epoch  5/25 | train_loss=0.0535 train_acc=0.9825 | val_loss=0.1297 val_acc=0.9625 | 69.3s
Epoch  6/25 | train_loss=0.0450 train_acc=0.9852 | val_loss=0.1515 val_acc=0.9637 | 68.6s
Epoch  7/25 | train_loss=0.0428 train_acc=0.9860 | val_loss=0.1693 val_acc=0.9613 | 68.7s
Epoch  8/25 | train_loss=0.0428 train_acc=0.9866 | val_loss=0.1391 val_acc=0.9649 | 68.0s
Epoch  9/25 | train_loss=0.0409 train_acc=0.9876 | val_loss=0.1196 val_acc=0.9649 | 68.1s
Epoch 10/25 | train_loss=0.0297 train_acc=0.9908 | val_loss=0.0988 val_acc=0.9673 | 68.4s
Epoch 11/25 | train_loss=0.0285 train_acc=0.9919 | val_loss=0.1459 val_acc=0.9625 | 68.0s
Epoch 12/25 | train_loss=0.0482 train_acc=0.9858 | val_loss=0.1210 val_acc=0.9649 | 68.3s
Epoch 13/25 | train_loss=0.0348 train_acc=0.9900 | val_loss=0.1267 val_acc=0.9709 | 69.8s
Epoch 14/25 | train_loss=0.0356 train_acc=0.9891 | val_loss=0.1500 val_acc=0.9625 | 68.2s
Epoch 15/25 | train_loss=0.0212 train_acc=0.9940 | val_loss=0.1319 val_acc=0.9649 | 68.2s
Epoch 16/25 | train_loss=0.0244 train_acc=0.9917 | val_loss=0.1186 val_acc=0.9661 | 68.4s
Epoch 17/25 | train_loss=0.0172 train_acc=0.9943 | val_loss=0.1023 val_acc=0.9637 | 68.5s
Epoch 18/25 | train_loss=0.0082 train_acc=0.9973 | val_loss=0.1152 val_acc=0.9673 | 68.4s
Early stopping at epoch 18 (no improvement for 5 epochs).

Best val accuracy: 0.9709
Saved best checkpoint to: checkpoints\resnet50_finetuned_best.pt
Saved training history to: outputs\resnet50_finetuned_history.json

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python evaluate.py --model custom_cnn

=== custom_cnn — Test Set Results ===
Accuracy: 0.7909

Class       Precision   Recall      F1          Support   
glass       0.8441      0.6978      0.7640      225       
metal       0.7039      0.7985      0.7483      134       
organic     0.8250      0.9296      0.8742      71        
paper       0.8214      0.8342      0.8278      193       
plastic     0.7661      0.7990      0.7822      209       

Macro avg   0.7921      0.8118      0.7993      

Saved full classification report to: outputs\custom_cnn_classification_report.json
Saved confusion matrix plot to: outputs\custom_cnn_confusion_matrix.png

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python evaluate.py --model resnet50 --freeze-backbone

=== resnet50_frozen — Test Set Results ===
Accuracy: 0.9315

Class       Precision   Recall      F1          Support   
glass       0.9127      0.9289      0.9207      225       
metal       0.8344      0.9403      0.8842      134       
organic     0.9855      0.9577      0.9714      71        
paper       0.9947      0.9741      0.9843      193       
plastic     0.9485      0.8804      0.9132      209       

Macro avg   0.9352      0.9363      0.9348      

Saved full classification report to: outputs\resnet50_frozen_classification_report.json
Saved confusion matrix plot to: outputs\resnet50_frozen_confusion_matrix.png

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python evaluate.py --model resnet50 --no-freeze-backbone

=== resnet50_finetuned — Test Set Results ===
Accuracy: 0.9663

Class       Precision   Recall      F1          Support   
glass       0.9692      0.9778      0.9735      225       
metal       0.9161      0.9776      0.9458      134       
organic     0.9861      1.0000      0.9930      71        
paper       0.9896      0.9845      0.9870      193       
plastic     0.9697      0.9187      0.9435      209       

Macro avg   0.9661      0.9717      0.9686      

Saved full classification report to: outputs\resnet50_finetuned_classification_report.json
Saved confusion matrix plot to: outputs\resnet50_finetuned_confusion_matrix.png
                                                                                                                          python compare_models.py
Run                   Test Acc    Macro F1    Weighted F1 nn_fineTune_and_transferLearning\smart-waste-classifier\project>
----------------------------------------------------------
custom_cnn            0.7909      0.7993      0.7902      
resnet50_frozen       0.9315      0.9348      0.9320      
resnet50_finetuned    0.9663      0.9686      0.9663      

Saved comparison plot to: outputs\model_comparison_val_acc.png

D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>



D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python predict.py --folder datasets/split/self_test
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 14 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
glass1.jpeg                                                  paper        69.85%
glass2.jpeg                                                  glass        74.43%
glass3.jpeg                                                  glass        87.06%
glass4.jpeg                                                  glass        46.96%
metal1.jpeg                                                  metal        66.83%
metal2.jpeg                                                  glass        84.32%
organic1.jpeg                                                organic      97.72%
paper1.jpeg                                                  paper        93.16%
plastic1.jpeg                                                plastic      50.09%
plastic2.jpeg                                                glass        50.42%
plastic3.jpeg                                                metal        53.03%
plastic4.jpeg                                                plastic      78.12%
plastic5.jpeg                                                plastic      57.61%
plastic6.jpeg                                                plastic      50.83%
====================================================================================

✓ Processing complete! Total: 14 images
✓ Results saved to: datasets\split\self_test\predictions_20260711_151403.csv

Summary by Class:
  glass           5 images
  metal           2 images
  organic         1 images
  paper           2 images
  plastic         4 images

Average Confidence: 68.60%




D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python predict.py --folder datasets/split/self_test --model custom_cnn
✓ Loaded model from: checkpoints\custom_cnn_best.pt

Found 14 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
glass1.jpeg                                                  paper        69.85%
glass2.jpeg                                                  glass        74.43%
glass3.jpeg                                                  glass        87.06%
glass4.jpeg                                                  glass        46.96%
metal1.jpeg                                                  metal        66.83%
metal2.jpeg                                                  glass        84.32%
organic1.jpeg                                                organic      97.72%
paper1.jpeg                                                  paper        93.16%
plastic1.jpeg                                                plastic      50.09%
plastic2.jpeg                                                glass        50.42%
plastic3.jpeg                                                metal        53.03%
plastic4.jpeg                                                plastic      78.12%
plastic5.jpeg                                                plastic      57.61%
plastic6.jpeg                                                plastic      50.83%
====================================================================================

✓ Processing complete! Total: 14 images
✓ Results saved to: datasets\split\self_test\predictions_20260711_151839.csv

Summary by Class:
  glass           5 images
  metal           2 images
  organic         1 images
  paper           2 images
  plastic         4 images

Average Confidence: 68.60%
10/14



D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python predict.py --folder datasets/split/self_test --model resnet50 --freeze-backbone
✓ Loaded model from: checkpoints\resnet50_frozen_best.pt

Found 14 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
glass1.jpeg                                                  paper        99.98%
glass2.jpeg                                                  glass        96.75%
glass3.jpeg                                                  glass        99.19%
glass4.jpeg                                                  plastic      53.37%
metal1.jpeg                                                  metal        99.45%
metal2.jpeg                                                  glass        55.53%
organic1.jpeg                                                organic      85.46%
paper1.jpeg                                                  paper        99.92%
plastic1.jpeg                                                plastic      55.94%
plastic2.jpeg                                                paper        99.31%
plastic3.jpeg                                                glass        51.07%
plastic4.jpeg                                                plastic      72.31%
plastic5.jpeg                                                plastic      79.17%
plastic6.jpeg                                                paper        77.52%
====================================================================================

✓ Processing complete! Total: 14 images
✓ Results saved to: datasets\split\self_test\predictions_20260711_152037.csv

Summary by Class:
  glass           4 images
  metal           1 images
  organic         1 images
  paper           4 images
  plastic         4 images

Average Confidence: 80.36%
8/14



D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python predict.py --folder datasets/split/self_test --model resnet50 --no-freeze-backbone
✓ Loaded model from: checkpoints\resnet50_finetuned_best.pt

Found 14 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
glass1.jpeg                                                  paper        95.72%
glass2.jpeg                                                  glass        98.71%
glass3.jpeg                                                  glass        99.97%
glass4.jpeg                                                  glass        76.23%
metal1.jpeg                                                  metal        99.61%
metal2.jpeg                                                  metal        73.87%
organic1.jpeg                                                organic      95.93%
paper1.jpeg                                                  paper        99.98%
plastic1.jpeg                                                plastic      99.94%
plastic2.jpeg                                                paper        90.95%
plastic3.jpeg                                                glass        75.02%
plastic4.jpeg                                                paper        99.24%
plastic5.jpeg                                                metal        82.18%
plastic6.jpeg                                                paper        99.93%
====================================================================================

✓ Processing complete! Total: 14 images
✓ Results saved to: datasets\split\self_test\predictions_20260711_152148.csv

Summary by Class:
  glass           4 images
  metal           3 images
  organic         1 images
  paper           5 images
  plastic         1 images

Average Confidence: 91.95%
8/14





D:\Projects_on_Python\Waste_Classifier_using_pytorch_seq_cnn_fineTune_and_transferLearning\smart-waste-classifier\project>python predict.py --folder datasets/split/test/glass --model resnet50 --no-freeze-backbone
✓ Loaded model from: checkpoints\resnet50_finetuned_best.pt

Found 225 images. Processing...

Image Path                                                   Predicted    Confidence  
====================================================================================
trashnet_10.jpg                                              glass        100.00%
trashnet_11.jpg                                              glass        97.82%
trashnet_121.jpg                                             glass        99.99%
trashnet_124.jpg                                             glass        100.00%
trashnet_125.jpg                                             glass        99.92%
trashnet_140.jpg                                             glass        98.79%
trashnet_17.jpg                                              glass        99.98%
trashnet_176.jpg                                             glass        83.45%
trashnet_190.jpg                                             glass        96.12%
trashnet_191.jpg                                             glass        99.98%
trashnet_196.jpg                                             glass        99.89%
trashnet_208.jpg                                             glass        97.21%
trashnet_213.jpg                                             glass        91.72%
trashnet_215.jpg                                             glass        99.97%
trashnet_216.jpg                                             glass        99.99%
trashnet_237.jpg                                             glass        99.69%
trashnet_258.jpg                                             glass        99.95%
trashnet_267.jpg                                             glass        100.00%
trashnet_270.jpg                                             glass        99.85%
trashnet_271.jpg                                             glass        99.65%
trashnet_279.jpg                                             glass        89.25%
trashnet_298.jpg                                             glass        100.00%
trashnet_3.jpg                                               glass        99.96%
trashnet_304.jpg                                             glass        99.91%
trashnet_311.jpg                                             glass        95.51%
trashnet_315.jpg                                             glass        99.96%
trashnet_321.jpg                                             plastic      99.75%
trashnet_332.jpg                                             glass        99.14%
trashnet_334.jpg                                             glass        99.70%
trashnet_342.jpg                                             glass        99.72%
trashnet_347.jpg                                             glass        99.90%
trashnet_353.jpg                                             glass        99.81%
trashnet_354.jpg                                             glass        100.00%
trashnet_357.jpg                                             glass        99.98%
trashnet_36.jpg                                              glass        99.95%
trashnet_361.jpg                                             glass        99.93%
trashnet_389.jpg                                             glass        99.94%
trashnet_392.jpg                                             glass        99.89%
trashnet_413.jpg                                             glass        99.76%
trashnet_419.jpg                                             glass        100.00%
trashnet_433.jpg                                             glass        99.95%
trashnet_436.jpg                                             glass        98.86%
trashnet_44.jpg                                              plastic      77.19%
trashnet_443.jpg                                             glass        87.32%
trashnet_446.jpg                                             glass        99.97%
trashnet_449.jpg                                             glass        99.94%
trashnet_451.jpg                                             glass        99.99%
trashnet_455.jpg                                             glass        99.96%
trashnet_456.jpg                                             glass        90.80%
trashnet_461.jpg                                             glass        99.97%
trashnet_470.jpg                                             glass        99.79%
trashnet_475.jpg                                             glass        100.00%
trashnet_476.jpg                                             glass        99.88%
trashnet_490.jpg                                             glass        100.00%
trashnet_52.jpg                                              glass        99.89%
trashnet_59.jpg                                              glass        99.99%
trashnet_60.jpg                                              glass        99.92%
trashnet_70.jpg                                              glass        99.69%
trashnet_94.jpg                                              glass        99.95%
v2_1004.jpg                                                  glass        99.99%
v2_1006.jpg                                                  glass        99.46%
v2_101.jpg                                                   glass        83.45%
v2_1019.jpg                                                  glass        66.36%
v2_1033.jpg                                                  glass        99.96%
v2_1056.jpg                                                  glass        99.89%
v2_106.jpg                                                   glass        99.35%
v2_1060.jpg                                                  glass        99.95%
v2_1073.jpg                                                  glass        99.94%
v2_111.jpg                                                   glass        99.99%
v2_1119.jpg                                                  glass        99.98%
v2_1129.jpg                                                  glass        99.14%
v2_1134.jpg                                                  glass        99.99%
v2_1136.jpg                                                  glass        99.99%
v2_1141.jpg                                                  glass        99.64%
v2_1146.jpg                                                  glass        100.00%
v2_1149.jpg                                                  glass        99.93%
v2_1155.jpg                                                  glass        99.99%
v2_1171.jpg                                                  glass        99.99%
v2_1177.jpg                                                  glass        99.99%
v2_1204.jpg                                                  glass        98.96%
v2_1248.jpg                                                  glass        99.15%
v2_1255.jpg                                                  glass        99.76%
v2_128.jpg                                                   glass        99.45%
v2_1288.jpg                                                  glass        99.97%
v2_1289.jpg                                                  glass        65.04%
v2_1319.jpg                                                  glass        99.65%
v2_1324.jpg                                                  glass        99.99%
v2_1327.jpg                                                  glass        99.99%
v2_133.jpg                                                   glass        100.00%
v2_1336.jpg                                                  glass        99.70%
v2_134.jpg                                                   glass        99.99%
v2_1352.jpg                                                  glass        99.98%
v2_1355.jpg                                                  glass        100.00%
v2_1356.jpg                                                  glass        99.97%
v2_1359.jpg                                                  glass        99.98%
v2_1363.jpg                                                  glass        99.98%
v2_137.jpg                                                   glass        99.97%
v2_1373.jpg                                                  glass        99.99%
v2_1375.jpg                                                  glass        100.00%
v2_1387.jpg                                                  glass        100.00%
v2_1390.jpg                                                  glass        99.44%
v2_1391.jpg                                                  glass        99.99%
v2_1395.jpg                                                  glass        99.99%
v2_14.jpg                                                    glass        99.97%
v2_1403.jpg                                                  glass        99.98%
v2_1404.jpg                                                  glass        99.99%
v2_1423.jpg                                                  glass        100.00%
v2_1429.jpg                                                  glass        99.64%
v2_1434.jpg                                                  glass        99.96%
v2_1442.jpg                                                  glass        99.89%
v2_1449.jpg                                                  glass        68.83%
v2_145.jpg                                                   glass        99.87%
v2_1456.jpg                                                  glass        100.00%
v2_1460.jpg                                                  glass        99.68%
v2_1473.jpg                                                  glass        99.13%
v2_1515.jpg                                                  glass        99.93%
v2_1517.jpg                                                  glass        62.93%
v2_1521.jpg                                                  glass        99.80%
v2_1522.jpg                                                  glass        100.00%
v2_1523.jpg                                                  glass        99.95%
v2_153.jpg                                                   glass        99.18%
v2_1530.jpg                                                  glass        100.00%
v2_1533.jpg                                                  glass        99.99%
v2_1535.jpg                                                  glass        100.00%
v2_1541.jpg                                                  glass        98.83%
v2_1543.jpg                                                  glass        99.98%
v2_1560.jpg                                                  glass        100.00%
v2_1561.jpg                                                  glass        99.89%
v2_1571.jpg                                                  glass        98.91%
v2_1572.jpg                                                  glass        99.88%
v2_1573.jpg                                                  glass        99.96%
v2_1613.jpg                                                  glass        99.43%
v2_1627.jpg                                                  glass        99.99%
v2_1629.jpg                                                  glass        99.99%
v2_1647.jpg                                                  glass        99.88%
v2_1680.jpg                                                  glass        99.98%
v2_1696.jpg                                                  glass        97.93%
v2_1706.jpg                                                  glass        99.99%
v2_171.jpg                                                   glass        99.43%
v2_18.jpg                                                    glass        99.94%
v2_180.jpg                                                   glass        99.97%
v2_187.jpg                                                   glass        98.15%
v2_202.jpg                                                   glass        99.96%
v2_225.jpg                                                   glass        99.97%
v2_230.jpg                                                   glass        95.18%
v2_239.jpg                                                   metal        98.56%
v2_252.jpg                                                   glass        99.62%
v2_291.jpg                                                   glass        100.00%
v2_295.jpg                                                   glass        99.90%
v2_308.jpg                                                   glass        99.93%
v2_318.jpg                                                   glass        99.98%
v2_319.jpg                                                   glass        100.00%
v2_322.jpg                                                   glass        99.92%
v2_327.jpg                                                   glass        100.00%
v2_33.jpg                                                    glass        99.95%
v2_35.jpg                                                    glass        99.98%
v2_361.jpg                                                   glass        99.94%
v2_363.jpg                                                   glass        99.99%
v2_374.jpg                                                   glass        99.72%
v2_380.jpg                                                   glass        99.18%
v2_384.jpg                                                   glass        99.97%
v2_386.jpg                                                   glass        99.98%
v2_392.jpg                                                   glass        99.98%
v2_403.jpg                                                   glass        100.00%
v2_443.jpg                                                   glass        99.97%
v2_456.jpg                                                   glass        99.99%
v2_457.jpg                                                   glass        99.99%
v2_460.jpg                                                   glass        96.69%
v2_464.jpg                                                   glass        97.10%
v2_474.jpg                                                   glass        99.98%
v2_486.jpg                                                   glass        99.99%
v2_50.jpg                                                    glass        99.99%
v2_526.jpg                                                   glass        99.70%
v2_531.jpg                                                   glass        95.81%
v2_540.jpg                                                   glass        100.00%
v2_541.jpg                                                   glass        99.94%
v2_543.jpg                                                   glass        99.99%
v2_545.jpg                                                   glass        97.27%
v2_546.jpg                                                   glass        98.79%
v2_563.jpg                                                   glass        99.98%
v2_564.jpg                                                   glass        99.93%
v2_573.jpg                                                   glass        99.57%
v2_574.jpg                                                   glass        100.00%
v2_588.jpg                                                   glass        99.98%
v2_597.jpg                                                   glass        100.00%
v2_60.jpg                                                    glass        99.99%
v2_640.jpg                                                   glass        99.99%
v2_65.jpg                                                    glass        100.00%
v2_657.jpg                                                   plastic      77.19%
v2_658.jpg                                                   glass        99.93%
v2_677.jpg                                                   glass        99.37%
v2_679.jpg                                                   glass        99.84%
v2_687.jpg                                                   glass        100.00%
v2_69.jpg                                                    glass        99.04%
v2_693.jpg                                                   glass        99.96%
v2_706.jpg                                                   glass        99.98%
v2_710.jpg                                                   glass        99.33%
v2_728.jpg                                                   glass        98.90%
v2_729.jpg                                                   glass        99.88%
v2_737.jpg                                                   glass        99.85%
v2_771.jpg                                                   glass        99.64%
v2_775.jpg                                                   plastic      99.60%
v2_777.jpg                                                   glass        99.96%
v2_787.jpg                                                   glass        99.60%
v2_804.jpg                                                   glass        96.50%
v2_823.jpg                                                   glass        99.99%
v2_825.jpg                                                   glass        99.97%
v2_830.jpg                                                   glass        100.00%
v2_848.jpg                                                   glass        99.77%
v2_851.jpg                                                   glass        98.56%
v2_857.jpg                                                   glass        99.99%
v2_864.jpg                                                   glass        99.95%
v2_891.jpg                                                   glass        99.96%
v2_91.jpg                                                    glass        99.93%
v2_929.jpg                                                   glass        99.99%
v2_961.jpg                                                   glass        99.97%
v2_962.jpg                                                   glass        99.60%
v2_963.jpg                                                   glass        99.41%
v2_970.jpg                                                   glass        100.00%
v2_976.jpg                                                   glass        99.97%
v2_977.jpg                                                   glass        96.80%
v2_982.jpg                                                   glass        53.22%
v2_984.jpg                                                   glass        99.99%
v2_994.jpg                                                   glass        99.83%
v2_999.jpg                                                   glass        99.99%
====================================================================================

✓ Processing complete! Total: 225 images
✓ Results saved to: datasets\split\test\glass\predictions_20260711_154119.csv

Summary by Class:
  glass         220 images
  metal           1 images
  organic         0 images
  paper           0 images
  plastic         4 images

Average Confidence: 98.31%


```
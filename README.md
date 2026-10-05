# Fetal Head Abnormality Detection

Fetal head segmentation in ultrasound images using a U-Net, trained on a combined
dataset of real HC18 images and SSIM-filtered synthetic images.

## Why this project

Head circumference (HC) is a standard measurement for monitoring fetal growth, and it
starts with an accurate segmentation of the fetal head. Annotated ultrasound data is
limited, so this project explores augmenting the real training set with synthetic
images and keeping only those that pass a quality check.

## What the notebook does

1. Loads the HC18 training data (999 image-mask pairs).
2. Filters generated images by SSIM against a mean real reference image
   (threshold 0.35).
3. Measures realism of the filtered set with FID.
4. Merges real and filtered synthetic pairs into one training set.
5. Trains a U-Net (from scratch, 2-class output) with a Cross-Entropy + Dice loss.
6. Evaluates with Dice and IoU and exports the best checkpoint.

## Results from the saved run

| Item | Value |
|---|---|
| Real images | 999 |
| Synthetic images kept by SSIM filter | 326 of 999 |
| Mean SSIM of kept images | 0.4076 |
| FID (real vs filtered synthetic) | 32.18 |
| Combined dataset | 1325 pairs (1127 train / 198 validation) |

The saved training output stops at epoch 3 (best validation Dice 0.2197). Run the
notebook to completion (20 epochs) and add your final Dice and IoU here, together with
the curves from `results/figures/`.

## Repository structure

    Fetal_Head_Abnormality_Detection/
      README.md
      requirements.txt
      .gitignore
      notebooks/
        Fetal_Head_Abnormality_Detection.ipynb
      data/
        README.md              how to obtain the dataset (images are not committed)
      results/
        figures/               training curves and sample predictions

## How to run

1. Open `notebooks/Fetal_Head_Abnormality_Detection.ipynb` in Google Colab
   (GPU runtime recommended).
2. Run all cells. Upload `train.zip` when prompted (see `data/README.md`).
3. After training, download `best_model.pth` and `curves.png` from the last cell.

Local setup:

    pip install -r requirements.txt

## Known limitations

- The generated-image paths in the configuration cell currently point to the real
  training folders (placeholder). Point them to the GAN-generated images and masks
  before reporting results; otherwise the validation split may contain duplicates of
  training images.
- No separate test set; a single random 85/15 split is used.
- SSIM against one mean image is a coarse quality filter.

See Section 10 of the notebook for next steps.

## Tech stack

Python, PyTorch, torchvision, OpenCV, scikit-image, pytorch-fid, Matplotlib, Google Colab.

## Dataset citation

HC18: van den Heuvel et al., "Automated measurement of fetal head circumference using
2D ultrasound images", PLOS ONE, 2018.

## Author

GitHub: [Iamaniketray](https://github.com/Iamaniketray)

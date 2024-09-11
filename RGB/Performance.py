import numpy as np
import cv2

from UNet import UNet
from MRIDataset import MRIDataset

# Classe che fornisce i metodi per calcolare le performance: Dice, Score, IoU, BIoU
class Performance():
    # Dice coefficient
    def dc_loss(self, pred, target):
        smooth = 1.

        predf = pred.view(-1)
        targetf = target.view(-1)
        intersection = (predf * targetf).sum()
        
        return 1 - ((2. * intersection + smooth) /
                (predf.sum() + targetf.sum() + smooth))

    def class_wise(arr: np.array, c: int) -> np.array:
        return arr == c

    # Intersection over union
    def iou(prediction: np.array, target: np.array) -> float:

        miou = []

        for c in range(2):

            pred = Performance.class_wise(prediction, c)
            tar = Performance.class_wise(target, c)

            if pred.dtype != bool:
                pred = np.asarray(pred, dtype=bool)

            if tar.dtype != bool:
                tar = np.asarray(tar, dtype=bool)

            overlap = pred * tar # Logical AND
            union = pred + tar # Logical OR

            if union.sum() != 0 and overlap.sum() != 0:
                iou = (float(overlap.sum()) / float(union.sum()))
            else:
                iou = 0

            if c in target:
                miou.append(iou)

        return np.asarray(miou).mean()

    # Funzione per ottenere i bordi di una maschera binaria
    def _mask_to_boundary(mask, dilation_ratio=0.02):

        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        
        return mask - mask_erode

    # Boundary Intersection over Union, usata per la qualità nell'identificazione dei bordi
    def biou(dt, gt, dilation_ratio=0.02):

        mboundary_iou = []

        for c in range(2):

            target = Performance.class_wise(gt, c)
            if not np.any(target):
                continue

            prediction = Performance.class_wise(dt, c)

            gt_boundary = Performance._mask_to_boundary(target.astype(np.uint8), dilation_ratio)
            dt_boundary = Performance._mask_to_boundary(prediction.astype(np.uint8), dilation_ratio)
            intersection = ((gt_boundary * dt_boundary) > 0).sum()
            union = ((gt_boundary + dt_boundary) > 0).sum()
            if union == 0 or intersection == 0:
                boundary_iou = 0
            else:
                boundary_iou = (intersection / union)

            mboundary_iou.append(boundary_iou)

        return np.asarray(mboundary_iou).mean()

    # Score calcolato a partire da IoU e BIoU
    def calculate_score(self, preds: np.array, tars: np.array) -> dict:
        
        preds = preds.astype(np.uint8)
        tars = tars.astype(np.uint8)

        assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
        assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
        assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
        assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
        assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
        assert type(tars[0][0][0]) == np.uint8, f"tars is not of type np.uint8, but {type(tars[0][0][0])}"

        bs = preds.shape[0]

        t_iou = 0
        t_biou = 0

        for i in range(bs):
            t_iou += Performance.iou(preds[i], tars[i])
            t_biou += Performance.biou(preds[i], tars[i])

        t_iou /= bs
        t_biou /= bs

        score = (t_iou + t_biou) / 2

        return {"score": score, "iou": t_iou, "biou": t_biou}
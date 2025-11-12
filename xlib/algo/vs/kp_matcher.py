import cv2
import cv2
import numpy as np
from ...device.sensor.camera import Camera
from ..RoMa.romatch import *


class RomaMatchAlgo:
    def __init__(self, model_type="roma_indoor", device="cuda") -> None:
        self.model_type = model_type
        self.model = eval(model_type)(device=device)
        self.device = device

    def match(
        self, img1: np.ndarray, img2: np.ndarray, mask=None, ransac=True, camera=None
    ):
        assert img1 is not None, "Color Image 1 not provided"
        assert img2 is not None, "Color Image 2 not provided"
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if self.model_type == "tiny_roma_v1_outdoor":
            warp, certainty = self.model.match(img1, img2)
        else:
            warp, certainty = self.model.match(img1, img2, device=self.device)
        matches, certainty = self.model.sample(warp, certainty)
        kptsA, kptsB = self.model.to_pixel_coordinates(matches, h1, w1, h2, w2)
        F, ransac_mask = cv2.findFundamentalMat(
            kptsA.cpu().numpy(),
            kptsB.cpu().numpy(),
            ransacReprojThreshold=0.2,
            method=cv2.RANSAC,
            confidence=0.999999,
            maxIters=10000,
        )
        kptsA_array = kptsA.cpu().numpy()
        kptsB_array = kptsB.cpu().numpy()
        if ransac:
            ransac_mask = ransac_mask.ravel().astype(bool)
            kptsA_array = kptsA_array[ransac_mask].reshape(-1, 2)
            kptsB_array = kptsB_array[ransac_mask].reshape(-1, 2)

        if mask is not None:
            kptsA_array_int = np.clip(
                kptsA_array.round().astype(int), [0, 0], [w1 - 1, h1 - 1]
            )
            kptsB_array_int = np.clip(
                kptsB_array.round().astype(int), [0, 0], [w2 - 1, h2 - 1]
            )

            x1, y1 = kptsA_array_int[:, 0], kptsA_array_int[:, 1]
            x2, y2 = kptsB_array_int[:, 0], kptsB_array_int[:, 1]
            mask = mask[y1, x1] & mask[y2, x2]
            # mask = mask[y1, x1]
            # print(mask.sum(), kptsA_array_int.shape[0])
            mask = ~mask

            kptsA_array = kptsA_array[mask]
            kptsB_array = kptsB_array[mask]
        if kptsA_array.shape[0] > 200:
            samples = np.random.choice(kptsA_array.shape[0], 200, replace=False)
            kptsA_array = kptsA_array[samples]
            kptsB_array = kptsB_array[samples]
        match_img = self._draw_matches(img1, kptsA_array, img2, kptsB_array)
        return kptsA_array, kptsB_array, match_img

    def _draw_matches(self, img1, keypoints1, img2, keypoints2):
        img1_height, img1_width = img1.shape[:2]
        img2_height, img2_width = img2.shape[:2]

        combined_image = np.zeros(
            (max(img1_height, img2_height), img1_width + img2_width, 3), dtype=np.uint8
        )

        combined_image[:img1_height, :img1_width] = img1

        combined_image[:img2_height, img1_width : img1_width + img2_width] = img2
        shifted_keypoints2 = [(x + img1_width, y) for (x, y) in keypoints2]

        for (x1, y1), (x2, y2) in zip(keypoints1, shifted_keypoints2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(combined_image, (int(x1), int(y1)), 3, color, 1)
            cv2.circle(combined_image, (int(x2), int(y2)), 3, color, 1)
            cv2.line(combined_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        return combined_image


class KpMatchAlgo:
    def __init__(self, kp_extractor: str = "SIFT", match_threshold=0.6) -> None:
        self.match_threshold = match_threshold
        self.kp_extractor = self._parser_kp_extractor(kp_extractor)
        self.matcher = cv2.FlannBasedMatcher()

    def _parser_kp_extractor(self, kp_extractor_str: str = "SIFT"):
        kp_extractor = "cv2." + kp_extractor_str.upper() + "_create"
        try:
            kp_extractor = eval(kp_extractor)
        except:
            raise ValueError("Invalid keypoint extractor in opencv")
        return kp_extractor()

    def _kp_extract(self, img1: np.ndarray, img2: np.ndarray):
        assert img1 is not None, "Color Image 1 not provided"
        assert img2 is not None, "Color Image 2 not provided"
        kp1, des1 = self.kp_extractor.detectAndCompute(img1, None)
        kp2, des2 = self.kp_extractor.detectAndCompute(img2, None)
        return kp1, des1, kp2, des2

    def match(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask=None,
        ransac: bool = True,
        camera: Camera = None,
    ):
        assert img1 is not None, "Color Image 1 not provided"
        assert img2 is not None, "Color Image 2 not provided"
        kp1, des1, kp2, des2 = self._kp_extract(img1, img2)
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches.append([m])

        if len(good_matches) < 3:
            print("Too few matches :{}".format(len(good_matches)))
            return None, None, None
        kp1_array = np.array(
            [kp1[good_matches[i][0].queryIdx].pt for i in range(len(good_matches))]
        )

        kp2_array = np.array(
            [kp2[good_matches[i][0].trainIdx].pt for i in range(len(good_matches))]
        )
        if mask is not None:
            kp1_array_int = kp1_array.round().astype(int)
            kp2_array_int = kp2_array.round().astype(int)

            x1, y1 = kp1_array_int[:, 0], kp1_array_int[:, 1]
            x2, y2 = kp2_array_int[:, 0], kp2_array_int[:, 1]
            mask = mask[y1, x1] & mask[y2, x2]
            # mask = mask[y1, x1]
            mask = ~mask
            kp1_array = kp1_array[mask]
            kp2_array = kp2_array[mask]
            mask_indices = np.where(mask)[0]
            good_matches = [good_matches[i] for i in mask_indices]

        if len(good_matches) > 4 and ransac:
            _, mask = cv2.findEssentialMat(
                kp1_array.reshape(-1, 1, 2),
                kp2_array.reshape(-1, 1, 2),
                camera.intrinsics_matrix,
                cv2.RANSAC,
                0.999,
                3.0,
            )
            mask = mask.ravel().astype(bool)
            kp1_array = kp1_array[mask].reshape(-1, 2)
            kp2_array = kp2_array[mask].reshape(-1, 2)
            mask_indices = np.where(mask)[0]
            good_matches = [good_matches[i] for i in mask_indices]
        if len(good_matches) < 20:
            return None, None, None
        match_img = cv2.drawMatchesKnn(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        cv2.putText(
            match_img,
            f"Matched Features: {len(good_matches)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return kp1_array, kp2_array, match_img

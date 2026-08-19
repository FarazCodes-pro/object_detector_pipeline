import cv2
def segment_frame(frame,
                  gaussian_kernel=(5, 5),
                  gaussian_sigma=0,
                  adaptive_block_size=11,
                  adaptive_C=2,
                  morph_kernel=(5, 5),
                  min_contour_area=500):
    """
    Apply Gaussian blur, adaptive threshold, morphological closing,
    then extract contours filtered by area.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian filter
    blur = cv2.GaussianBlur(gray, gaussian_kernel, gaussian_sigma)
    # Adaptive threshold matrix
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_C
    )
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            filtered_contours.append({
                "contour": cnt,
                "bbox": [x, y, w, h],
                "centroid": (cx, cy),
                "area": area
            })

    return morph, filtered_contours
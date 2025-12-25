import cv2
import numpy as np

data = np.array([
    [377, 117, 463, 117, 465, 130, 378, 130],
    [493, 115, 519, 115, 519, 131, 493, 131],
    [374, 155, 409, 155, 409, 170, 374, 170],
    [492, 151, 551, 151, 551, 170, 492, 170],
    [376, 198, 422, 198, 422, 212, 376, 212],
    [494, 190, 539, 189, 539, 205, 494, 206],
    [374, 1, 494, 0, 492, 85, 372, 86]
])

data = np.array([[367,87,426,86,433,140,375,141],
                [381,212,431,217,434,240,384,236],
                [386,261,447,265,450,287,389,283],
                [393,286,446,291,447,311,393,306],
                [446,328,490,334,491,360,447,355],
                [398,368,457,375,458,398,399,391],
                [459,375,502,380,502,398,459,394],
                [394,325,445,328,445,357,399,353]
])

# 1. Create the canvas (Height, Width)
canvas = np.zeros((720, 1280, 3), dtype="uint8")

# 2. Iterate through each row in data
for row in data:
    # Reshape the row (8 elements) into 4 pairs of (x, y) coordinates
    # We use .astype(np.int32) because fillPoly requires integer types
    pts = row.reshape((-1, 1, 2)).astype(np.int32)
    
    # 3. Use fillPoly to draw the filled box
    color = (0, 255, 0)  # Green
    cv2.fillPoly(canvas, [pts], color)

# 4. Display the result
cv2.imshow("Drawing", canvas)
cv2.waitKey(0) 
cv2.destroyAllWindows()
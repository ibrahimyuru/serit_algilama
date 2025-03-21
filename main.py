import cv2
import numpy as np

def cizgi_birlestirme(lines):
    if not lines:
        return None

    x_min, y_min, x_max, y_max = lines[0]

    for x1, y1, x2, y2 in lines:
        x_min, y_min = min((x_min, y_min), (x1, y1), (x2, y2))
        x_max, y_max = max((x_max, y_max), (x1, y1), (x2, y2))

    return x_max, y_max, x_min, y_min

def kesisim_nokta(cizgi1, cizgi2):
    x1, y1, x2, y2 = cizgi1
    x3, y3, x4, y4 = cizgi2

    #ax+by=c
    a1, b1, c1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    a2, b2, c2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3

    det = a1 * b2 - a2 * b1

    # det 0 ise paralel
    if det == 0:
        return None

    #[a1,b1//a2,b2] x [x//y] = [c1//c2]

    #cramer metodu
    x, y = (c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det
    return int(x), int(y)
def serit(impath):
    image = cv2.imread(impath)

    if image is None:
        return None
    son = np.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape

    left_mask = np.zeros_like(edges)
    #sol alt sağ alt sağ üst sol üst
    left_polygon = np.array(
        [[(width * 0.16, height),
          (width * 0.5, height),
          (width * 0.5, height * 0.7),
          (width * 0.4, height * 0.7)]],
        np.int32)

    cv2.fillPoly(left_mask, [left_polygon], 255)

    right_mask = np.zeros_like(edges)
    # sol alt sağ alt sağ üst sol üst
    right_polygon = np.array(
        [[(width * 0.5, height),
          (width * 0.84, height),
          (width * 0.6, height * 0.7),
          (width * 0.5, height * 0.7)]],
        np.int32)

    cv2.fillPoly(right_mask, [right_polygon], 255)
    
    left_masked = cv2.bitwise_and(edges, left_mask)
    right_masked = cv2.bitwise_and(edges, right_mask)

    left_lines, right_lines = [], []

    lines = cv2.HoughLinesP(
        left_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=100,
        maxLineGap=100
    )
    #eğim sol için + sağ için -

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            egim = (y2 - y1) / (x2 - x1)
            if egim < 0:
                left_lines.append(line[0])
    
    lines = cv2.HoughLinesP(
        right_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=100,
        maxLineGap=100
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            egim = (y2 - y1) / (x2 - x1)
            if egim > 0:
                right_lines.append(line[0])

    left_line, right_line = cizgi_birlestirme(left_lines), cizgi_birlestirme(right_lines)

    cv2.line(son, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 8)
    cv2.line(son, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 8)

    if left_line is not None and right_line is not None:
        kesisim = kesisim_nokta(left_line, right_line)
        if kesisim is not None:
            cv2.circle(son, kesisim, 15, (255, 0, 0), 8)
            cv2.line(son, kesisim, (960, 1080), (255, 0, 0), 8)
            sapma = (kesisim[0] - 960) / (1080 - kesisim[1])
            yon = f"{'sol' if sapma < 0 else 'sag'} {sapma:.4f}"
            cv2.putText(son, yon, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

    cv2.imwrite("output.png", son)

if __name__ == '__main__':
    serit("input.png")

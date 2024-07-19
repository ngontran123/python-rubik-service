import numpy as np
import twophase.solver as sv
import twophase.performance as pf
from typing import Union, List
import cv2
import random
import operator


async def rubikSolver(facelets: Union[str, None] = None) -> str:
    print('facelets is:' + facelets)
    print(len(facelets))
    res = sv.solve(facelets, 20, 3)
    print(f'solution is:{res}')
    return res


async def rubikSolveTo(original_cube: str, facelets: str) -> str:
    res = sv.solveto(original_cube, facelets, 50, 2)
    return res


async def convertColor(color: str):
    color_convert = \
    {
        'Red': 'R',
        'Green': 'F',
        'Blue': 'B',
        'Yellow': 'D',
        'White': 'U',
        'Orange': 'L'
    }
    return color_convert.get(color, '')


def processed_image(img_path: str):
    #image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img = cv2.imread(img_path)
    if img is None:
        print('No image')
    img = processBrightness(img)
    # cv2.imshow("image_process", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edge = cv2.Canny(blurred, 30, 60, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilatedFrame = cv2.dilate(edge, kernel)
    thresh = cv2.threshold(edge, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    return img, edge, grey


def find_square(edged_image):
    contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    pos_x = []
    post_y = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)
        print("approx:", len(approx))
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            squareness = float(w) / h
            pos_x.append(x)
            post_y.append(y)
            #print('square:', squareness)
            squares.append(approx)

    return squares, (pos_x, post_y)


def find_square_temp(img_path):
    img = cv2.imread(img_path)

    final_contours = []

    if img is None:
        print('No image')
    img = processBrightness(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decrease_noise = cv2.fastNlMeansDenoising(grey, 10, 15, 5, 21)
    blurred = cv2.GaussianBlur(decrease_noise, (3, 3), 0)
    edge = cv2.Canny(blurred, 30, 60, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilatedFrame = cv2.dilate(edge, kernel)
    thresh = cv2.threshold(edge, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(dilatedFrame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours

    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.1 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            #print('this approx here:', approx)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                final_contours.append((x, y, w, h))
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow('cutted contour', img[y:y + h, x:w + x])
            #print('Average color (BGR): ', np.array(cv2.mean(img[y:y + h, x:x + w])).astype(np.uint8))
            #cv2.waitKey(0)
    if len(final_contours) < 9:
        return []
    found = False
    contour_neighbors = {}
    for index, contour in enumerate(final_contours):
        (x, y, w, h) = contour
        contour_neighbors[index] = []
        center_x = x + w / 2
        center_y = y + h / 2
        radius = 1.5
        neighbor_positions = [
            # top left
            [(center_x - w * radius), (center_y - h * radius)],

            # top middle
            [center_x, (center_y - h * radius)],

            # top right
            [(center_x + w * radius), (center_y - h * radius)],

            # middle left
            [(center_x - w * radius), center_y],

            # center
            [center_x, center_y],

            # middle right
            [(center_x + w * radius), center_y],

            # bottom left
            [(center_x - w * radius), (center_y + h * radius)],

            # bottom middle
            [center_x, (center_y + h * radius)],

            # bottom right
            [(center_x + w * radius), (center_y + h * radius)],
        ]

        for neighbor in final_contours:
            (x2, y2, w2, h2) = neighbor
            for (x3, y3) in neighbor_positions:
                # The neighbor_positions are located in the center of each
                # contour instead of top-left corner.
                # logic: (top left < center pos) and (bottom right > center pos)
                if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                    contour_neighbors[index].append(neighbor)

    for (contour, neighbors) in contour_neighbors.items():
        if len(neighbors) == 9:
            found = True
            final_contours = neighbors
            break

    if not found:
        return []
    y_sorted = sorted(final_contours, key=lambda item: item[1])

    # Split into 3 rows and sort each row on the x-value.
    top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
    middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
    bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

    sorted_contours = top_row + middle_row + bottom_row

    color_list = draw_temp_contours(img, sorted_contours)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return color_list


def find_squares(img_path):
    img = cv2.imread(img_path)
    color_name_list =[]
    if img is None:
        print('No image')
    final_contours = []
    full_list = []
    img = processBrightness(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decrease_noise = cv2.fastNlMeansDenoising(grey, 10, 15, 5, 21)
    blurred = cv2.GaussianBlur(decrease_noise, (3, 3), 0)
    edge = cv2.Canny(blurred, 30, 60, 3)
    thresh = cv2.threshold(edge, 200, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours
    min_y = 1000
    set_color = []
    for cnt in cnts:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if (len(approx) == 4):
            area = cv2.contourArea(cnt)
            if area > 600 and area < 1200:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                full_list.append((x, y, w, h))
                list_data = [(x, y) for x, y, w, h in final_contours]
                tuple_check = (x, y)
                print('list data x:' + str(list_data))
                if tuple_check not in list_data:
                    if y < min_y and abs(min_y - y) > 2:
                        min_y = y
                    final_contours.append((x, min_y, w, h))
                else:
                    print('existed:' + str(tuple_check))
                #cv2.imshow('cutted contour', img[y:y + h, x:w + x])
                #print('Average color (BGR): ', np.array(cv2.mean(img[y:y + h, x:x + w])).astype(np.uint8))
                # colors = img[y + 7:y + h - 7, x + 14:x + w - 14]
                # dominant_color = get_dominant_color(colors)
                # if dominant_color not in set_color:
                #     set_color.append(dominant_color)

    print('Full List:', str(full_list))
    for (x, y, w, h) in final_contours:
        print('Value y here', x, y)

    contours_sorted = sorted(final_contours, key=lambda item: item[1])

    top_row = sorted((contours_sorted[0:3]), key=lambda item: item[0])
    middle_row = sorted((contours_sorted[3:6]), key=lambda item: item[0])
    bottom_row = sorted((contours_sorted[6:9]), key=lambda item: item[0])

    sorted_contours = top_row + middle_row + bottom_row

    for ((x2, y2, w2, h2)) in sorted_contours:
        print('Value y2 here:', y2)

    for index, (x, y, w, h) in enumerate(sorted_contours):
        color = img[y:y + h, x:x + w]
        dominant_color = get_dominant_color(color)
        if (dominant_color not in set_color):
            color_name = detect_color(dominant_color)
            if color_name is not None:
                color_name_list.append((color_name))
            print('Dominant color detected', dominant_color)
            print('Color Name:', color_name)

    #cv2.waitKey(0)
    # for color in set_color:
    #     color_name = detect_color(color)
    #     print('Dominant color here is:', color)
    #     print('Color name here is:', color_name)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return color_name_list


def remove_background(image):
    hh, ww = image.shape[:2]
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(image, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    morp = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    mask = 255 - morp

    img_res = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('new_image', img_res)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return img_res


def draw_temp_contours(image, contours):
    color_name_list = []
    for index, (x, y, w, h) in enumerate(contours):
        cv2.rectangle(image, (x, y), (x + w, y + h), (250, 35, 12), 2)
        #print('Average color(BGR): ', np.array(cv2.mean(image[y+7:y + h-7, x+14:x + w-14])).astype(np.uint8))
        colors = image[y + 8:y + h - 8, x + 16:x + w - 16]
        dominant_colors = get_dominant_color(colors)
        color_name = detect_color(dominant_colors)
        if color_name is not None:
            color_name_list.append(color_name)
        print('Dominant color: ', dominant_colors)
        print('Color for this contour is:', color_name)
    return color_name_list


def get_dominant_color(colors):
    pixels = np.float32(colors.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.3)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    #print(f"Dominant: {dominant}")

    return tuple(dominant)


# def get_closest_color(bgr):
#         """
#         Get the closest color of a BGR color using CIEDE2000 distance.
#
#         :param bgr tuple: The BGR color to use.
#         :returns: dict
#         """
#         lab = bgr2lab(bgr)
#
#         distances = []
#         for color_name, color_bgr in self.cube_color_palette.items():
#             distances.append({
#                 'color_name': color_name,
#                 'color_bgr': color_bgr,
#                 'distance': ciede2000(lab, bgr2lab(color_bgr))
#             })
#         print(f'Color distances: {distances}')
#         closest = min(distances, key=lambda item: item['distance'])
#         return closest


#
# def bgr2lab(inputColor):
#     """Convert BGR to LAB."""
#     # Convert BGR to RGB
#     inputColor = (inputColor[2], inputColor[1], inputColor[0])
#
#     num = 0
#     RGB = [0, 0, 0]
#
#     for value in inputColor:
#          value = float(value) / 255
#
#          if value > 0.04045:
#               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
#          else:
#               value = value / 12.92
#
#          RGB[num] = value * 100
#          num = num + 1
#
#     XYZ = [0, 0, 0,]
#
#     X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
#     Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
#     Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
#     XYZ[ 0 ] = round( X, 4 )
#     XYZ[ 1 ] = round( Y, 4 )
#     XYZ[ 2 ] = round( Z, 4 )
#
#     XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047            # ref_X =  95.047    Observer= 2°, Illuminant= D65
#     XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0             # ref_Y = 100.000
#     XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883          # ref_Z = 108.883
#
#     num = 0
#     for value in XYZ:
#
#          if value > 0.008856:
#               value = value ** ( 0.3333333333333333 )
#          else :
#               value = ( 7.787 * value ) + ( 16 / 116 )
#
#          XYZ[num] = value
#          num = num + 1
#
#     Lab = [0, 0, 0]
#
#     L = ( 116 * XYZ[ 1 ] ) - 16
#     a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
#     b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )
#
#     Lab [ 0 ] = round( L, 4 )
#     Lab [ 1 ] = round( a, 4 )
#     Lab [ 2 ] = round( b, 4 )
#
#     return Lab

def detect_color(bgr):
    b, g, r = bgr
    if b > 150 and g < 130 and r < 100:
        return 'Blue'
    elif b < 130 and g > 150 and r < 100:
        return 'Green'
    elif b < 130 and g < 100 and r > 150:
        return 'Red'
    elif b < 100 and g > 128 and r > 128:
        return 'Yellow'
    elif b < 100 and (64 < g < 165) and r > 128:
        return 'Orange'
    elif b > 200 and g > 200 and r > 200:
        return 'White'
    else:
        return 'Unknown'


def draw_squares(image, squares, pos_x, post_y, grey):
    count = 0
    print(len(squares))
    for square in squares:
        #print(f"xy:{pos_x[count]}-{post_y[count]}")
        count += 1
        mask = np.zeros_like(grey, dtype=np.uint8)
        cv2.drawContours(image, [square], -1, (0, 255, 0), 2)
        mean_color = cv2.mean(image, mask)[:3]
        #print("Mean color:", mean_color)
    return image


def handleRectangle(image_path):
    val = find_square_temp(image_path)
    print('value of val here is:', ''.join(val))
    modified_val = ''.join(val)
    if modified_val is '':
        print('did here')
        find_squares(image_path)
    return val
    # original_img, edged_img, grey = processed_image(image_path)
    # squares, (xs, ys) = find_square(edged_img)
    # result_img = draw_squares(original_img, squares, xs, ys, grey)
    # cv2.imshow("Square detected", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def processBrightness(image):
    brightness = 5
    contrast = 1.5
    img2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    return img2


async def maskImage(hsv_image, color_range):
    try:
        lower_bound, upper_bound = color_range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        return mask
    except Exception as e:
        print(e)

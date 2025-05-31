import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode="ltr"):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append(
            [box[1], min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0]
        )  # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7] == 0]) > 0:
        box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
        # new group
        if len([box for box in box_group if box[7] == current_group]) == 0:
            box_group0[0][7] = current_group  # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (
                    min_gx <= box[2] <= max_gx
                )
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (
                    min_gy <= box[4] <= max_gy
                )
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box == False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ""
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [
                box for box in current_box_group if box[6] < highest + 0.4 * mean_height
            ]
            # get the far left
            if mode == "ltr":
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left:
                        best_box = box
            elif mode == "rtl":
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right:
                        best_box = box
            text += " " + best_box[0]
            current_box_group.remove(best_box)

        result.append(
            [
                [
                    [min_gx, min_gy],
                    [max_gx, min_gy],
                    [max_gx, max_gy],
                    [min_gx, max_gy],
                ],
                text[1:],
            ]
        )

    return result

def get_location(raw_result):
    result = []
    for i in raw_result:
        location = ""
        center_x = i[0][0][0] + i[0][1][0] + i[0][2][0] + i[0][3][0] / 4
        center_y = i[0][0][1] + i[0][1][1] + i[0][2][1] + i[0][3][1] / 4

        if center_y < 0.3:
            location = location + "bottom"
        elif center_y < 0.7:
            location = location + "center"
        else:
            location = location + "top"

        if center_x < 0.3:
            location = location + " left"
        elif center_y < 0.7:
            location = location
        else:
            location = location + " right"
        result.append([i[1], location])

    return result
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(
    use_angle_cls=True, lang="en"
)  # need to run only once to download and load model into memory
img_path = "path/to/codes/MultimodalOCR/data/ocrVQA/images/177046025X.jpg"
result = ocr.ocr(img_path, cls=True)
img = Image.open(img_path)
w = img.width  # 图片的宽
h = img.height  # 图片的高
res = result[0]
# if res is not None:
#     for i in range(len(res)):
#         res[i][0][0][0] = round(res[i][0][0][0] / w, 3)
#         res[i][0][0][1] = round(res[i][0][0][1] / h, 3)
#         res[i][0][1][0] = round(res[i][0][1][0] / w, 3)
#         res[i][0][1][1] = round(res[i][0][1][1] / h, 3)
#         res[i][0][2][0] = round(res[i][0][2][0] / w, 3)
#         res[i][0][2][1] = round(res[i][0][2][1] / h, 3)
#         res[i][0][3][0] = round(res[i][0][3][0] / w, 3)
#         res[i][0][3][1] = round(res[i][0][3][1] / h, 3)
#         content_tp = res[i][1][0]
#         score_tp = round(res[i][1][1], 3)
#         res[i][1] = [content_tp, score_tp]

# res = result[0]
if res is not None:
    for i in range(len(res)):
        content_tp = res[i][1][0]
        score_tp = round(res[i][1][1], 3)
        res[i][1] = content_tp
para = get_paragraph(res)
para_one = "\n".join([pa[1] for pa in para])

# location = get_location(res)
# query_ocr_string = [str(i) for i in location]
# para_loc = "\n".join(query_ocr_string)
for line in res:
    print(line)
print(para_one)

# # 显示结果
# from PIL import Image

# result = result[0]
# image = Image.open(img_path).convert("RGB")
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(
#     image, boxes, txts, scores, font_path="path/to/codes/LLaVA/data/simfang.ttf"
# )
# im_show = Image.fromarray(im_show)
# im_show.save("./data/result2.jpg")

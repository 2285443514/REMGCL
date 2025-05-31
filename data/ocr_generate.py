from paddleocr import PaddleOCR
from PIL import Image
import os
import json


def paddleocr_pipeline(directory):
    # 获取目录下的所有文件和子目录
    file_list = os.listdir(directory)
    
    ocr_json = []
    file_count = 1
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
    # 遍历文件和子目录
    for file_item in file_list:
        # 使用 os.path.join() 构建文件的完整路径
        full_path = os.path.join(directory, file_item)
        
        # 判断是否是文件
        if os.path.isfile(full_path):
            # 获取文件名
            file_name = os.path.basename(full_path)
            file_name_only, file_extension = os.path.splitext(file_name)
            
            result = ocr.ocr(full_path)
            
            img = Image.open(full_path)
            w = img.width       #图片的宽
            h = img.height      #图片的高
            res = result[0]
            if res is not None:
                for i in range(len(res)):
                    res[i][0][0][0] = round(res[i][0][0][0]/w, 3)
                    res[i][0][0][1] = round(res[i][0][0][1]/h, 3)
                    res[i][0][1][0] = round(res[i][0][1][0]/w, 3)
                    res[i][0][1][1] = round(res[i][0][1][1]/h, 3)
                    res[i][0][2][0] = round(res[i][0][2][0]/w, 3)
                    res[i][0][2][1] = round(res[i][0][2][1]/h, 3)
                    res[i][0][3][0] = round(res[i][0][3][0]/w, 3)
                    res[i][0][3][1] = round(res[i][0][3][1]/h, 3)
                    content_tp = res[i][1][0]
                    score_tp = round(res[i][1][1], 3)
                    res[i][1] = [content_tp,score_tp]


            res_json = {
                "id": file_name_only,
                "image": file_name,
                "ocr" : res
            }
            ocr_json.append(res_json)
            print(str(file_count)+" / "+str(len(file_list)))
            file_count += 1
    
    # 指定保存 JSON 的文件路径
    json_file_path = 'path/to/data/LLaVAR/pretrain/images/ocr_llavar_pretrain.json'

    # 将数组转换为 JSON 格式的字符串
    json_data = json.dumps(ocr_json, indent=2)  # indent参数是为了可读性，它会添加缩进

    # 将 JSON 数据写入文件
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

    print(f"JSON 数据已写入文件: {json_file_path}")

# 指定要遍历的文件夹路径
directory_path = 'path/to/data/LLaVAR/pretrain/images'
# 调用函数进行遍历
paddleocr_pipeline(directory_path)


# 显示结果
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='path/to/codes/LLaVA/data/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
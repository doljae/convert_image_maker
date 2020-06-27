import cv2
import os
from pix2pix.pix2pix import *

class convertImageMaker:
  def __init__(self):
    self.box=[]
    self.abs_dir="./result_images"
    self.crop_location=str()
    self.convert_type=int()
    self.original_dir=""
    pass

  def image_extract(self):
    original_dir = 'image/result_images/original_image.jpg'
    self.original_dir=original_dir
    print(original_dir)
    img_ori = cv2.imread(original_dir, cv2.IMREAD_COLOR)
    # print(img_ori)
    img = img_ori.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 100, 200)
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
      cnt = contours[i]
      if i != len(contours) - 1:
        cnt2 = contours[i + 1]
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        rect_area2 = w2 * h2  # area size
      x, y, w, h = cv2.boundingRect(cnt)
      rect_area = w * h  # area size
      aspect_ratio = float(w) / h  # ratio = width/height
      if (aspect_ratio >= 0.1) and (aspect_ratio <= 10000.0) and (rect_area >= 500) and (rect_area <= 10000):
        if (0 <= abs(rect_area - rect_area2)) and (abs(rect_area - rect_area2) <= 20):
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
          self.box.append(cv2.boundingRect(cnt))

    for i in range(len(self.box)):  # Buble Sort on python
      for j in range(len(self.box) - (i + 1)):
        if self.box[j][0] > self.box[j + 1][0]:
          temp = self.box[j]
          self.box[j] = self.box[j + 1]
          self.box[j + 1] = temp
          pass
        pass
      pass
  def image_save_crop_location(self):
    crop_list = []
    img_ori = cv2.imread(self.original_dir, cv2.IMREAD_COLOR)
    img_cut = img_ori.copy()
    img_cut2 = img_ori.copy()
    crop_dir = './image/crop_images'
    crop_text = open(os.path.join(crop_dir, "crop_location.txt"), 'w')
    for i in range(len(self.box)):
      crop_img = img_ori[self.box[i][1]:self.box[i][1] + self.box[i][3], self.box[i][0]:self.box[i][0] + self.box[i][2]]
      data = str(self.box[i][1]) + " " + str(self.box[i][3]) + " " + str(self.box[i][0]) + " " + str(self.box[i][2]) + "\n"
      crop_text.write(data)
      crop_list.append(crop_img)
      # 자른 이미지를 폴더에 저장
      cv2.imwrite(os.path.join(crop_dir, "crop_" + str(i) + ".jpg"), crop_img)
      img_cut[self.box[i][1]:self.box[i][1] + self.box[i][3], self.box[i][0]:self.box[i][0] + self.box[i][2]] = 0
      img_cut2[self.box[i][1]:self.box[i][1] + self.box[i][3], self.box[i][0]:self.box[i][0] + self.box[i][2]] = 255
      cv2.imwrite(os.path.join(crop_dir, "cut_image1.jpg"), img_cut)
      cv2.imwrite(os.path.join(crop_dir, "cut_image2.jpg"), img_cut2)
      pass
    crop_text.close()
    self.crop_location=os.path.join(crop_dir, "crop_location.txt")
  def image_convert(self, convert_type):
    dir = 'image/crop_images/'
    img_ori = cv2.imread(self.original_dir, cv2.IMREAD_COLOR)
    img_cut1 = cv2.imread(os.path.join(dir, "cut_image1.jpg"), cv2.IMREAD_COLOR)
    crop_text = open(os.path.join(dir, "crop_location.txt"), 'r')
    lines = crop_text.readlines()
    # 뒤집기
    if convert_type == 1:
      dirs = os.listdir(dir)
      for item in dirs:
        if os.path.isfile(dir + item) and ".jpg" in item and "crop" in item:
          item_index = item.replace("crop_", "")
          item_index = int(item_index.replace(".jpg", ""))
          img_paste = cv2.imread(dir + item, cv2.IMREAD_COLOR)
          height, width, channel = img_paste.shape
          matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 180, 1)
          dst = cv2.warpAffine(img_paste, matrix, (width, height))

          crop_location = lines[item_index].split()
          crop_location = list(map(int, crop_location))
          img_cut1[crop_location[0]:crop_location[0] + crop_location[1],
          crop_location[2]:crop_location[2] + crop_location[3]] = dst
          pass
        pass
      dst2 = cv2.resize(img_cut1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      cv2.imwrite(os.path.join("./image/result_images/converted_image.jpg"), dst2)
      pass
    # 색 반전
    elif convert_type == 2:
      dirs = os.listdir(dir)
      for item in dirs:
        if os.path.isfile(dir + item) and ".jpg" in item and "crop" in item:
          # print(item)
          item_index = item.replace("crop_", "")
          item_index = int(item_index.replace(".jpg", ""))
          img_paste = cv2.imread(dir + item, cv2.IMREAD_COLOR)
          dst = cv2.bitwise_not(img_paste)

          crop_location = lines[item_index].split()
          crop_location = list(map(int, crop_location))
          # print(crop_location)
          # print(img_paste.shape)
          # print(crop_location[0], crop_location[0] + crop_location[1], crop_location[2], crop_location[2] + crop_location[3])
          img_cut1[crop_location[0]:crop_location[0] + crop_location[1],
          crop_location[2]:crop_location[2] + crop_location[3]] = dst
          pass
        pass

      # cv2.imshow("test",img_cut1)
      # cv2.waitKey()
      dst2 = cv2.resize(img_cut1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      # cv2.imshow("test",dst2)
      # cv2.waitKey()
      cv2.imwrite(os.path.join("./image/result_images/converted_image.jpg"), dst2)
      pass

    # 다른 오브젝트로 대체
    elif convert_type == 3:
      img_ori = cv2.imread(os.path.join("image/result_images/original_image.jpg"), cv2.IMREAD_COLOR)
      # cv2.imshow("test", img_ori)
      # cv2.waitKey()
      # 잘려진 이미지를 가져옴
      img_crop = cv2.imread(os.path.join(dir, "cut_image1.jpg"), cv2.IMREAD_COLOR)
      # 잘려진 이미지에서 채울 부분의 좌표관련 내용을 읽어옴
      crop_text = open(os.path.join(dir, "crop_location.txt"), 'r')

      line_len = 0
      lines = []
      while True:
        line1 = crop_text.readline()
        if not line1: break
        line_len = line_len + 1
        lines.append(line1)
        pass
      crop_text.close()

      # 채울 이미지의 리스트를 가져옴
      dir2 = '.image/clean_back'
      print(os.path.join(dir2, "apple.png"))
      img_paste = cv2.imread("./image/clean_back/apple.png", cv2.IMREAD_COLOR)
      # cv2.imshow("test", img_paste)
      # cv2.waitKey()

      for i in range(0, line_len):
        crop_location = lines[i].split()
        crop_location = list(map(int, crop_location))
        # 채울 이미지를 채울 부분의 사이즈로 변환함
        img_paste = cv2.resize(img_paste, dsize=(crop_location[3], crop_location[1]),interpolation=cv2.INTER_AREA)
        # cv2.imshow("test", img_paste)
        # cv2.waitKey()
        # cv2.imshow("test", img_ori)
        # cv2.waitKey()
        y_offset = crop_location[0]
        x_offset = crop_location[2]
        y1, y2 = y_offset, y_offset + img_paste.shape[0]
        x1, x2 = x_offset, x_offset + img_paste.shape[1]
        alpha_s = img_paste[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s
        # 원본 이미지에 채울 이미지를 붙임
        for c in range(0, 3):
          img_ori[y1:y2, x1:x2, c] = (alpha_s * img_paste[:, :, c] + alpha_l * img_ori[y1:y2, x1:x2, c])
          pass
        # 잘려진 이미지에 채울 이미지를 붙임
        '''
        for c in range(0, 3):
            img_ori[y1:y2, x1:x2, c] = (alpha_s * img_paste[:, :, c] + alpha_l * img_ori[y1:y2, x1:x2, c])
            pass
        pass
        '''
      dst2 = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      cv2.imwrite(os.path.join("./image/result_images/converted_image.jpg"), dst2)
      pass

    # GAN, pix2pix
    elif convert_type == 4:
      print("이미지합성시작")
      pix = pixStart()
      pix.start()
      dir = './pix2pix/datasets/tmp/saved'
      dirs = os.listdir(dir)
      for item in dirs:
        if os.path.isfile(os.path.join(dir, item)) and ".jpg" in item and "crop" in item:
          # print(item)
          item_index = item.replace("crop_", "")
          item_index = int(item_index.replace(".jpg", ""))
          img_paste = cv2.imread(os.path.join(dir, item), cv2.IMREAD_COLOR)

          crop_location = lines[item_index].split()
          crop_location = list(map(int, crop_location))
          # print(crop_location)
          # print(img_paste.shape)
          # print(crop_location[0], crop_location[0] + crop_location[1], crop_location[2], crop_location[2] + crop_location[3])
          img_cut1[crop_location[0]:crop_location[0] + crop_location[1],
          crop_location[2]:crop_location[2] + crop_location[3]] = img_paste
          pass
        pass
      dst2 = cv2.resize(img_cut1, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      cv2.imwrite(os.path.join("./image/result_images/converted_image.jpg"), dst2)
      print("이미지합성끗")
      pass
    # Mix
    elif convert_type == 5:
      # dir = 'crop/'
      # img_ori = cv2.imread("original_image.jpg", cv2.IMREAD_COLOR)
      # img_cut1 = cv2.imread(os.path.join(dir, "cut_image1.jpg"), cv2.IMREAD_COLOR)
      # crop_text = open(os.path.join(dir, "crop_location.txt"), 'r')
      # lines = crop_text.readlines()
      pass
    pass
  #
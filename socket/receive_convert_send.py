import socketserver
import socket
import datetime
import base64
import numpy as np
import cv2
from image.image_handle import convertImageMaker
import os
from pix2pix import pix2pix
import random


class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        print("get....")
        image1 = []
        try:
            while True:
                data = self.request.recv(82100)  # 클라이언트가보낸데이터를가져옵니다
                # print('data,', data)
                # self.andRaspTCP.sendAll("321\n")
                # data = base64.b64decode(data)
                if not data or len(data) == 0:
                    break
                image1.extend(data)
            print("get over")
            image = np.asarray(bytearray(image1), dtype="uint8")
            # print("1", image)
            # print("2",len(image))  39559
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite("./original_image.jpg", image)
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print("받았습니다")
            # self.request.sendall("get your connet!".encode("utf-8"))

        except Exception:
            print(self.client_address, "연결해제")
        finally:
            self.request.close()  # 예외 후 연결을 닫습니다

    # before handle,연결설정：
    def setup(self):
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time)
        print("연결설정：", self.client_address)

    # finish run  after handle
    def finish(self):
        print("수신 완료, 연결해제")
        # 수신을 받고 7777포트에서 통신이 끝나면 7778 포트로 새로 염
        # 동일 포트의 소켓을 닫아도 일정 시간 동안 소켓이 남아있기 때문에 포트번호를 다르게 해줘서 충돌을 피함
        print("송신 준비, 연결설정")

        convertTest=convertImageMaker()

        # convertTest.imageConvert(convert_type=1)
        # convertTest.imageConvert(convert_type=2)
        # convertTest.imageConvert(convert_type=3)
        # convertTest.imageConvert(convert_type=4)
        # convertTest.imageConvert(convert_type=5)
        convertTest.image_extract()
        convertTest.image_save_crop_location()
        convertTest.image_convert(convert_type=random.randint(1,5))
        print("변환이미지송신")

        HOST = '192.168.0.2'
        PORT = 7778
        ADDR = (HOST, PORT)
        BUFSIZE = 4096
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        filename = "converted_image.jpg"

        # Bind Socket
        serv.bind(ADDR)
        serv.listen(5)
        conn, addr = serv.accept()
        print('client connected ... ', addr)

        # Open the file
        # Read and then Send to Client

        f = open(filename, 'rb')  # open file as binary
        data = f.read()
        # print(data, ',,,')
        exx = conn.sendall(data)
        # print(exx, '...')
        f.flush()
        f.close()

        # Close the Socket
        print('finished writing file')
        conn.close()
        serv.close()
        print("송신 완료, 연결 해제")
        print("")

        print("=============================== text 전송 ==================================")

        print("송신 준비, 연결설정")

        HOST = '192.168.0.2'
        PORT = 7779
        ADDR = (HOST, PORT)
        BUFSIZE = 4096
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dir='./crop'
        filename = "crop_location.txt"

        # Bind Socket
        serv.bind(ADDR)
        serv.listen(5)
        conn, addr = serv.accept()
        print('client connected ... ', addr)

        # Open the file
        # Read and then Send to Client

        f = open(os.path.join(dir,filename), 'rb')
        data = f.read()
        print(data, ',,,')
        exx = conn.sendall(data)
        print(exx, '...')
        f.flush()
        f.close()

        # Close the Socket
        print('finished writing file')
        conn.close()
        serv.close()
        print("송신 완료, 연결 해제")
        print("")

if __name__ == "__main__":
    HOST, PORT = "192.168.0.2", 7777
    # server=socketserver.TCPServer((HOST,PORT),MyTCPHandler)  # 매개 변수를 전달하는 인스턴스 객체

    # 멀티스레드
    server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    server.serve_forever()  # 계속연결
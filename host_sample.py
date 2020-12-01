import socket
import cv2
import numpy
from queue import Queue
from _thread import *

enclosure_queue = Queue()

video_path = 'testing/test2.mov'
# video_path = 1
HOST = '127.0.0.1'
# HOST = '143.248.229.64'
# HOST = '143.248.229.3'
PORT = 9999

video_ended = False


def threaded(client_socket_, addr_, queue):
    global video_ended
    print('Connected by :', addr_[0], ':', addr_[1])
    while True:
        try:
            data = client_socket_.recv(1024)
            if video_ended:
                break
            if not data:
                print('Disconnected by ' + addr_[0], ':', addr_[1])
                break
            string_data = queue.get()
            client_socket_.send(str(len(string_data)).ljust(16).encode())
            client_socket_.send(string_data)
        except ConnectionResetError:
            print('Disconnected by ' + addr_[0], ':', addr_[1])
            break
    client_socket_.close()


def web_cam(queue):
    while True:
        capture = cv2.VideoCapture(video_path)

        while True:
            ret, frame = capture.read()
            if ret is False:
                break

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            result, img_encode = cv2.imencode('.jpg', frame, encode_param)

            data = numpy.array(img_encode)
            string_data = data.tostring()

            queue.put(string_data)

            cv2.imshow('hosted image', frame)

            key = cv2.waitKey(500)
            if key == 27:
                break

    global video_ended
    video_ended = True


if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print('server start')

    start_new_thread(web_cam, (enclosure_queue,))

    while True:
        print('wait')
        if video_ended:
            break
        client_socket, addr = server_socket.accept()
        start_new_thread(threaded, (client_socket, addr, enclosure_queue,))

    server_socket.close()
    print('end')

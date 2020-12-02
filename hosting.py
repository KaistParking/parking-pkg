import socket
import cv2
import numpy
from _thread import start_new_thread
import time
import argparse
from tqdm import tqdm

is_video_ended = True
img_string_flow = None


# 쓰레드 함수
def hosting(client, address):
    global img_string_flow, is_video_ended

    print('Connected by : {} : {}'.format(address[0], address[1]))
    while not is_video_ended:
        try:
            data = client.recv(1024)
            if not data:
                print('Disconnected by : {} : {}'.format(
                    address[0], address[1]))
                break
            if is_video_ended:
                print('video ended')
                return

            img_string = img_string_flow
            client.send(str(len(img_string)).ljust(16).encode())
            client.send(img_string)

        except ConnectionResetError as e:
            print(e)
            print('{}: Disconnected by : {} : {}'.format(
                e, address[0], address[1]))
            break

    client.close()
    print('client closed ({})'.format(address[0]))


def web_cam(video_id='0'):
    global img_string_flow, is_video_ended

    is_video_ended = False
    cap = cv2.VideoCapture(int(video_id) if video_id.isdigit() else video_id)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    if not video_id.isdigit():
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(video_length), total=video_length, desc=video_id):
            ret, frame = cap.read()
            if ret is False:
                print('video ended')
                return
            result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
            img_string_flow = numpy.array(img_encoded).tostring()
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        while True:
            ret, frame = cap.read()
            if ret is False:
                print('image read failed... check camera connection')
                return
            result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
            img_string_flow = numpy.array(img_encoded).tostring()
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    is_video_ended = True
    cap.release()


def main(host='127.0.0.1', port=9999, video=0):
    global is_video_ended
    thread_list = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(2)

    print('Image hosting start')
    print('HOST: {}, PORT: {}'.format(host, port))

    thread_list.append(start_new_thread(web_cam, (video, )))
    time.sleep(1)

    while not is_video_ended:
        print('wait')
        client_socket, client_address = server_socket.accept()
        if is_video_ended:
            print('video ended')
            break
        thread_list.append(start_new_thread(
            hosting, (client_socket, client_address, )))

    server_socket.close()
    print('server socket closed')

    for thread in thread_list:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image hosting')
    parser.add_argument('--video-path', type=str,
                        default='0', help='video file path')
    args = parser.parse_args()

    HOST = socket.gethostbyname(socket.gethostname())
    PORT = 9999

    video_path = args.video_path

    main(HOST, PORT, video_path)

import socket
import numpy as np
import cv2
import bluetooth


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


class Client:
    def __init__(self, host='127.0.0.1', port=9999, use_bluetooth=True):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

        self.use_bluetooth = use_bluetooth
        if use_bluetooth:
            self.blue_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.blue_socket.connect(("98:d3:31:fd:3e:0d", 1))
            print("bluetooth connected")

    def receive(self):
        message = '1'
        self.client_socket.send(message.encode())

        length = recvall(self.client_socket, 16)
        string_data = recvall(self.client_socket, int(length))
        data = np.frombuffer(string_data, dtype='uint8')

        dec_img = cv2.imdecode(data, 1)
        return dec_img

    def send_bluetooth_control(self, control_in: str):
        self.blue_socket.send(control_in.encode(encoding="utf-8"))

    def close(self):
        self.client_socket.close()
        if self.use_bluetooth:
            self.blue_socket.close()


if __name__ == "__main__":

    HOST = '127.0.0.1'
    PORT = 9999

    client = Client(host=HOST, port=PORT, use_bluetooth=False)
    while True:
        img = client.receive()
        cv2.imshow('image', img)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    client.close()

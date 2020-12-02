import socket
import numpy as np
import cv2
import bluetooth
import os
import serial


def search_modem(basename='usbmodem'):
    files = os.listdir('/dev/')
    for file in files:
        if basename in file:
            return '/dev/{}'.format(file)
    return None


def receive_all(sock, count):
    buf = b''
    while count:
        new_buf = sock.recv(count)
        if not new_buf:
            return None
        buf += new_buf
        count -= len(new_buf)
    return buf


class Client:
    def __init__(self, host='127.0.0.1', port=9999, use_bt=False, basename='usbmodem', vis=True):
        # init host
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        self.vis = vis

        # vehicle connection
        self.use_bt = use_bt
        if not use_bt:
            self.modem = search_modem(basename=basename)
            if self.modem is None:
                print('no modem found')
                exit(0)
            else:
                print('modem: {}'.format(self.modem))
            self.serial = serial.Serial(
                port=self.modem,
                baudrate=9600,
            )
            print("serial connected")
        else:
            self.bt_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bt_socket.connect(("98:d3:31:fd:3e:0d", 1))
            print("bt connected")

    def receive(self):
        message = '1'
        self.client_socket.send(message.encode())

        length = receive_all(self.client_socket, 16)
        string_data = receive_all(self.client_socket, int(length))
        data = np.frombuffer(string_data, dtype='uint8')

        dec_img = cv2.imdecode(data, 1)
        if self.vis:
            cv2.imshow('received', dec_img)
            cv2.waitKey(1)
        return dec_img

    def send(self, msg: str):
        if self.use_bt:
            self.bt_socket.send(msg.encode('utf-8'))
        else:
            self.serial.write(msg.encode('utf-8'))

    def close(self):
        self.client_socket.close()
        if self.use_bt:
            self.bt_socket.close()


if __name__ == "__main__":

    HOST = '127.0.0.1'
    PORT = 9999

    client = Client(host=HOST, port=PORT, use_bt=False)
    while True:
        img = client.receive()
        cv2.imshow('received', img)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    client.close()

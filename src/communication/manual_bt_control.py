from bluetooth import *


client_socket = BluetoothSocket(RFCOMM)
client_socket.connect(("98:d3:31:fd:3e:0d", 1))
print("connected")

while True:
    msg = input("send :")
    print(msg)
    client_socket.send(msg.encode(encoding="utf-8"))

print("finished")

client_socket.close()

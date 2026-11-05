# command_sender.py
# TCP client used by the controller (Pi 1) to send commands to Pi 2.

#code for sending commands
#doesnt listen but when the response is expected
#execute on pi 1

import socket
import pickle

class CommandSender:
    def __init__(self, target_ip, port=6001, buffer_size=1024):
        self.target_ip = target_ip
        self.port = port
        self.buffer_size = buffer_size
        self.sock = None

    def connect(self):
        """Connect to the target Pi."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.target_ip, self.port))
            print(f"[+] Connected to {self.target_ip}:{self.port}")
        except Exception as e:
            print(f"[!] Connection error: {e}")

    def send(self, command, expect_response=False):
        """Send a command. Optionally wait for a response."""
        if self.sock is None:
            raise RuntimeError("Socket not connected. Call connect() first.")

        try:
            # Add the response prefix if needed
            if expect_response:
                command = f"REQUIRES_RESPONSE:{command}"

            self.sock.sendall(command.encode())
            print(f"[Sent] {command}")

            if expect_response:
                response = pickle.loads(self.sock.recv(self.buffer_size))
                print(f"[Received] {response}")
                return response

        except Exception as e:
            print(f"[!] Error during send: {e}")
            return None

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            print("[+] Socket closed")

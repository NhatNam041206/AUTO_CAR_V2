import serial
import serial.tools.list_ports
import time

class Car:

    def __init__(self):
        self.connect_serial() # Factor : the previous position is on send_command(); In future, may need to adjust to continuously checking whether the process board connected esp.

    serial_conn = None
    def list_serial_ports(self):
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]

    def connect_serial(self):
        available_ports = self.list_serial_ports()
        port = '/dev/ttyUSB0' if '/dev/ttyUSB0' in available_ports else (available_ports[0] if available_ports else None)
        if not port:
            raise Exception("No serial ports found.")
        self.serial_conn = serial.Serial(port, baudrate=115200, timeout=1)
        return self.serial_conn


    def send_command(self, command: str):
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(f"{command}\n\r\n".encode())
            #response = self.serial_conn.read_until(expected=CRLF)
            #print(command)
    '''
    
    def forward(self, log = False):
        command = "mctl 70 70"
        self.send_command(command)
        time.sleep(1.5)
        command = "mctl 0 0"
        self.send_command(command)
        time.sleep(.5)
        if not (log):
            print("forward! \n")
    
    def right(self, log = False):
        command = "mctl 50 -50"
        self.send_command(command)
        time.sleep(1.3)
        command = "mctl 0 0"
        self.send_command(command)
        
        if not (log):
            print("forward! \n")
           ''' 


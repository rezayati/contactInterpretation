import serial
import time
import serial.tools.list_ports

# List available serial ports
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found port: {port.device}")

list_serial_ports()

# Replace '/dev/ttyACM0' with the correct device name if necessary
device = '/dev/ttyACM0'
baud_rate = 9600 #115200  # Adjust this to your device's baud rate

try:
    print(f"Trying to open serial port {device} at {baud_rate} baud rate")
    ser = serial.Serial(device, baud_rate, timeout=0.001)
    print(f"Serial port {device} opened successfully")

    while True:
        try:
            # Read a single byte from the serial device
            byte_data = ser.ser.readline().decode('utf-8').strip()
            if byte_data:
                print('Received byte:', byte_data)

            # Small sleep to avoid overwhelming the CPU
            time.sleep(0.001)
        except serial.SerialException as e:
            print(f'Error reading from serial port: {e}')
            break

except serial.SerialException as e:
    print(f'Serial error: {e}')
except KeyboardInterrupt:
    print("Exiting program")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(f'Serial port {device} closed')

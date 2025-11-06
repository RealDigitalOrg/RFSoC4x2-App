# RFSoC 4x2 ADC DAC Oscilloscope and Function Generator
Using the ADC and DAC stored in the RFSoC 4x2, we can build an oscilloscope and function generator. The ADC and DAC inside the board will generate and capture sine wave and work like an oscilloscope and function generator. Therefore, when needing a simple data collection or signal generation, this board can be a good choice then having two big machines by your side.

# Things to do before running the application
- Reference tutorial: https://github.com/realdigitalorg/rfsoc4x2-bsp

After acquiring the RfSoC 4x2 board, the MicroSD card that is supplied has the majority of the necessary files which will be used for its setup. 
First insert the MicroSD card into the Micro SD card slot on the board, and make sure the 
Boot switch is set to SD.
If you have problems with the SD card please refer to the referenced RFSoC4x2-BSP Github.
After the MicroSD boot configuration is setup, you will need to connect the Micro-USB cable supplied into the PROG-UART port located on the board and into your computer. The next steps will depend on your operating system.

# For Windows
You will need to bring up a terminal emulator such as Tera Term or Putty. For this example we used Tera Term. First, open Device manager, and locate the new COM port which appears when you plug in your board. Note: This COM value may change if you switch USB ports on your machine. For this COM port, (COM3 in our instance), open properties and change the Baud rate to 115200,  Data Bits to 8, Parity to None, Stop Bits to 1, and Flow control to None. From here, in your terminal emulator, set the conection to your COM port, and the same settings as described above. 
Now, Power on your board. if you configured it correctly, you should see the board boot up on the terminal as the default user xilinx@pynq.

# For Linux
Configure the Linux terminal with the exact same COM settings as the Windows version

# User Interface setup
To make visualization of the graphs possible, we decided to use xfce4 as our desktop environment, connected remotly through XRDP but any lightweight Linux desktop environment should work just fine.

# Install Xfce4
First you will need an ethernet connection to the board, from here use the following series of commands:
- sudo apt update
- sudo apt upgrade
- sudo apt install xfce4-goodies xfce4
  
https://linuxconfig.org/guide-to-installing-xfce-desktop-on-ubuntu-linux

# install XRDP
- sudo apt install xrdp
- sudo systemctl enable xrdp
- sudo systemctl start xrdp

# install basic packages

Run the following script to install the basic libraries used in the application:
- sudo apt-get update -y
- pip install Pyside6
- pip install PySide6 pyqtgraph
- 



# Accessing the hardware file
To access the hardware file, go to the /home/xilinx/pynq/overlays/base/base.bit file.
This file is the hardware file for the RFSoC 4x2 board.

# Running the application


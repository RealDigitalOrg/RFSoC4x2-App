# RFSoC 4x2 ADC DAC Oscilloscope and Function Generator
Using the ADC and DAC stored in the RFSoC 4x2, we can build an oscilloscope and function generator. The ADC and DAC inside the board will generate and capture sine wave and work like an oscilloscope and function generator. Therefore, when needing a simple data collection or signal generation, this board can be a good choice then having two big machines by your side.

Before installing necessary files, ethernet connection is necessary in order to download the files.

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


# Downloading the application
git clone this repository to your PYNQ environment

# install basic packages

Run the following script to install the basic libraries used in the application:
- chmod +x install_packages.sh
- ./install_packages.sh


# Running the application
Make the file executable using the chmod +x 
Run the Executable program. 


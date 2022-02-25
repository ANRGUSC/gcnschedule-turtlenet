#Change this to turtlebot number
IPADDR="192.168.7.2/24"


sudo service network-manager stop
sleep 1
sudo ip link set wlan0 down
sleep 1
sudo iwconfig wlan0 mode ad-hoc
sudo iwconfig wlan0 essid 'TurtleNet'
sleep 1
sudo ifconfig wlan0 $IPADDR netmask 255.255.255.0 up

#change this correct wireless interface(type iwconfig in your terminal)
WIRELESS_INTERFACE="wlx1cbfce96f922"

sudo ip link set $WIRELESS_INTERFACE down 

sudo iwconfig $WIRELESS_INTERFACE mode ad-hoc

sudo ip link set $WIRELESS_INTERFACE up

sudo ip addr add 192.168.7.10/24 dev $WIRELESS_INTERFACE

sudo iwconfig $WIRELESS_INTERFACE essid "TurtleNet"
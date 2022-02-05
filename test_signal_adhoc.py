import os
import re
import subprocess
import time

SYS_NET_PATH = '/sys/class/net'

def main():
    hz = 1
    dev = None
    num_tries = 5

    if not dev:
        wldevs = [d for d in os.listdir(SYS_NET_PATH) if d.startswith('wl') or d.startswith('wifi')]
        if wldevs:
            dev = wldevs[0]
        else:
            raise InputError("No wireless device found to monitor")
        print("Monitoring %s" % dev)

    try:
        ip_str = subprocess.check_output(['ip','addr','show',dev], stderr=subprocess.STDOUT)
        if re.search(b'^\s*inet\s', ip_str, re.MULTILINE):
            print("connected")
    except subprocess.CalledProcessError:
        print("not connected")

    avg = []

    for i in range(num_tries):
        try:
            wifi_str = subprocess.check_output(['iw', 'dev', dev, 'station', 'dump'], stderr=subprocess.STDOUT).decode()
            # print(wifi_str)
            fields_str = re.split('\n+', wifi_str)
            # print(fields_str)

            connections = {}

            for field_str in fields_str:
                if field_str.startswith('Station '):
                    stat = field_str
                    connections[stat] = {}
                else:
                    try:
                        key, value = re.split(':+', field_str)
                        key = re.split('\t+', key)[1]
                        value = re.split('\t+', value)[1]
                        connections[stat][key] = value
                    except:
                        pass

            # print(connections)

            for stat, stat_dict in connections.items():
                print(stat_dict)
                print(stat, stat_dict['signal'])
                avg.append(int(re.split(' +', stat_dict['signal'])[0]))

            time.sleep(1)

        except subprocess.CalledProcessError:
            print("error checking status")
            return -1

    print(sum(avg)/len(avg))
    return

if __name__=='__main__':
    import sys
    sys.exit(main())

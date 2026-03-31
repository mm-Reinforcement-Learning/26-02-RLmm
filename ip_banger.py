# send a bing to IP-adresses 
# ("167.71.60.74",
#  "142.93.167.231",
#  "103.125.177.250",
#  "27.5.87.138",
#  "138.197.123.21",
# "104.219.236.109 ",
# "98.159.37.99",
# "216.24.210.192",
# "98.159.37.92",
# "103.178.231.51",
# "104.219.236.61",
# "38.248.90.19",
# "104.244.78.206",
# "107.173.193.234",
# "115.187.36.2",
# "136.158.122.209",
# "122.173.225.15")

from threading import Thread
import os
import subprocess
import sys
from queue import Queue

repeat = 50000

num_threads = 4
queue = Queue()
ips = ["103.125.177.250", "27.5.87.138"]
# wraps system ping command
def pinger(i, q):
    """Pings subnet"""
    while True:
        ip = q.get()
        print("Thread %s: Pinging %s" % (i, ip))
        ping_flag = "-n" if sys.platform.startswith("win") else "-c"
        ret = subprocess.call(
            "ping %s 1 %s" % (ping_flag, ip),
            shell=True,
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
        if ret == 0:
            print("%s: is alive" % ip)
        else:
            print("%s: did not respond" % ip)
        q.task_done()


# Spawn thread pool
for j in range(repeat):
    print(j)
    for i in range(num_threads):

        worker = Thread(target=pinger, args=(i, queue))
        worker.daemon = True
        worker.start()


# Place work in queue
for ip in ips:
    queue.put(ip)


# Wait until worker threads are done to exit
queue.join()
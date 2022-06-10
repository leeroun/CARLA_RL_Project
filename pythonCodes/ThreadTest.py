from threading import Thread
import time


# Define a function for the thread
def print_time(threadName, delay):
    count = 0
    while count < delay:
        time.sleep(1)
        count += 1
        print("%s: %s" % (threadName, time.ctime(time.time())))
        if count < delay:
            count = 0


# Create two threads as follows
try:
    thread1 = Thread(target=print_time('thread1', 10))
    thread1.start()
    thread2 = Thread(target=print_time('thread2', 5))
    thread2.start()


except:
    print("Error: unable to start thread")

while 1:
    time.sleep(1)
    print("thread0: %s" % (time.ctime(time.time())))
    pass

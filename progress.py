import os
import sys

class Progress:
    def __init__(self, finishstring="done", infostring=None):
        self.finishstring = finishstring
        if infostring is not None:
            print infostring

    def _progress(self, val):
        cols = int(os.popen('stty size', 'r').read().split()[1])
        totlength = cols - len(self.finishstring) - 4
        current = int(round(val*totlength))
        sys.stdout.write("\r|")
        for i in range(current):
            sys.stdout.write("=")
        sys.stdout.write(">")
        for i in range(current,totlength):
            sys.stdout.write(" ")
        sys.stdout.write("|")

    def progress(self,val):
        self._progress(val)
        sys.stdout.write("%3d" % int(round(val*100)))
        sys.stdout.write("%")
        sys.stdout.flush()

    def done(self):
        self._progress(1.)
        print self.finishstring

if __name__ == "__main__":
    import time
    prg = Progress("done", "Testing")
    for i in range(10):
        prg.progress(float(i/10.))
        time.sleep(1)
    prg.done()

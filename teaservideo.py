#!/usr/bin/python

import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import pylab as plt

import scipy.io.wavfile as wavfile

import skimage.data as skidat
import skimage.io, skimage
import skimage.color

import Queue, threading

import time, sys

import progress


fps = 30.
sample_length = 0.2
imgsize = 800

gamma = 0.5
light = 0.3


def alphablend(src, destination, x,y):
    h,w, chans = src.shape
    dst = destination[y:y+h, x:x+w, :]
    if chans == 3:
        src = np.append(src,np.ones((h,w,1)), axis=2)
    result = destination
    result[y:y+h, x:x+w, 3] = src[:,:,3] + dst[:,:,3] * (1.-src[:,:,3])
    result[y:y+h, x:x+w,:3] = src[:,:,:3]*src[:,:,3][..., None] \
                                          + dst[:,:,:3]*dst[:,:,3][..., None] \
                                          * ((1.-src[:,:,3])/result[y:y+h, x:x+w, 3])[...,None]

    return result


class CalculateThread(threading.Thread):
    def __init__(self,sampledata, prms, rng, q):
        threading.Thread.__init__(self)
        self.data = sampledata
        self.prms = prms
        self.rng = rng
        self.q = q

    def run(self):
        startX, startY, width, height, framesamples, spectrum_part, bins = self.prms
        self.lower, self.upper = self.rng
        last_spectrumL = None
        last_spectrumR = None
        self.barvalsL = np.zeros((self.upper-self.lower, bins))
        self.barvalsR = np.zeros((self.upper-self.lower, bins))

        for i in range(self.lower,self.upper):
            maxX = framesamples/spectrum_part

            samplesL = self.data[i*step:i*step+framesamples,0]
            spectrumL = np.abs(scipy.fft(samplesL)[range(maxX)]*np.abs(np.mean(samplesL)))

            samplesR = self.data[i*step:i*step+framesamples,1]
            spectrumR = np.abs(scipy.fft(samplesR)[range(maxX)]*np.abs(np.mean(samplesR)))

            if last_spectrumL is not None:
                spectrumL = (spectrumL+last_spectrumL)/2.
            if last_spectrumR is not None:
                spectrumR = (spectrumR+last_spectrumR)/2.

            binwidth = maxX/bins

            for j in range(bins):
                self.barvalsL[i-self.lower,j] = np.log(1.+np.mean(spectrumL[j*binwidth:(j+1)*binwidth]))
                self.barvalsR[i-self.lower,j] = np.log(1.+np.mean(spectrumR[j*binwidth:(j+1)*binwidth]))

            last_spectrumL = np.copy(spectrumL)
            last_spectrumR = np.copy(spectrumR)

            queueLock.acquire()
            self.q.put(1)
            queueLock.release()


class RenderThread (threading.Thread):
    def __init__(self, bardata, prms, rng, q):
        threading.Thread.__init__(self)
        self.bardata = bardata
        self.prms = prms
        self.rng = rng
        self.q = q

    def run(self):
        startX, startY, height, bins = self.prms
        barvalsL, barvalsR = self.bardata
        lower, upper = self.rng
        self.frames = 0
        for i in range(upper-lower):
            frame = np.copy(background)[:,:,:]
            for j in range(bins):
                X = startX+j*width
                vL = barvalsL[i,j]
                hL = int(np.round(vL*height))
                vR = barvalsR[i,j]
                hR = int(np.round(vR*height))

                frame[startY:startY+hR,X:X+width-2,:3] = episode_scaled[startY:startY+hR,X:X+width-2,:]

                startY2 = 1080-startY
                frame[startY2-hL:startY2,X:X+width-2,:3] = episode_scaled[startY2-hL:startY2,X:X+width-2,:]
            self.frames+=1
            skimage.io.imsave("frame%04d.png" % (i+lower), frame)
            queueLock.acquire()
            self.q.put(1)
            queueLock.release()


class CounterThread (threading.Thread):
    def __init__(self, framenum, q, infostring, threads):
        threading.Thread.__init__(self)
        self.frames_done = 0
        self.framenum = framenum
        self.q = q
        self.infostring = infostring
        self.threads = threads

    def run(self):
        global exitFlag
        prg = progress.Progress(infostring=self.infostring)
        while not exitFlag:
            f = 0
            queueLock.acquire()
            while not self.q.empty():
                f += self.q.get()
            queueLock.release()
            if f > 0:
                self.frames_done += f
                prg.progress(float(self.frames_done)/self.framenum)
            time.sleep(.1)
        f = 0
        queueLock.acquire()
        self.q.queue.clear()
        queueLock.release()
        prg.done()


edge = (1080-imgsize)/2
logo_width = 1900-3*edge-imgsize


logo_img_orig = skimage.img_as_float(skidat.imread("logo.png"))
lH, lW, foo = logo_img_orig.shape
logo_height = lH*logo_width/lW
logo_img = scipy.misc.imresize(logo_img_orig, (logo_height, logo_width))/255.
episode_img_orig = skimage.img_as_float(skidat.imread("KP078_Google IO.jpg"))

episode_scaled = scipy.misc.imresize(episode_img_orig, (1900,1900))[410:1490:,:]/255.
#episode_scaled_grey = 0.5+skimage.color.rgb2grey(episode_scaled)/2.
episode_scaled_grey = np.power(skimage.color.rgb2grey(episode_scaled),gamma)
episode_scaled_grey = light+episode_scaled_grey*(1.-light)
episode_scaled_grey = scipy.ndimage.gaussian_filter(episode_scaled_grey, sigma=10)

episode_img = scipy.misc.imresize(episode_img_orig, (imgsize,imgsize))/255.

img_hsv = skimage.color.rgb2hsv(episode_img)
maxS = np.where(img_hsv[:,:,1].flatten() == np.amax((img_hsv[:,:,1].flatten())))
maxS_colors = img_hsv[:,:,0].flatten()[maxS]
colorcounts = np.bincount((maxS_colors*255).astype(int))
color1H = maxS_colors[np.amax(colorcounts)]
color2H = color1H + 0.3333333
if color2H > 1.0:
    color2H -= 1.0
color3H = color2H + 0.3333333
if color3H > 1.0:
    color3H -= 1.0

color1 = np.append(skimage.color.hsv2rgb([[[color1H,1,1]]]), [[1.]])
color2 = np.append(skimage.color.hsv2rgb([[[color2H,1,1]]]), [[1.]])
color3 = np.append(skimage.color.hsv2rgb([[[color3H,1,1]]]), [[1.]])

audioclip_filename = 'kp078-google-io.wav'

sample_rate,sample_data = wavfile.read(audioclip_filename)

samplenum = sample_data.shape[0]

tottime = float(samplenum)/sample_rate
framenum = int(np.round(tottime*fps))

framenum = 150

framesamples = int(np.round(float(sample_rate)*sample_length))
step = (samplenum-framesamples)/framenum

background = np.ones((1080,1900,4))
for i in range(3):
    background[:,:,i] = episode_scaled_grey

x, y = edge, 540-logo_img.shape[0]/2
background = alphablend(logo_img, background, x,y )

h, w, foo = episode_img.shape
x, y = 1900-imgsize-edge, 540-imgsize/2

background = alphablend(episode_img, background, x,y)



spectrum_part = 10
bins = 24
startX, startY = edge, edge
width, height = logo_width/bins, 540-logo_height/2-edge-20

queueLock = threading.Lock()
workQueue = Queue.Queue(100)

def start_threads(ThreadClass, data, prms, infostring):
    global exitFlag, threadnum, framenum
    exitFlag = False
    threads = []
    thread_frames = framenum/threadnum
    for i in range(threadnum):
        lower = i*thread_frames
        upper = (i+1)*thread_frames
        if (i+1 == threadnum):
            upper += framenum % threadnum

        thr = ThreadClass(data, prms, (lower,upper), workQueue)
        thr.start()
        threads.append(thr)

    counter = CounterThread(framenum, workQueue, infostring, threads)
    counter.start()

    for t in threads:
        t.join()

    exitFlag = True
    counter.join()

    return threads

threadnum = 4

prms = startX, startY, width, height, framesamples, spectrum_part, bins


barvalsL = np.zeros((framenum,bins))
barvalsR = np.zeros((framenum,bins))

exitFlag = False

calcthreads = start_threads(CalculateThread, sample_data, prms, "Calculating spectra")

for t in calcthreads:
    barvalsL[t.lower:t.upper,:] = t.barvalsL
    barvalsR[t.lower:t.upper,:] = t.barvalsR
    if t.lower > 0:
        barvalsL[t.lower,:] = (barvalsL[t.lower-1] + barvalsL[t.lower-1]) / 2.
        barvalsR[t.lower,:] = (barvalsR[t.lower-1] + barvalsR[t.lower-1]) / 2.

barvalsL -= np.min(barvalsL)
barvalsL /= np.max(barvalsL)
barvalsR -= np.min(barvalsR)
barvalsR /= np.max(barvalsR)

prms = startX, startY, height, bins

exitFlag = False

renderthreads = start_threads(RenderThread, (barvalsL, barvalsR), prms, "Rendering frames")

print "All done"

sys.exit()

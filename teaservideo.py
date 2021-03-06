#!/usr/bin/python

import numpy as np
import scipy
import scipy.misc
import scipy.ndimage

import scipy.io.wavfile as wavfile

import skimage.data as skidat
import skimage.io, skimage
import skimage.color

import Queue, threading

import time, sys

import progress

logo_fn, clip_fn, image_fn = sys.argv[1:]

fps = 30.
sample_length = 0.2
imgsize = 800

gamma = 0.5
light = 0.3

edge = (1080-imgsize)/2
logo_width = 1900-3*edge-imgsize

logo_img_orig = skimage.img_as_float(skidat.imread(logo_fn))

lH, lW, foo = logo_img_orig.shape
logo_height = lH*logo_width/lW
logo_img = scipy.misc.imresize(logo_img_orig, (logo_height, logo_width))/255.
episode_img_orig = skimage.img_as_float(skidat.imread(image_fn))

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

sample_rate,data = wavfile.read(clip_fn)

samplenum = data.shape[0]

tottime = float(samplenum)/sample_rate
framenum = int(np.round(tottime*fps))

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



part = 10
bins = 24
startX, startY = edge, edge
width, height = logo_width/bins, 540-logo_height/2-edge-20
last_spectrumL = None
last_spectrumR = None

barvalsL = np.zeros((framenum,bins))
barvalsR = np.zeros((framenum,bins))

for i in range(framenum):
#    print "calculating %03d/%3d" % (i,framenum)
    maxX = framesamples/part

    samplesL = data[i*step:i*step+framesamples,0]
    spectrumL = np.abs(scipy.fft(samplesL)[range(maxX)]*np.abs(np.mean(samplesL)))

    samplesR = data[i*step:i*step+framesamples,1]
    spectrumR = np.abs(scipy.fft(samplesR)[range(maxX)]*np.abs(np.mean(samplesR)))

    if last_spectrumL is not None:
        spectrumL = (spectrumL+last_spectrumL)/2.
    if last_spectrumR is not None:
        spectrumR = (spectrumR+last_spectrumR)/2.

    binwidth = maxX/bins

    for j in range(bins):
        barvalsL[i,j] = np.log(1.+np.mean(spectrumL[j*binwidth:(j+1)*binwidth]))
        barvalsR[i,j] = np.log(1.+np.mean(spectrumR[j*binwidth:(j+1)*binwidth]))

    last_spectrumL = np.copy(spectrumL)
    last_spectrumR = np.copy(spectrumR)


barvalsL -= np.min(barvalsL)
barvalsL /= np.max(barvalsL)
barvalsR -= np.min(barvalsR)
barvalsR /= np.max(barvalsR)


last_time = time.clock()

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

            skimage.io.imsave("frame%04d.png" % (i+lower), frame)
            queueLock.acquire()
            self.q.put(1)
            queueLock.release()


class CounterThread (threading.Thread):
    def __init__(self, framenum, q, infostring):
        threading.Thread.__init__(self)
        self.frames_done = 0
        self.framenum = framenum
        self.q = q
        self.infostring = infostring

    def run(self):
        prg = progress.Progress(infostring=self.infostring)
        while not exitFlag:
            queueLock.acquire()
            f = 0
            while not self.q.empty():
                f += self.q.get()
            queueLock.release()
            if f > 0:
                self.frames_done += f
                prg.progress(float(self.frames_done)/self.framenum)
                time.sleep(.1)
        prg.done()


queueLock = threading.Lock()
workQueue = Queue.Queue(10)

threadnum = 4

thread_frames = framenum / threadnum
prms = startX, startY, height, bins

threads = []

exitFlag = False

for i in range(threadnum):
    l = i*thread_frames
    u = (i+1)*thread_frames
    if (i+1==threadnum):
        u+=framenum % threadnum
    rt = RenderThread((barvalsL[l:u,:],barvalsR[l:u,:]), prms, (l,u), workQueue)
    rt.start()
    threads.append(rt)

counter = CounterThread(framenum, workQueue, "Rendering frames")
counter.start()

for t in threads:
    t.join()

exitFlag = True

counter.join()

print "All done"

sys.exit()

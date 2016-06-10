
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import pylab as plt

import scipy.io.wavfile as wavfile

import skimage.data as skidat
import skimage.io, skimage

import time, sys

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


fps = 30.
sample_length = 0.2

logo_img = skimage.img_as_float(skidat.imread("logo.png"))
episode_img = skimage.img_as_float(skidat.imread("KP077_rifugxintoj.jpg"))

episode_scaled = scipy.misc.imresize(episode_img, (1080,1900))

blurred = np.zeros((1080,1900,3))
for i in range(3):
    blurred[:,:,i] = scipy.ndimage.gaussian_filter(episode_scaled[:,:,i], sigma=30)/255.

print np.max(blurred), np.min(blurred)
print np.max(episode_img), np.min(episode_scaled)

skimage.io.imsave("test.jpg", blurred)



audioclip_filename = 'kp077-rifugxintoj.wav'

sample_rate,data = wavfile.read(audioclip_filename)
print data.shape, sample_rate

samplenum = data.shape[0]

tottime = float(samplenum)/sample_rate
framenum = int(np.round(tottime*fps))

framesamples = int(np.round(float(sample_rate)*sample_length))
step = (samplenum-framesamples)/framenum

print framesamples

background = np.ones((1080,1900,4))
background[:,:,:3] = blurred

x, y = 90, 540-logo_img.shape[0]/2
background = alphablend(logo_img, background, x,y )

h, w, foo = episode_img.shape
x, y = 910, 540-episode_img.shape[0]/2

background = alphablend(episode_img, background, x,y)



part = 10
bins = 24
totwidth = 730
startX, startY = 90, 90
width, height = totwidth/bins, 300
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

for i in range(14,15):
    print "rendering %03d/%3d" % (i,framenum),
    frame = np.copy(background)[:,:,0:3]
    for j in range(bins):
        X = startX+j*width
        vL = barvalsL[i,j]
        hL = int(np.round(vL*height))
        vR = barvalsR[i,j]
        hR = int(np.round(vR*height))

        rectU = np.ones((hR,width-2,3))
        rectU[:,:,:] = [0.,1.,0.]

        rectL = np.ones((hL,width-2,3))
        rectL[:,:,:] = [0.,1.,0.]

        l = int(np.round(0.5*height))
        if vL > 0.5:
            rectL[:-l,:,:] = [1.,1.,0]
        if vR > 0.5:
            rectU[l:,:,:] = [1.,1.,0]

        l = int(np.round(0.8*height))
        if vL > 0.8:
            rectL[:-l,:,:] = [1.,0.,0]
        if vR > 0.7:
            rectU[l:,:,:] = [1.,0.,0]

        frame[startY:startY+hR,X:X+width-2,:] = rectU

        startY2 = 1080-startY
        frame[startY2-hL:startY2,X:X+width-2,:] = rectL

    skimage.io.imsave("frame%03d.png" % i, frame)


    this_time = time.clock()
    print this_time - last_time
    sys.stdout.flush()
    last_time = this_time

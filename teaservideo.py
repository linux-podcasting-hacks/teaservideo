
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import pylab as plt

import scipy.io.wavfile as wavfile

import skimage.data as skidat
import skimage.io, skimage
import skimage.color

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
episode_img = skimage.img_as_float(skidat.imread("KP078_Google IO.jpg"))

episode_scaled = scipy.misc.imresize(episode_img, (1080,1900))

blurred = np.zeros((1080,1900,3))
for i in range(3):
    blurred[:,:,i] = scipy.ndimage.gaussian_filter(episode_scaled[:,:,i], sigma=30)/255.

print np.max(blurred), np.min(blurred)
print np.max(episode_img), np.min(episode_scaled)

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

print np.array([[[color1H,1,1]]]).shape

color1 = np.append(skimage.color.hsv2rgb([[[color1H,1,1]]]), [[1.]])
color2 = np.append(skimage.color.hsv2rgb([[[color2H,1,1]]]), [[1.]])
color3 = np.append(skimage.color.hsv2rgb([[[color3H,1,1]]]), [[1.]])

print color1

audioclip_filename = 'kp078-google-io.wav'

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

for i in range(framenum):
    print "rendering %03d/%3d" % (i,framenum),
    frame = np.copy(background)[:,:,:]
    for j in range(bins):
        X = startX+j*width
        vL = barvalsL[i,j]
        hL = int(np.round(vL*height))
        vR = barvalsR[i,j]
        hR = int(np.round(vR*height))

        alpha = 0.6

        rectU = np.ones((hR,width-2,4))
        rectU[:,:,:] = color1

        rectL = np.ones((hL,width-2,4))
        rectL[:,:,:] = color1

        colorsteps = [ (0.33, color2),
                       (0.67, color3)]

        for v,c in colorsteps:
            l = int(np.round(v*height))
            if vL > v:
                rectL[:-l,:,:] = c
            if vR > v:
                rectU[l:,:,:] = c

        frame = alphablend(rectU, frame, X, startY)
#        frame[startY:startY+hR,X:X+width-2,:] = rectU

        startY2 = 1080-startY
        frame = alphablend(rectL, frame, X, startY2-hL )
#        frame[startY2-hL:startY2,X:X+width-2,:] = rectL

    skimage.io.imsave("frame%03d.png" % i, frame)


    this_time = time.clock()
    print this_time - last_time
    sys.stdout.flush()
    last_time = this_time

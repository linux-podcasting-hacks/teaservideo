# Teaservideo
Ugly hack to render a teaser video for a podcast episode

## Disclaimer

This is not a tool, but a hack. So there is no settings menu or something like
that. Everything can be changed in the python code. Furthermore there's no
error handling whatsoever (see also "Restrictions").

## Usecase

I am doing podcasts and I wanted to have short teaser videos for my episode to
post them to Facebook, Twitter, Youtube, etc. A bit like Clammr but
different. So I hacked a python script that takes

* a logo image
* a sound clip
* an episode image

and renders out of them an VU-animated video clip.

Examples here: https://www.youtube.com/channel/UCVKivSw1Kr7rhOtPcciEBWg

## Usage

Put your logo image `logo.png`, the audio clip `audioclip.wav` and the episode image
`episodeimage.jpg` into the folder of the script and invoke

`$ ./doit.sh logo.png audioclip.wav episodeimage.jpg videoclip.mp4`

This should result in a video clip file `videoclip.mp4`


## Prequisites

You need to have installed

* python 2.7
  * scipy
  * skimage
* ffmpeg


## Restrictions

This is a hack, not a released tool. Some restrictions that I know
spontaneously.

* The video resolution is 1900x1080
* The audio clip format must be `wav`
* It's tested with episode images of 900x900, don't know what happens if yours
  are different.
* There must be no file matching the pattern `frame*.png` in the working
  directory. They will be erased.
* ... probably a lot more


## Support

Good luck!


./teaservideo.py $1 $2 $3

ffmpeg -r 30 -f image2  -i frame%04d.png -i $2 -ar 48000 -c:a aac -strict -2 \
       -shortest -deinterlace -c:v libx264 -flags cgop $4

rm frame*.png

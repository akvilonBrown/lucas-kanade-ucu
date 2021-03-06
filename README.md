# lucas-kanade-ucu
Computer Vision Home Assignment Module #3  UCU

Folder **matching** contains notebook with template matching examples *SAD, SSD, NCC*

File tracker.py contains the first version of *Lukas-Kanade* tracker implementation and it can be launched from the console with a command like:
```bash
python tracker.py "./Toy/img/" 152 102 192 169 -debug 20
```
where **152 102 192 169** are roi coordinates *x1 y1 x2 y2*, **-debug** is for *IPython* mode so the frames can be plotted and **20** is the number of frames.
The rest of the scripts are made in *IPython* version only.
The script can draw frames in IPython and writes a video file.

Tracker performance is not good.
For example, in *Toy* dataset the **roi** degenerates when non-affine transformations occur

Partially it can be explained by cv.warpAffine restrictions - this operation has a reference point at the top left corner.
So when the roi is rotated, the algorithm can't follow due to the long lever (it can't rotate around the center of **roi** - these dependencies are not reflected in the derivative Jacobi matrix).

I tried to solve it by clipping the image to make this reference point closer to the **roi** and restricting scaling position in the Jacobi matrix but it didn't help much.

In the second version I implemented custom affine transformation with a reference point at the center of the **roi**.
But it appears slow and has some bugs (like  scaling overshooting, so I had to correct by restricting scaling in warp matrix)
Although it performs slightly better with **roi** rotations but eventually, the degeneration of **roi** is inevitable as well.

In *Car* sequence the **roi** degenerates due to extra information on background so it expands to track the moving background.

Pyramidal extention performs worse - I can explain by inacuracies of **roi** during the transition between pyramidal levels.

The results are in the folder **outputs**: video and html files.

So at this point, I have to give up, because the task already took quite a long time to tune the algorithm.

Folder **drafts** contains draft notebook from *Colab*

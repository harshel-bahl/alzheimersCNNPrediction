import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
    
def augmentation_pipeline(images: tf.Tensor, 
                          randomRotation: bool = True,
                          randomFlip: bool = True,
                          randomZoom: bool = False,
                          filtering: bool = False) -> tf.Tensor:

    if randomRotation:
        images = tf.keras.layers.RandomRotation(factor=0.05)(images)

    if randomFlip:
        images = tf.keras.layers.RandomFlip(mode="horizontal")(images)

    if randomZoom:
        images = tf.keras.layers.RandomZoom(height_factor=0.5, width_factor=0.5)(images)

    if filtering:
        for channel in range(images.shape[-1]):
            images[:, :, :, channel] = _anisotropicFiltering2D(images[:, :, :, channel])
            
    return images

def _anisotropicFiltering2D(img,
                            niter=1,
                            kappa=50,
                            gamma=0.1,
                            step=(1.,1.),
                            option=1):
    '''
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Alistair Muldal
    Department of Pharmacology
    University of Oxford
    '''

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout

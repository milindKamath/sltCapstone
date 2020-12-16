import cv2
import os
import numpy as np
import pdb


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32
    return flow

############################################################################
# extract optical flow features
def extract_optical_flow( params, pair ):
    vid_name, optical_flow_out = params
    prev, curr = pair
    name, _ = os.path.splitext( curr.split('/')[-1] )
    out_path = os.path.join( optical_flow_out, f'{name}.jpg' )

    if not os.path.exists( out_path ):
        # read prev, curr image
        prev_img = cv2.imread( prev )
        # masking
        mask = np.zeros_like( prev_img )
        # Sets image saturation to maximum
        mask[..., 1] = 255
        prev_img = cv2.cvtColor( prev_img, cv2.COLOR_BGR2GRAY )
        curr_img = cv2.imread( curr )
        curr_img = cv2.cvtColor( curr_img, cv2.COLOR_BGR2GRAY )
        
        # optical flow
        tmp_flow = compute_TVL1( prev_img, curr_img )
        magnitude, angle = cv2.cartToPolar(tmp_flow[..., 0], tmp_flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize( magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        rgb = cv2.cvtColor( mask, cv2.COLOR_HSV2BGR )

        # save optical flow image
        cv2.imwrite( out_path, rgb )
    return out_path

############################################################################
# get pairs
def get_pairs( paths ):
    res = [ ( paths[0], paths[0] ) ]
    i = 0
    while i < len(paths) - 1:
        prev = paths[i]
        curr = paths[i+1]
        i+= 1
        if prev == curr:
            continue
        else:
            res.append( (prev, curr) )
    return res

############################################################################
# you also get duplicates
def add_duplicates( paths, results ):
    
    # check duplicates
    duplicates = {}
    prev = paths[0]
    for i in range( 1, len(paths) ):
        curr = paths[i]
        if prev == curr:
            name, _ = os.path.splitext( curr.split('/')[-1] )
            duplicates[name] = duplicates[name]+1 if name in duplicates.keys() else 1
        prev = curr
    
    if len(duplicates) == 0:
        return results
    
    ############################################################################
    # duplicates present
    res = []
    for path in results:
        res.append( path )
        name, _ = os.path.splitext( path.split('/')[-1] )
        if name in duplicates.keys():
            while duplicates[name] > 0:
                res.append( path )
                duplicates[name] -= 1
    return res





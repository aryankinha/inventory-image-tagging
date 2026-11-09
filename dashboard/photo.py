"""Classical foreground preparation for one clothing item, separate from the benchmark."""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def isolate_item(image, crop=None, rotation=0, shadow_background=False):
    """Estimate a border background, keep one connected foreground and centre it.

    This uses no labels, models or predicted scores. It needs a plain background
    distinct from the item. Complex scenes and multiple objects require a crop.
    """
    image=image.copy()
    if crop is not None:
        if len(crop)!=4 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError('Choose a crop with positive width and height inside the picture.')
        box=(round(crop[0]*image.width),round(crop[1]*image.height),round(crop[2]*image.width),round(crop[3]*image.height))
        if box[2]-box[0]<2 or box[3]-box[1]<2:
            raise ValueError('The selected crop is too small.')
        image=image.crop(box)
    if rotation not in (0,90,180,270):
        raise ValueError('Choose a rotation of 0, 90, 180 or 270 degrees.')
    if rotation:
        image=image.rotate(rotation,expand=True)
    image.thumbnail((384,384),Image.Resampling.LANCZOS)
    if min(image.size)<8:
        raise ValueError('Use a wider crop with the whole item visible.')
    rgb=np.asarray(image.convert('RGB'))
    lab=rgb2lab(rgb)
    edge=max(1,round(min(image.size)*.03))
    border=np.zeros(rgb.shape[:2],bool)
    border[:edge]=True; border[-edge:]=True; border[:,:edge]=True; border[:,-edge:]=True
    if shadow_background:
        # Several border colours describe a wall containing light and shadow.
        edge_colours=lab[border][::max(1,int(border.sum()/1500))]
        groups=min(3,len(np.unique(np.round(edge_colours),axis=0)))
        with threadpool_limits(limits=2):
            background=KMeans(n_clusters=groups,random_state=42,n_init=10).fit(edge_colours).cluster_centers_
        distance=np.linalg.norm(lab[:,:,None,:]-background,axis=3).min(axis=2)
    else:
        background=np.median(lab[border],axis=0)
        distance=np.linalg.norm(lab-background,axis=2)
    distance=ndi.gaussian_filter(distance,1)
    if np.ptp(distance)<1:
        raise ValueError('The item does not stand out from its background. Use a plain contrasting background.')
    threshold=max(2.,float(threshold_otsu(distance)))
    mask=distance>threshold
    mask=ndi.binary_closing(mask,iterations=2)
    mask=ndi.binary_fill_holes(mask)
    components,count=ndi.label(mask)
    areas=np.bincount(components.ravel());areas[0]=0
    if not count or areas.max()<.015*mask.size:
        raise ValueError('Could not find a clear clothing item. Crop around one item on a plain background.')
    order=np.argsort(areas)[::-1]
    if len(order)>1 and areas[order[1]]>max(.02*mask.size,.20*areas[order[0]]):
        raise ValueError('More than one foreground region was found. Crop around a single item first.')
    mask=components==order[0]
    if mask.mean()>.85 or mask[border].mean()>.25:
        raise ValueError('The background is unclear. Leave space around one item on a plain background.')
    ys,xs=np.nonzero(mask)
    fill=mask.sum()/((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1))
    aspect=max((ys.max()-ys.min()+1)/(xs.max()-xs.min()+1),(xs.max()-xs.min()+1)/(ys.max()-ys.min()+1))
    if fill<.25 or aspect>5:
        raise ValueError('Only fragments of the item were found. Use a contrasting plain background or a tighter crop.')
    gray=np.asarray(ImageOps.grayscale(image),dtype=np.float32)
    background_gray=float(np.median(gray[border]))
    # Both dark and light garments become light foreground on the benchmark's black background.
    foreground=gray if float(np.median(gray[mask]))>=background_gray else 255-gray
    low,high=np.percentile(foreground[mask],[2,98])
    if high-low<1:
        scaled=np.full(gray.shape,200,dtype=np.float32)
    else:
        scaled=50+190*np.clip((foreground-low)/(high-low),0,1)
    isolated=Image.fromarray(np.where(mask,scaled,0).astype(np.uint8))
    ys,xs=np.nonzero(mask)
    isolated=isolated.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
    scale=min(26/isolated.width,26/isolated.height)
    size=tuple(max(1,min(26,round(side*scale))) for side in isolated.size)
    isolated=isolated.resize(size,Image.Resampling.LANCZOS)
    prepared=Image.new('L',(28,28),0)
    prepared.paste(isolated,((28-isolated.width)//2,(28-isolated.height)//2))
    return prepared,[('Selected picture',image),('Foreground mask',Image.fromarray(mask.astype(np.uint8)*255)),('Prepared input',prepared)]


def photo_controls(prefix):
    """Small shared controls for the Phase 1 and Phase 2 photo interfaces."""
    import streamlit as st
    isolate=st.checkbox('Remove a plain background',value=True,key=prefix+'_isolate',
                        help='For one item on a background that looks different from the item. Turn off for an already prepared image.')
    crop=None
    rotation=0
    shadows=False
    if isolate:
        shadows=st.checkbox('Background has strong shadows',key=prefix+'_shadows')
        if st.checkbox('Crop around one item',key=prefix+'_crop_enabled'):
            horizontal=st.slider('Keep left to right (%)',0,100,(0,100),key=prefix+'_crop_x')
            vertical=st.slider('Keep top to bottom (%)',0,100,(0,100),key=prefix+'_crop_y')
            crop=(horizontal[0]/100,vertical[0]/100,horizontal[1]/100,vertical[1]/100)
        rotation=st.selectbox('Turn the picture', [0,90,180,270],key=prefix+'_rotation',
                              format_func=lambda d: 'Keep upright' if d==0 else f'{d}° anticlockwise')
        st.caption('Use one complete item. Crop out people, other clothes and extra shoes. Footwear should face sideways.')
    return {'isolate':isolate,'crop':crop,'rotation':rotation,'shadow_background':shadows}

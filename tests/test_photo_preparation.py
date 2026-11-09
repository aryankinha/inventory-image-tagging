"""Meaningful foreground and geometry checks independent of internet images or labels."""
from io import BytesIO
import numpy as np
from PIL import Image,ImageDraw
import pytest
from dashboard.services import prepare_image


def content(image):
    out=BytesIO();image.save(out,format='PNG');return out.getvalue()


@pytest.mark.parametrize('background,foreground',[('white','black'),('black','white'),('#dddddd','#f8f8f8')])
def test_single_object_is_isolated_with_black_padding(background,foreground):
    image=Image.new('RGB',(100,140),background)
    ImageDraw.Draw(image).rectangle((30,25,70,115),fill=foreground)
    pixels,steps=prepare_image(content(image),experimental=True,isolate=True)
    assert pixels.shape==(28,28) and pixels.dtype==np.uint8
    assert pixels[0].max()==pixels[-1].max()==0
    assert pixels[14,14]>0
    assert any('mask' in name for name,_ in steps)
    assert np.array_equal(pixels,prepare_image(content(image),experimental=True,isolate=True)[0])


def test_two_objects_require_a_crop_and_crop_selects_one():
    image=Image.new('RGB',(200,140),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,75,120),fill='black')
    draw.rectangle((125,20,180,120),fill='black')
    with pytest.raises(ValueError,match='More than one'):
        prepare_image(content(image),experimental=True,isolate=True)
    selected,_=prepare_image(content(image),experimental=True,isolate=True,crop=(0,0,.49,1))
    assert selected[14,14]>0 and selected[0].max()==0


@pytest.mark.parametrize('crop',[(.5,0,.4,1),(-.1,0,1,1),(0,0,0,1)])
def test_bad_crops_are_rejected(crop):
    with pytest.raises(ValueError,match='crop'):
        prepare_image(content(Image.new('RGB',(100,100),'white')),experimental=True,isolate=True,crop=crop)


def test_exact_benchmark_bypasses_all_photo_operations():
    values=np.random.default_rng(42).integers(0,256,(28,28),dtype=np.uint8)
    actual,_=prepare_image(content(Image.fromarray(values)),isolate=True,crop=(.2,.2,.8,.8),rotation=90)
    assert np.array_equal(actual,values)


def test_rotation_changes_geometry_without_stretching():
    image=Image.new('RGB',(160,100),'white')
    ImageDraw.Draw(image).rectangle((30,35,130,65),fill='black')
    a,_=prepare_image(content(image),experimental=True,isolate=True)
    b,_=prepare_image(content(image),experimental=True,isolate=True,rotation=90)
    ya,xa=np.nonzero(a);yb,xb=np.nonzero(b)
    assert np.ptp(xa)>np.ptp(ya) and np.ptp(yb)>np.ptp(xb)


def test_shadow_background_uses_border_colours_without_labels():
    image=Image.new('RGB',(160,160),'#eeeeee')
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,65,159),fill='#777777')
    draw.rectangle((45,35,115,125),fill='black')
    pixels,_=prepare_image(content(image),experimental=True,isolate=True,shadow_background=True)
    assert pixels[14,14]>0 and pixels[0].max()==0


def test_thin_fragment_is_not_forced_into_a_clothing_class():
    image=Image.new('RGB',(180,100),'white')
    ImageDraw.Draw(image).rectangle((25,48,155,52),fill='black')
    with pytest.raises(ValueError,match='fragments'):
        prepare_image(content(image),experimental=True,isolate=True)


@pytest.mark.parametrize('foreground', ['black', 'white'])
@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_transparent_item_retains_visible_silhouette(foreground, mode):
    image=Image.new('RGBA',(100,140),(255,0,0,0))
    ImageDraw.Draw(image).rectangle((30,25,70,115),fill=foreground)
    if mode=='LA':
        image=image.convert('LA')
    elif mode=='P':
        image=image.convert('P')
    pixels,_=prepare_image(content(image),experimental=True,isolate=True)
    assert pixels[14,14]>0 and pixels[0].max()==0


def test_fully_transparent_item_is_explained():
    with pytest.raises(ValueError,match='fully transparent'):
        prepare_image(content(Image.new('RGBA',(100,100),(255,0,0,0))),experimental=True,isolate=True)


def test_small_item_is_enlarged_to_match_input_geometry():
    image=Image.new('RGB',(40,40),'white')
    ImageDraw.Draw(image).rectangle((14,10,25,29),fill='black')
    pixels,_=prepare_image(content(image),experimental=True,isolate=True)
    ys,xs=np.nonzero(pixels>40)
    assert np.ptp(ys)>=23 and np.ptp(xs)>=12


def test_hidden_transparent_colours_do_not_change_preparation():
    outputs=[]
    for hidden in [(255,0,0,0),(0,255,255,0)]:
        image=Image.new('RGBA',(100,140),hidden)
        ImageDraw.Draw(image).rectangle((30,25,70,115),fill=(20,20,20,255))
        outputs.append(prepare_image(content(image),experimental=True,isolate=True)[0])
    assert np.array_equal(*outputs)


def test_camera_orientation_matches_the_visible_picture():
    from PIL import ImageOps
    source=Image.new('RGB',(140,100),'white')
    ImageDraw.Draw(source).polygon([(20,25),(90,25),(90,45),(120,45),(120,75),(20,75)],fill='black')
    exif=Image.Exif();exif[274]=6
    encoded=BytesIO();source.save(encoded,format='JPEG',exif=exif)
    with Image.open(BytesIO(encoded.getvalue())) as opened:
        visible=ImageOps.exif_transpose(opened).copy()
    actual,_=prepare_image(encoded.getvalue(),experimental=True,isolate=True)
    expected,_=prepare_image(content(visible),experimental=True,isolate=True)
    assert np.array_equal(actual,expected)


def test_partially_transparent_dark_item_is_visible():
    image=Image.new('RGBA',(100,140),(200,200,200,0))
    ImageDraw.Draw(image).rectangle((30,25,70,115),fill=(20,20,20,60))
    pixels,_=prepare_image(content(image),experimental=True,isolate=True)
    assert pixels[14,14]>0 and pixels[0].max()==0

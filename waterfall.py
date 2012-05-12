#! /usr/bin/env python2

# RTL-SDR Waterfall
# licensed GPLv3

import sys, math, time, ctypes, numpy, pyglet

from rtlsdr import *
from itertools import *
from pyglet.gl import *
from pyglet.window import key

from radio_math import *

# todo
# interleaved scans
# multithreaded async scan
# textures instead of quadstrip could save a lot
# (3 bytes per sample instead of 14)
# resizable selection
# autocorrelation
# croccydile wants 30 minutes at 4 fps.
# middle mouse velocity drag to free pan the viewport?
# middle scroll to reach history?
# automatic offset for low bandwidth

if len(sys.argv) != 3:
    print "use: waterfall.py <lower freq> <upper freq>"
    print "    frequencies in hertz"
    print "    example: waterfall.py 929e6 930e6"
    print "    arrow keys pan and zoom (shift for bigger steps)"
    print "    brackets to adjust gain"
    print "    click and drag to select"
    print "    esc to quit"
    sys.exit(2)

class Stateful(object):
    "bucket of globals"
    def __init__(self):
        self.freq_lower = None
        self.freq_upper = None
        self.vertexes   = []   # (timestamp, vertex_list)
        self.time_start = None
        self.viewport   = None
        self.history    = 60   # seconds
        self.focus      = False
        self.hover      = 0
        self.highlight  = False
        self.hl_lo      = None
        self.hl_hi      = None
        self.hl_filter  = None
        self.hl_pixels  = None


state = Stateful()

state.freq_lower = float(sys.argv[1])
state.freq_upper = float(sys.argv[2])
state.time_start = time.time()
state.viewport = (0,0,1,1)

# Since this is dealing with a stupid amount of data in the video ram,
# the x axis is MHz and the y axis is seconds.
# Nothing is ever updated to scroll, instead panning moves the viewport
# and changes the aspect ratio.
# Good luck drawing widgets on top of that.
# (See the textbox() function for the required contortions to overlay.)

class SdrWrap(object):
    "wrap sdr and try to manage tuning"
    def __init__(self):
        self.sdr = RtlSdr()
        self.read_samples = self.sdr.read_samples
        self.prev_fc = None
        self.prev_fs = None
        self.prev_g  = 19
        self.sdr.gain = 19
    def tune(self, fc, fs, g):
        if fc == self.prev_fc and fs == self.prev_fs and g == self.prev_g:
            return
        if fc != self.prev_fc:
            self.sdr.center_freq = fc
        if fs != self.prev_fs:
            self.sdr.sample_rate = fs
        if g != self.prev_g:
            self.sdr.gain = g
        self.prev_fc = fc
        self.prev_fs = fs
        self.prev_g  = g
        time.sleep(0.04)  # wait for settle
        self.sdr.read_samples(2**11)  # clear buffer
        configure_highlight()
    def gain_change(self, x):
        # the whole 10x gain number is annoying
        real_g = int(self.prev_g * 10)
        i = self.sdr.GAIN_VALUES.index(real_g)
        i += x
        i = min(len(self.sdr.GAIN_VALUES) -1, i)
        i = max(0, i)
        new_g = self.sdr.GAIN_VALUES[i]
        self.sdr.gain = new_g / 10.0
        self.prev_g   = new_g / 10.0

sdr = SdrWrap()

try:
    config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
    window = pyglet.window.Window(config=config, resizable=True)
except pyglet.window.NoSuchConfigException:
    print 'Disabling 4xAA'
    window = pyglet.window.Window(resizable=True)
window.clear()
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_BLEND)
glEnable(GL_LINE_SMOOTH)
glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

fnt = pyglet.font.load('Monospace')
fnt.size = 48

@window.event
def on_draw():
    pass

@window.event
def on_key_press(symbol, modifiers):
    delta = state.freq_upper - state.freq_lower
    if   symbol == key.LEFT and modifiers & key.MOD_SHIFT:
        state.freq_lower -= delta
        state.freq_upper -= delta
    elif symbol == key.RIGHT and modifiers & key.MOD_SHIFT:
        state.freq_lower += delta
        state.freq_upper += delta
    elif symbol == key.UP    and modifiers & key.MOD_SHIFT:
        state.freq_lower += delta * 0.3
        state.freq_upper -= delta * 0.3
    elif symbol == key.DOWN  and modifiers & key.MOD_SHIFT:
        state.freq_lower -= delta * 0.75
        state.freq_upper += delta * 0.75
    elif symbol == key.LEFT:
        state.freq_lower -= delta * 0.1
        state.freq_upper -= delta * 0.1
    elif symbol == key.RIGHT:
        state.freq_lower += delta * 0.1
        state.freq_upper += delta * 0.1
    elif symbol == key.UP:
        state.freq_lower += delta * 0.1
        state.freq_upper -= delta * 0.1
    elif symbol == key.DOWN:
        state.freq_lower -= delta * 0.125
        state.freq_upper += delta * 0.125
    elif symbol == key.BRACKETLEFT:
        sdr.gain_change(-1)
    elif symbol == key.BRACKETRIGHT:
        sdr.gain_change(1)
    state.freq_lower = max(60e6, state.freq_lower)
    state.freq_upper = min(1700e6, state.freq_upper)

@window.event
def on_mouse_motion(x, y, dx, dy):
    state.hover = x_to_freq(x)

@window.event
def on_mouse_enter(x, y):
    state.focus = True

@window.event
def on_mouse_leave(x, y):
    state.focus = False

@window.event
def on_mouse_press(x, y, buttons, modifiers):
    state.highlight = False
    state.hl_filter = None

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    pass

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    cursor = x_to_freq(x)
    if not state.highlight:
        state.highlight = True
        state.hl_lo  = cursor
        state.hl_hi   = cursor
        return
    if cursor > state.hl_lo:
        state.hl_hi  = cursor
    else:
        state.hl_lo = cursor
    state.hover = cursor
    if state.hl_lo != state.hl_hi:
        configure_highlight()

batch  = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch()

def x_to_freq(x):
    vp = state.viewport
    delta = state.freq_upper - state.freq_lower
    return delta * x / window.width + state.freq_lower

def configure_highlight():
    if not state.highlight:
        return
    pass_fc = (state.hl_lo + state.hl_hi) / 2
    pass_bw = state.hl_hi - state.hl_lo
    if pass_bw == 0:
        return
    state.hl_filter = Bandpass(sdr.prev_fc, sdr.prev_fs, 
                               pass_fc, pass_bw)

def constellation(stream):
    if state.hl_filter is None:
        state.hl_pixels = None
        return
    vp = state.viewport
    stream2 = state.hl_filter(stream)
    bounds = max(numpy.abs(stream2))
    if bounds > 1:
        stream2 = stream2/bounds
    points = []
    x,y = vp[0]*0.10 + vp[1]*0.90, vp[2]*0.80 + vp[3]*0.20
    ratio = ((vp[3]-vp[2])/window.height) / ((vp[1]-vp[0])/window.width)
    for p in stream2:
        xp,yp= p.real, p.imag
        points.append(xp + x)
        points.append(yp*ratio + y)
    state.hl_pixels = points

def mapping(x):
    "assumes -50 to 0 range, returns color"
    r = int((x+50) * 255 // 50)
    r = max(0, r)
    r = min(255, r)
    return r,r,100

def log2(x):
    return math.log(x)/math.log(2)

def raw_image(colors):
    "convert a list of RGB into a pyglet image"
    # still have to bind it to a polygon and make it work
    ca1 = ctypes.c_uint8 * len(colors)
    ca2 = ca1(*colors)
    return pyglet.image.ImageData(len(colors)//3, 1, 'RGB', ca2)

def acquire_offset(center, bw, detail, samples=8, relay=None):
    "a better view for high zoom"
    assert bw <= 1.4e6
    if detail < 8:
        detail = 8
    sdr.tune(center-0.7e6, 2.8e6, sdr.prev_g)
    detail = 2**int(math.ceil(log2(detail)))
    scale = 2.8e6 / bw
    sample_count = samples * detail * scale
    data = sdr.read_samples(sample_count)
    # should probably cache these filters
    data = Translate(1, 4)(data)
    data = DownsampleFloat(scale)(data)
    ys,xs = psd(data, NFFT=detail, Fs=bw/1e6, Fc=center/1e6)
    ys = 10 * numpy.log10(ys)
    if relay:
        relay(data)
    return xs, ys


def acquire_sample(center, bw, detail, samples=8, relay=None):
    "collect a single frequency"
    assert bw <= 2.8e6
    if detail < 8:
        detail = 8
    sdr.tune(center, bw, sdr.prev_g)
    detail = 2**int(math.ceil(log2(detail)))
    sample_count = samples * detail
    data = sdr.read_samples(sample_count)
    ys,xs = psd(data, NFFT=detail, Fs=bw/1e6, Fc=center/1e6)
    ys = 10 * numpy.log10(ys)
    if relay:
        relay(data)
    return xs, ys

def acquire_range(lower, upper):
    "automatically juggles frequencies"
    delta  = upper - lower
    center = (upper+lower)/2
    #if delta < 1.4e6:
    #    return acquire_offset(center, delta,
    #        detail=window.width, relay=constellation)
    if delta < 2.8e6:
        # single sample
        return acquire_sample(center, 2.8e6,
            detail=window.width*2.8e6/delta,
            relay=constellation)
    xs2 = numpy.array([])
    ys2 = numpy.array([])
    detail = window.width // ((delta)/(2.8e6))
    for f in range(int(lower), int(upper), int(2.8e6)):
        xs,ys = acquire_sample(f+1.4e6, 2.8e6, detail=detail)
        xs2 = numpy.append(xs2, xs) 
        ys2 = numpy.append(ys2, ys) 
    return xs2, ys2

def render_sample(now, dt, freqs, powers):
    quads = []
    colors = []
    for i,f in enumerate(freqs):
        quads.extend([f, now, f, now-dt])
        rgb = mapping(powers[i])
        colors.extend(rgb)
        colors.extend(rgb)
    # quads/colors are slanted?
    quads = quads[:2] + quads + quads[-2:]
    colors = colors[:3] + colors + colors[-3:]
    # something here leaks memory at 50MB/minute
    vert_list = batch.add(len(quads)//2, GL_QUAD_STRIP, None,
        ('v2f/static', tuple(quads)), ('c3B/static', tuple(colors)))
    state.vertexes.append((now, vert_list))

def change_viewport(x1, x2, y1, y2):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(x1, x2, y1, y2, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    #buff = pyglet.image.BufferManager()
    #tuple(pyglet.image.BufferManager.get_viewport(buff))
    # todo - find way of reading viewport
    state.viewport = (x1, x2, y1, y2)

def textbox(lines):
    # there has to be a better way to do this
    # multiple viewports?  off screen render?
    # consider a prettier paragraph style
    s = '\n'.join('%s %s' % pair for pair in lines)
    vp = state.viewport
    x,y = vp[0]*0.98 + vp[1]*0.02, vp[2]*0.98 + vp[3]*0.02
    ratio = ((vp[3]-vp[2])/window.height) / ((vp[1]-vp[0])/window.width)
    # this is technically deprecated
    # but is the easiest way to do multiline text
    label = pyglet.font.Text(fnt, text=s, width=1000, color=(1,1,1,0.5),
        x=0, y=0, halign='left', valign='bottom')
    for i,vl in enumerate(label._layout._vertex_lists):
        verts = []
        for j,v in enumerate(vl.vertices):
            if j%2:  # y
                verts.append(v/20 + y)
            else:
                verts.append(v/ratio/20 + x)
        label._layout._vertex_lists[i].vertices = verts
    label.draw()

def highlighter():
    if not state.highlight:
        return
    # draw a single translucent quad
    vp = state.viewport
    x1,x2,y1,y2 = state.hl_lo/1e6, state.hl_hi/1e6, vp[2], vp[3]
    quad = (x1,y1, x2,y1, x2,y2, x1,y2)
    color = (255, 255, 255, 128) * 4
    pyglet.graphics.draw(4, GL_POLYGON, ('v2f', quad),
                         ('c4B', color))

def update(dt):
    now = time.time() - state.time_start
    freqs,power = acquire_range(state.freq_lower, state.freq_upper)
    render_sample(now, dt, freqs, power)
    window.clear()
    batch.draw()
    batch2.draw()
    change_viewport(state.freq_lower/1e6, state.freq_upper/1e6,
                    now - state.history, now)
    vp = state.viewport
    delta = vp[1] - vp[0]
    text = [('Lower:', '%0.3fMHz' % (state.freq_lower/1e6)),
            ('Upper:', '%0.3fMHz' % (state.freq_upper/1e6)),
            ('Gain: ', '%0.1fdB'  % sdr.sdr.gain),]
    if state.highlight:
        text.append(('Width:', '%0.3fkHz' % ((state.hl_hi-state.hl_lo)/1e3)))
    if state.highlight and state.hl_pixels:
        pyglet.graphics.draw(len(state.hl_pixels)/2, GL_POINTS, ('v2f', state.hl_pixels))
    if state.focus:
        text.append(('Mouse:', '%0.3fMHz' % (state.hover/1e6)))
    textbox(text)
    highlighter()
    while state.vertexes and state.vertexes[0][0] < (now-60):
        state.vertexes[0][1].delete()
        v = state.vertexes.pop(0)
        del(v)

def batch_swap(dt):
    "call occasionally to actually free batch's memory"
    global batch, batch2
    batch, batch2 = batch2, batch


pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.clock.schedule_interval(batch_swap, 70)
pyglet.app.run()



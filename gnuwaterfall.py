#! /usr/bin/env python2

# GNU Waterfall
# licensed GPLv3

import sys, math, time, numpy, pyglet

from rtlsdr import *
from itertools import *
from pyglet.gl import *
from pyglet.window import key

from radio_math import psd

# todo
# graph lines
# mouse control
# interleaved scans
# constellation plot


if len(sys.argv) != 3:
    print "use: gnuwaterfall.py <lower freq> <upper freq>"
    print "    frequencies in hertz"
    print "    example: gnuwaterfall.py 929e6 930e6"
    print "    arrow keys pan and zoom"
    print "    brackets to adjust gain"
    print "    esc to quit"
    sys.exit(2)

class Stateful(object):
    "bucket of globals"
    def __init__(self):
        self.freq_lower = None
        self.freq_upper = None
        self.vertexes   = {}   # timestamp : vertex_list
        self.time_start = None
        self.viewport   = None
        self.history    = 60   # seconds
        self.focus      = False
        self.hover      = 0


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
    # the whole 10x gain number is annoying
    def gain_change(self, x):
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
    if   symbol == key.LEFT:
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
    vp = state.viewport
    delta = state.freq_upper - state.freq_lower
    freq = delta * x / window.width + state.freq_lower
    state.hover = freq

@window.event
def on_mouse_enter(x, y):
    state.focus = True

@window.event
def on_mouse_leave(x, y):
    state.focus = False

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    pass

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    pass

batch = pyglet.graphics.Batch()

def mapping(x):
    "assumes -50 to 0 range, returns color"
    r = int((x+50) * 255 // 50)
    r = max(0, r)
    r = min(255, r)
    return r,r,100

def log2(x):
    return math.log(x)/math.log(2)

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
        relay(center, bw, data)
    return xs, ys

def acquire_range(lower, upper):
    "collect multiple frequencies"
    delta = upper - lower
    if delta < 2.8e6:
        # single sample
        return acquire_sample((upper+lower)/2, 2.8e6, detail=window.width*2.8e6/delta)
    xs2 = numpy.array([])
    ys2 = numpy.array([])
    detail = window.width // ((delta)/(2.8e6))
    for f in range(int(lower), int(upper), int(2.8e6)):
        xs,ys = acquire_sample(f+1.4e6, 2.8e6, detail=detail)
        xs2 = numpy.append(xs2, xs) 
        ys2 = numpy.append(ys2, ys) 
    return xs2, ys2

def render_sample(now, dt, freqs, powers):
    min_p = min(powers)
    max_p = max(powers)
    quads = []
    colors = []
    for i,f in enumerate(freqs):
        quads.extend([f, now, f, now-dt])
        rgb = mapping(powers[i])
        colors.extend(rgb)
        colors.extend(rgb)
    quads = quads[:2] + quads + quads[-2:]
    colors = colors[:3] + colors + colors[-3:]
    vert_list = batch.add(len(quads)//2, GL_QUAD_STRIP, None,
        ('v2f/static', tuple(quads)), ('c3B/static', tuple(colors)))
    state.vertexes[now] = vert_list

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
        

def update(dt):
    now = time.time() - state.time_start
    freqs,power = acquire_range(state.freq_lower, state.freq_upper)
    render_sample(now, dt, freqs, power)
    window.clear()
    batch.draw()
    change_viewport(state.freq_lower/1e6, state.freq_upper/1e6,
                    now - state.history, now)
    vp = state.viewport
    delta = vp[1] - vp[0]
    stat = [('Lower:', '%0.3fMHz' % (state.freq_lower/1e6)),
            ('Upper:', '%0.3fMHz' % (state.freq_upper/1e6)),
            ('Gain: ', '%0.1fdB'  % sdr.sdr.gain),]
    if state.focus:
        stat.append(('Mouse:', '%0.3fMHz' % (state.hover/1e6)))
    textbox(stat)
    for k in list(state.vertexes.keys()):
        if k < now-60:
            state.vertexes[k].delete()
            state.vertexes.pop(k)

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()



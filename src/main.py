import taichi as ti
import taichi.math as tm
import menger as mg
import mandelbulb as mb
import sierpinski as sp

ti.init(ti.gpu)

screen = W, H = 700,500

pos = ti.Vector.field(3, ti.f32, shape=())
camrot = ti.Vector.field(3, ti.f32, shape=())

pixels = ti.Vector.field(3, dtype=ti.f32, shape=screen)

@ti.func
def map(p: tm.vec3, iter, scale, fractal) -> ti.f32:
    res=0.
    if fractal==1:
        res = mb.mandelbulb(p, iter, scale)
    elif fractal==0:
        res = mg.menger(p,iter,scale)
    elif fractal == 2:
        res=sp.sierpinski(p,iter,scale)
    return res

@ti.kernel
def rotateCam(yaw: ti.f32, pitch: ti.f32):
    lookDir = tm.vec3(
        tm.cos(pitch)*tm.sin(yaw),
        tm.sin(pitch),
        tm.cos(pitch)*tm.cos(yaw)
    )
    camrot[None] = tm.normalize(lookDir)

@ti.kernel
def moveCam(dir:bool, reverse:bool, speed:float):
    if dir:
        if reverse:
            pos[None]-=camrot[None]*speed
        else:
            pos[None]+=camrot[None]*speed
    else:
        if reverse:
            pos[None]-=tm.normalize(tm.cross(camrot[None], tm.vec3(0,1,0)))*speed
        else:
            pos[None]+=tm.normalize(tm.cross(camrot[None], tm.vec3(0,1,0)))*speed

@ti.kernel
def camUp(speed:float):
    pos[None].y += speed


@ti.kernel
def draw(res:tm.vec2, fov:float, iter:int, scale:float, step:float,precision:float,red:float,green:float,blue:float,lumi:float,fractal:int):

    ratio = res.x/res.y

    #Camera direction vectors: forward, right, up
    f = camrot[None]
    r = tm.normalize(tm.cross(f,tm.vec3(0,-1,0)))
    u = tm.normalize(tm.cross(f, r))

    ro = pos[None]

    for x, y in pixels:
        uv = (tm.vec2(x, y)/screen)*2 -1 #Scale uv into [-1; 1]
        uv.x *= ratio                    #Fix stretching

        rd = tm.normalize(f*fov+
                          r*uv.x+
                          u*uv.y) #Ray direction

        td = 0. #total distance
        d = 0.

        col = tm.vec3(0)
        tempcol=tm.vec3(0)

        #Raymarching
        for i in range(step):
            p = ro + rd * td
            d = map(p, iter, scale,fractal)
            td += max(d, 0.00001) #fix float precision

            tempcol = i/80
            if d < .002/precision: break
            if td > 100: break

        if td > 100:
            col=tm.vec3(.7,.7,.7)
        else:
            col=(1-tempcol/lumi)*tm.vec3(red,green,blue)
        pixels[x,y] = col

w = ti.ui.Window(name="PyFractals", res=screen)

c = w.get_canvas()
gui = w.get_gui()

dspeed = .1
dspeed2=1.
sensi = .05
step = 80
fov = 3.
scale = 1.
iter = 1
precision = 1.
r=.5
g=.5
b=.5
lumi = 1.
fractal=0

yaw, pitch = 0.,0.

pos[None] = 0, 0, -6

while w.running:
    gui.text("Fractal",(1,1,1))
    iter = gui.slider_int("Iter", old_value=iter, minimum=-10, maximum=12)
    scale = gui.slider_float("Size", old_value=scale, minimum=-1, maximum=10)
    lumi=gui.slider_float("Luminosity",old_value=lumi,minimum=.01,maximum=10)
    if gui.button("Menger"): fractal = 0
    if gui.button("Mandelbulb"): fractal = 1
    if gui.button("Sierpinsky"): fractal = 2
    if gui.button("Scale to 1"):
        scale=1
    gui.text("Camera",(1,1,1))
    dspeed = gui.slider_float("Speed", old_value=dspeed, minimum=.001, maximum=.1)
    dspeed2 = gui.slider_float("Speed 2",old_value=dspeed2,minimum=.001,maximum=1.)
    sensi=gui.slider_float("Sensibility",old_value=sensi,minimum=.001,maximum=.5)
    fov = gui.slider_float("FOV", old_value=fov, minimum=-1, maximum=10)
    precision = gui.slider_float("Precision",old_value=precision,minimum=1,maximum=1000)
    gui.text("Color",(r,g,b))
    r=gui.slider_float("Red",old_value=r,minimum=0,maximum=1)
    g=gui.slider_float("Green",old_value=g,minimum=0,maximum=1)
    b=gui.slider_float("Blue",old_value=b,minimum=0,maximum=1)

    speed = dspeed*dspeed2
    if w.is_pressed(ti.ui.SHIFT): speed *= 5

    if w.is_pressed('w'): moveCam(True, False, speed)
    if w.is_pressed('a'): moveCam(False, False, speed)
    if w.is_pressed('s'): moveCam(True, True, speed)
    if w.is_pressed('d'): moveCam(False, True, speed)
    if w.is_pressed(ti.ui.SPACE): camUp(speed)
    if w.is_pressed(ti.ui.CTRL):  camUp(-speed)
    
    if w.is_pressed(ti.ui.UP): pitch += sensi
    if w.is_pressed(ti.ui.LEFT): yaw -= sensi
    if w.is_pressed(ti.ui.DOWN): pitch -= sensi
    if w.is_pressed(ti.ui.RIGHT): yaw += sensi
    pitch = tm.max(-1.55,ti.min(1.55,pitch))

    rotateCam(yaw,pitch)

    draw(w.get_window_shape(), fov, iter, scale, step,precision,r,g,b,lumi,fractal)
    c.set_image(pixels)
    w.show()

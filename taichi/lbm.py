#type:ignore
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import time
ti.init(arch=ti.gpu)

NY = 300
NX = 300
Q = 9
OMEGA = 1.7
RHO = 1.
U_0 = 0.1

w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
cx = [0,1,0,-1,0,1,-1,-1,1]
cy = [0,0,1,0,-1,1,1,-1,-1]
refl = [0,3,4,1,2,7,8,5,6]

# f = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
f = ti.field(ti.f32)
ti.root.dense(ti.k, Q).dense(ti.ij, (NX,NY)).place(f)

buf = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
# buf = ti.field(ti.f32)
# ti.root.dense(ti.ij, (NX,NY)).dense(ti.k, Q).place(buf)


wf = ti.field(dtype=ti.f32, shape=(9,))
c = ti.Vector.field(n=2, dtype=ti.f32, shape=(9,))
dr = ti.Vector.field(n=2, dtype=ti.int32, shape=(9,))
u_max = ti.field(dtype=ti.f64, shape=())
u_mag = ti.field(dtype=ti.f32, shape=(NX,NY))

wf.from_numpy(np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32))
dr.from_numpy(np.array(
    [[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],
    dtype=np.int32
))
c.from_numpy(np.array(
    [[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],
    dtype=np.float32
))
u_max = ti.field(dtype=ti.f32, shape=())
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(NX,NY))
u_mag = ti.field(dtype=ti.f32, shape=(NX,NY))

@ti.kernel
def init_shearwave():
    for x, y, i in f:
        u = ti.Vector([ti.static(U_0) * tm.sin((2.*tm.pi*y)/ti.cast(NY, ti.f32)), 0.])
        f[x,y,i] = f_eq(i, ti.static(RHO), u)

@ti.kernel
def init_rest():
    for x, y, i in f:
        f[x,y,i] = f_eq(i, ti.static(RHO), ti.Vector([0., 0.])) # = wf[i] * ti.static(RHO)

@ti.func
def f_eq(i:int, rho:float, u:ti.template()) -> ti.f32:
    ciui = tm.dot(u, c[i])
    return wf[i] * rho * (1 + 3*ciui + 4.5*ciui*ciui - 1.5*tm.dot(u,u))

@ti.func
def f_eq_opt(w_i:float, f_i:float, rho_i:float, u_i:ti.template(), c_i:ti.template()) -> ti.f32:
    ciui = tm.dot(u_i, c_i)
    f_eq_i = w_i * rho_i * (1 + 3*ciui + 4.5*ciui*ciui - 1.5*tm.dot(u_i,u_i))
    return f_i + ti.static(OMEGA) * (f_eq_i - f_i)

@ti.func
def rho(x:int,y:int) -> ti.f32:
    rho = 0.
    for i in ti.static(range(Q)):
        rho += f[x,y,i]
    return rho

@ti.func
def u(x:int,y:int,rho:ti.f32) -> ti.f32:
    u = ti.Vector([0.,0.])
    for i in ti.static(range(Q)):
        u += f[x,y,i] * c[i]
    return u / rho

@ti.kernel
def push_periodic_simple():
    for x, y in ti.ndrange(NX, NY):
        # compute density
        rho_i = 0.
        for i in ti.static(range(Q)):
            rho_i += f[x,y,i]

        # compute velocity
        u_i = ti.Vector([0.,0.])
        for i in ti.static(range(Q)):
            u_i += f[x,y,i] * c[i]
        u_i /= rho_i

        # for each channel:
        for i in ti.static(range(Q)):
            # collide
            f_eq_i = f[x,y,i] + ti.static(OMEGA) * (f_eq(i,rho_i,u_i) - f[x,y,i])
            # stream
            buf[(x+dr[i].x+NX)%NX, (y+dr[i].y+NY)%NY, i] = f_eq_i

@ti.kernel
def push_periodic(f:ti.template(), buf:ti.template()):
    for x, y in ti.ndrange(NX, NY):
        f_0 = f[x,y,0]
        f_1 = f[x,y,1]
        f_2 = f[x,y,2]
        f_3 = f[x,y,3]
        f_4 = f[x,y,4]
        f_5 = f[x,y,5]
        f_6 = f[x,y,6]
        f_7 = f[x,y,7]
        f_8 = f[x,y,8]
        # compute density and velocity
        rho_i = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8
        rho_inv = 1.0/rho_i
        ux = (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv
        uy = (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv
        # periodic boundaries
        xr = 0 if x+1==NX else x+1
        xl = NX-1 if x==0 else x-1
        yu = 0 if y+1==NY else y+1
        yd = NY-1 if y==0 else y-1
        # common subexpressions
        ux_2 = 4.5 * ux * ux
        uy_2 = 4.5 * uy * uy
        uymux = uy - ux
        uymux_2 = 4.5 * uymux * uymux
        uxpuy = ux + uy
        uxpuy_2 = 4.5 * uxpuy * uxpuy
        u_2_times_3_2 =1.5 * (ux * ux + uy * uy)
        w_4_9  = rho_i * 4./9.
        w_1_9  = rho_i * 1./9.
        w_1_36 = rho_i * 1./36.
        # collide and stream
        # | 6   2   5 |
        # |   \ | /   |
        # | 3 - 0 - 1 |
        # |   / | \   |
        # | 7   4   8 |
        buf[xl, yd, 7] = f_7+OMEGA*((w_1_36*(1-3*uxpuy+uxpuy_2-u_2_times_3_2))-f_7)
        buf[x , yd, 4] = f_4+OMEGA*((w_1_9*(1-3*uy+uy_2-u_2_times_3_2))-f_4)
        buf[xr, yd, 8] = f_8+OMEGA*((w_1_36*(1-3*uymux+uymux_2-u_2_times_3_2))-f_8)
        buf[xl, y , 3] = f_3+OMEGA*((w_1_9*(1-3*ux+ux_2-u_2_times_3_2))-f_3)
        buf[x , y , 0] = f_0+OMEGA*((w_4_9*(1-u_2_times_3_2))-f_0)
        buf[xr, y , 1] = f_1+OMEGA*((w_1_9*(1+3*ux+ux_2-u_2_times_3_2))-f_1)
        buf[xl, yu, 6] = f_6+OMEGA*((w_1_36*(1+3*uymux+uymux_2-u_2_times_3_2))-f_6)
        buf[x , yu, 2] = f_2+OMEGA*((w_1_9*(1+3*uy+uy_2-u_2_times_3_2))-f_2)
        buf[xr, yu, 5] = f_5+OMEGA*((w_1_36*(1+3*uxpuy+uxpuy_2-u_2_times_3_2))-f_5)

@ti.kernel
def push_lid_driven():
    for x, y in ti.ndrange(NX, NY):
        # compute density and velocity
        rho_i = rho(x,y)
        u_i = u(x,y,rho_i)
        for i in ti.static(range(Q)):
            # collide
            ciui = tm.dot(u_i, ti.Vector([ti.static(cx)[i], ti.static(cy)[i]]))
            f_eq_i = ti.static(w)[i] * rho_i * (1 + 3*ciui + 4.5*ciui*ciui - 1.5*tm.dot(u_i,u_i))
            f_eq_i = f[x,y,i] + ti.static(OMEGA) * (f_eq_i - f[x,y,i])
            # contribution from top moving wall - flip this for channel 6
            df = ti.static(RHO * U_0 / 6.) if (y==NY-1 and (i==5 or i==6)) else 0.
            df = (- df) if i==5 else df
            # keep inside domain and reflect channel if applicable
            drx = ti.static(cx)[i]
            dry = ti.static(cy)[i]
            i_refl = ti.static(refl)[i] if (
                (x==0 and drx<0) or 
                (x==NX-1 and drx>0) or 
                (y==0 and dry<0) or 
                (y==NY-1 and dry>0)
            ) else i
            drx = 0 if (x==0 and drx<0) or (x==NX-1 and drx>0) else drx
            dry = 0 if (y==0 and dry<0) or (y==NY-1 and dry>0) else dry
            
            # stream
            buf[x+drx, y+dry, i_refl] = f_eq_i + df

@ti.func 
def colour_map(col:ti.float32) -> ti.types.vector(3, ti.float32): 
    x = 1.0-col
    # https://github.com/kbinani/colormap-shaders/blob/master/shaders/glsl/IDL_CB-Spectral.frag
    r:ti.float32 = 0.0 # type: ignore
    g:ti.float32 = 0.0 # type: ignore
    b:ti.float32 = 0.0 # type: ignore
    # RED
    if (x < 0.09752005946586478):
        r = 5.63203907203907E+02 * x + 1.57952380952381E+02
    elif (x < 0.2005235116443438):
        r = 3.02650769230760E+02 * x + 1.83361538461540E+02
    elif (x < 0.2974133397506856):
        r = 9.21045429665647E+01 * x + 2.25581007115501E+02
    elif (x < 0.5003919130598823):
        r = 9.84288115246108E+00 * x + 2.50046722689075E+02
    elif (x < 0.5989021956920624):
        r = -2.48619704433547E+02 * x + 3.79379310344861E+02
    elif (x < 0.902860552072525):
        r = ((2.76764884219295E+03 * x - 6.08393126459837E+03) * x + 3.80008072407485E+03) * x - 4.57725185424742E+02
    else:
        r = 4.27603478260530E+02 * x - 3.35293188405479E+02
    # GREEN
    if (x < 0.09785836420571035):
        g = 6.23754529914529E+02 * x + 7.26495726495790E-01
    elif (x < 0.2034012006283468):
        g = 4.60453201970444E+02 * x + 1.67068965517242E+01
    elif (x < 0.302409765476316):
        g = 6.61789401709441E+02 * x - 2.42451282051364E+01
    elif (x < 0.4005965758690823):
        g = 4.82379130434784E+02 * x + 3.00102898550747E+01
    elif (x < 0.4981907026473237):
        g = 3.24710622710631E+02 * x + 9.31717541717582E+01
    elif (x < 0.6064345916502067):
        g = -9.64699507389807E+01 * x + 3.03000000000023E+02
    elif (x < 0.7987472620841592):
        g = -2.54022986425337E+02 * x + 3.98545610859729E+02
    else:
        g = -5.71281628959223E+02 * x + 6.51955082956207E+02
    # BLUE
    if (x < 0.0997359608740309):
        b = 1.26522393162393E+02 * x + 6.65042735042735E+01;
    elif (x < 0.1983790695667267):
        b = -1.22037851037851E+02 * x + 9.12946682946686E+01;
    elif (x < 0.4997643530368805):
        b = (5.39336225400169E+02 * x + 3.55461986381562E+01) * x + 3.88081126069087E+01;
    elif (x < 0.6025972254407099):
        b = -3.79294261294313E+02 * x + 3.80837606837633E+02;
    elif (x < 0.6990141388105746):
        b = 1.15990231990252E+02 * x + 8.23805453805459E+01;
    elif (x < 0.8032653181119567):
        b = 1.68464957265204E+01 * x + 1.51683418803401E+02;
    elif (x < 0.9035796343050095):
        b = 2.40199023199020E+02 * x - 2.77279202279061E+01;
    else:
        b = -2.78813846153774E+02 * x + 4.41241538461485E+02;
    return tm.clamp(ti.Vector([r, g, b]) / 255.0, 0.0, 1.0)

@ti.kernel
def display_vel():
    for x, y in ti.ndrange(NX, NY):
        rho_i = rho(x,y)
        u_mag_i = tm.length(u(x,y,rho_i)) / U_0
        u_mag[x,y] = u_mag_i
        colour = colour_map(u_mag_i)
        pixels[x,y] = colour

@ti.kernel
def calculate_u_max():
    u_max[None] = 0.
    for x,y in  ti.ndrange(NX, NY):
        rho_i = rho(x,y)
        ti.atomic_max(u_max[None], tm.length(u(x,y,rho_i)))

def step(periodic=True, even=True):
    if periodic:
        push_periodic(f if even else buf, buf if even else f)
        # push_periodic_simple()
        # push_periodic()
        # f.copy_from(buf)
    else:
        push_lid_driven()

def run_gui():
    gui = ti.GUI("LBM D2Q9", res=(NX, NY), fast_gui = True)
    init_shearwave()
    # init_rest()
    while gui.running:
        for _ in range(100):
            push_periodic_simple()
            f.copy_from(buf)
        display_vel()
        gui.set_image(pixels)
        gui.show()

# def run_until_end():
#     init_shearwave()
#     for _ in range(10_000):
#         step()
#     gui = ti.GUI("LBM D2Q9", res=(NX, NY), fast_gui = True)
#     while gui.running:
#         display_vel()
#         gui.set_image(pixels)
#         gui.show()

# if __name__ == "__main__":
#     run_gui()

def benchmark(N=1000):
    init_shearwave()
    step(True, True)
    step(True, False)
    start = time.perf_counter_ns()
    for t in range(N):
        step(True, t%2==0)
    ti.sync()
    end = time.perf_counter_ns()
    span = (end-start)
    print(span)
    print(NX*NY*N, "lattice updates in", span*1e-9, "s =>", (NX*NY*N*1_000)//span, "MLUPS" )


def plot_umax():
    ts = []
    us = []
    init_shearwave()
    for t in range(100_000):
        step(True, t%2==0)
        if t%100==0:
            ts += [t]
            calculate_u_max()
            us += [u_max[None]]

    plt.plot(ts, us)
    plt.show()

run_gui()
# benchmark()
# plot_umax()
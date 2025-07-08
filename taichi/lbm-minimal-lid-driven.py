# type: ignore
import taichi as ti
import taichi.math as tm 
ti.init(arch=ti.gpu)
NY = 800
NX = 800
Q = 9
OMEGA = 1.7
RHO = 1.
U_0 = 0.1
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
cx = [0,1,0,-1,0,1,-1,-1,1]
cy = [0,0,1,0,-1,1,1,-1,-1]
refl = [0,3,4,1,2,7,8,5,6]
f = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
buf = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(NX,NY))

@ti.kernel
def push():
    for x, y in ti.ndrange(NX, NY):
        rho = 0.
        for i in ti.static(range(Q)):
            rho += f[x,y,i]
        u = ti.Vector([0.,0.])
        for i in ti.static(range(Q)):
            u += f[x,y,i] * ti.Vector([cx[i],cy[i]])
        u /= rho
        pixels[x,y] = tm.length(u)/U_0
        for i in ti.static(range(Q)):
            f_eq = w[i] * rho * (1 + 3*(ciui := u.x*cx[i]+u.y*cy[i]) + 4.5*ciui*ciui - 1.5*tm.dot(u,u))

            df = ti.static(RHO * U_0 / 6.) * (-1 if i==5 else 1) if (y==NY-1 and (i==5 or i==6)) else 0.
            i_refl = refl[i] if ((x==0 and cx[i]<0) or (x==NX-1 and cx[i]>0) or (y==0 and cy[i]<0) or (y==NY-1 and cy[i]>0)) else i
            drx = 0 if (x==0 and cx[i]<0) or (x==NX-1 and cx[i]>0) else cx[i]
            dry = 0 if (y==0 and cy[i]<0) or (y==NY-1 and cy[i]>0) else cy[i]
            
            buf[x+drx, y+dry, i_refl] = f_eq + df
@ti.kernel
def init_rest():
    for x, y in ti.ndrange(NX, NY):
        for i in ti.static(range(Q)):
            f[x,y,i] = w[i] * RHO 

def run_gui():
    gui = ti.GUI("LBM D2Q9", res=(NX, NY))
    init_rest()
    t = 0
    while gui.running:
        for _ in range(120):
            push()
            f.copy_from(buf)
        gui.set_image(pixels)
        gui.show(
            f'frame_{t:05d}.png' 
        )
        print(t)
        t += 1
        if t == 600:
            return
run_gui()
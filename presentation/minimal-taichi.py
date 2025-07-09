# Imports
import taichi as ti
import taichi.math as tm 

# Start Taichi
ti.init(arch=ti.gpu)

# Define constants
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

# Define fields
buf = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(NX,NY))
f = ti.field(ti.f32)
ti.root.dense(ti.k, Q).dense(ti.ij, (NX,NY)).place(f)

# Initialize a fluid at rest
@ti.kernel
def init_rest():
    for x, y in ti.ndrange(NX, NY):
        for i in ti.static(range(Q)):
            f[x,y,i] = w[i] * RHO 

# Fused collide and stream kernel
@ti.kernel
def push():
    for x, y in ti.ndrange(NX, NY):
        # Compute density
        rho = 0.
        for i in ti.static(range(Q)):
            rho += f[x,y,i]

        # Compute velocity
        u = ti.Vector([0.,0.])
        for i in ti.static(range(Q)):
            u += f[x,y,i] * ti.Vector([cx[i],cy[i]])
        u /= rho

        # Visualize results
        pixels[x,y] = tm.length(u)/U_0

        for i in ti.static(range(Q)):
            # collide
            f_eq = w[i] * rho * (1 + 3*(ciui := u.x*cx[i]+u.y*cy[i]) + 4.5*ciui*ciui - 1.5*tm.dot(u,u))

            # sliding lid boundary
            df = ti.static(RHO * U_0 / 6.) * (-1 if i==5 else 1) if (y==NY-1 and (i==5 or i==6)) else 0.
            i_refl = refl[i] if ((x==0 and cx[i]<0) or (x==NX-1 and cx[i]>0) or (y==0 and cy[i]<0) or (y==NY-1 and cy[i]>0)) else i
            drx = 0 if (x==0 and cx[i]<0) or (x==NX-1 and cx[i]>0) else cx[i]
            dry = 0 if (y==0 and cy[i]<0) or (y==NY-1 and cy[i]>0) else cy[i]
            
            buf[x+drx, y+dry, i_refl] = f[x,y,i] + ti.static(OMEGA) * (f_eq + df - f[x,y,i])

# Create and run GUI
def run_gui():
    gui = ti.GUI("LBM D2Q9", res=(NX, NY), fast_gui = True)
    init_rest()
    while gui.running:
        push()
        f.copy_from(buf)
        gui.set_image(pixels)
        gui.show()

run_gui()
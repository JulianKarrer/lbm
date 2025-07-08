# type: ignore
import taichi as ti
import taichi.math as tm 

# Start Taichi
ti.init(arch=ti.gpu)

# Define constants
NY = 300
NX = 300
Q = 9
OMEGA = 1.7
RHO = 1.
U_0 = 0.1
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
cx = [0,1,0,-1,0,1,-1,-1,1]
cy = [0,0,1,0,-1,1,1,-1,-1]

# Define fields
f = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
buf = ti.field(dtype=ti.f32, shape=(NX,NY,Q,))
pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(NX,NY))

# Initialize a shear wave
@ti.kernel
def init_shearwave():
    for x, y in ti.ndrange(NX, NY):
        for i in ti.static(range(Q)):
            u = ti.Vector([ti.static(U_0) * tm.sin((2.*tm.pi*y)/ti.cast(NY, ti.f32)), 0.])
            f[x,y,i] = w[i] * RHO * (1 + 3*(ciui := u.x*cx[i]+u.y*cy[i]) + 4.5*ciui*ciui - 1.5*tm.dot(u,u))

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

        # Collide and Stream
        for i in ti.static(range(Q)):
            f_eq = w[i] * rho * (1 + 3*(ciui := u.x*cx[i]+u.y*cy[i]) + 4.5*ciui*ciui - 1.5*tm.dot(u,u))
            buf[(x+cx[i]+NX)%NX, (y+cy[i]+NY)%NY, i] = f[x,y,i] + ti.static(OMEGA) * (f_eq - f[x,y,i])

# Create and run GUI
def run_gui():
    gui = ti.GUI("LBM D2Q9", res=(NX, NY), fast_gui = True)
    init_shearwave()
    while gui.running:
        push()
        f.copy_from(buf)
        gui.set_image(pixels)
        gui.show()

run_gui()
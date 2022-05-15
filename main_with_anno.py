from re import T
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=2)
scene.set_floor(-0.8, (0.5, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((0.5, 1, 1), 0.8, (1, 1, 1))

@ti.func
def sdf_ellipsoid(p, r ):
    k0 = (p/r).norm()
    k1 = (p/(r*r)).norm()
    ret = k0*(k0-1.0)/k1
    if k0<1.0:
        ret = (k0-1.0)*min(min(r.x,r.y),r.z)
    return ret 

@ti.func
def sdf_line(p, a, b, r ):
    h = clamp( dot(p-a,b-a)/dot(b-a,b-a), 0.0, 1.0 )
    return ( p-a - (b-a)*h ).norm()-r


@ti.func
def sdf_cylinder(  p,  h,  r ):
    d = abs(vec2(ti.sqrt(p.x*p.x+p.z*p.z),p.y)) - vec2(h,r)
    return min(max(d.x,d.y),0.0) + max(d,0.0).norm()

@ti.func
def sd_flower( uv, size, rpetals, npetals):
    uv = vec2(ti.atan2(uv.x, uv.y), uv.norm())        
    m = (fract((uv.x+ uv.y) / 3.1415926 / 2. * npetals) - 0.5) * rpetals
    return  ti.math.smoothstep(size, size-0.01, uv.y + min(m, -m))

@ti.func
def sdf_carpet(p, offset_x, offset_y, offset_z):
    t = abs(0.025* (p.x - offset_x)*(p.x - offset_x)+ 0.025 * (p.z - offset_z)*(p.z - offset_z)-offset_y+p.y)
    if t < 1.0:
        t = 0.0
    return t

@ti.func
def op_minus(d1,d2):
    return max(d1,-d2)


@ti.func
def sdf_rabit(p):

    color = vec3(0.7,0.7,0.7)

    #ear
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(5,55,0), vec3(3,6,3))
    d2 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(5,49,0), vec3(2,5,2)) 
    d3 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(5,52,1), vec3(1,10,6)) 
    d  = op_minus(min(d1,d2),d3)

    #head
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,41,3.5), vec3(2,2,2))
    if d1 < 0.0:
        color = vec3(0.0,0.0,0.0)
        d  = min(d1,d)
    
    #beard
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,37,6.5),   vec3(4,7,1))
    d2 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,38.5,6.5), vec3(4,7,1))
    if op_minus(d1,d2) < 0.0:
        color = vec3(0.0,0.0,0.0)
        d  = min(op_minus(d1,d2),d)
    
    #nose
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(0,36,5.5), vec3(1,1,2))
    if d1 < 0.0:
        color = vec3(1.0,0.0,1.0)
        d  = min(d1,d)
    d  = min(min(sdf_ellipsoid(p-vec3(0,41,0), vec3(7,7,5)),sdf_ellipsoid(p-vec3(0,33,0), vec3(8,9,7))),d)

    #body
    d1 = sdf_ellipsoid(p-vec3(0,17,0), vec3(8, 12,5))
    d2 = sdf_ellipsoid(p-vec3(0, 8,0), vec3(10, 14,5))
    d3 = sdf_ellipsoid(p-vec3(0, 0,0), vec3(8,16,5))
    if min(min(d1,d2),d3) < 0.0:
        color = vec3(int(p.x/2)%2,0.0,0.0)
        d  = min(min(min(d1,d2),d3),d)
    #arm
    d  = min(sdf_line(vec3(abs(p.x),p.y,p.z), vec3(8,17,0),  vec3(6,0,5), 3),d)
    
    #foot
    d1 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(3,-14,0), vec3(5,-33,1), 3.5)
    d2 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(5,-33,1), vec3(7,-48,2), 3)
    d3 = sdf_ellipsoid((abs(p.x),p.y,p.z-2.0)-vec3(7,-48,2), vec3(3,3,6))
    if min(min(d1,d2),d3) < 0.0:
        color = vec3(0.0,0.0,int(p.z/2)%2*0.5)
        d  = min(min(min(d1,d2),d3),d)
    
    #carpet on hand
    if abs(p.x) < 20.0  and p.z > 6.0 and p.z < 36.0 and p.y<10.0 and p.y > -20.0:
        d  = min(sdf_carpet(vec3(abs(p.x),p.y,p.z),5,10,4),d)
        color = vec3(0.7,0.2,0.2)
        if sdf_ellipsoid(vec3(p.x/20.0, 1.0, (p.z-21.0) / 15.0), vec3(1.0,1.0,1.0)) < 0.1:
            color = vec3(0.5,0.5,0.1)
        elif sd_flower(vec2(p.x/40.0, (p.z-21.0) / 30.0),0.3,0.3,7.0) < 0.1:
            color = vec3(0.5,0.5,0.3)
    
    #floor
    if  p.y == -50.0:
        if sd_flower(vec2(p.x/128.0, p.z/128.0),0.3,0.3,3.0) < 0.1:
            color = vec3(0.1,0.1,0.5)  
        d = 0.0
    
    #hat with holes
    d1 = sdf_ellipsoid(p-vec3(30,-49,40), vec3(12, 1,12))
    d2 = sdf_cylinder(p-vec3(30, -43,40), 8, 6)
    d3 = sdf_cylinder(p-vec3(33, -43,40), 2, 8)
    d4 = sdf_cylinder(p-vec3(27, -43,40), 2, 8)
    if op_minus(op_minus(min(d1,d2),d3),d4) < 0.0:
        color = vec3(0.6,0.1,0.1)  
        d = min(min(d1,d2),d3)
    return color, d

@ti.kernel
def initialize_voxels():
    for X in ti.grouped(ti.ndrange((-64, 64), (-64,64), (-64, 64))):
        color,d = sdf_rabit(X)
        if d < 0.5:
            scene.set_voxel(X, 1, color)
initialize_voxels()
scene.finish()
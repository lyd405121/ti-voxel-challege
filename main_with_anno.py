from re import T
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=3)
scene.set_floor(-1.0, (0.5, 1.0, 1.0))
scene.set_background_color((0, 0, 0))
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))

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
    pa = p-a
    ba = b-a
    h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 )
    return ( pa - ba*h ).norm()-r

@ti.func
def symetric_x(p):
    return vec3(abs(p.x),p.y,p.z)

@ti.func
def op_union(d1,d2):
    return min(d1,d2)

@ti.func
def op_minus(d1,d2):
    return max(d1,-d2)

@ti.func
def mod(x,y):
    return x - y * ti.floor(x/y)

@ti.func
def sdStar(p,  r,  n,  m):
    an = 3.141593/float(n)
    en = 3.141593/m; 
    acs = vec2(ti.cos(an),ti.sin(an))
    ecs = vec2(ti.cos(en),ti.sin(en))

    #bn = ti.floor(ti.atan2(p.x,p.y) / (2.0*an)) - an

    bn =ti.atan2(p.x,p.y)  - 2.0*an * ti.floor(ti.atan2(p.x,p.y) /(2.0*an))- an
    p = p.norm()*vec2(ti.cos(bn),abs(ti.sin(bn)))
    p -= r*acs
    p += ecs*clamp( -dot(p,ecs), 0.0, r*acs.y/ecs.y)
    return p.norm()*sign(p.x)

@ti.func
def sdCross( p, b,  r ) :
    p = abs(p)
    if p.y>p.x:
        t = p.x
        p.x = p.y
        p.y = t
    q = p - b
    k = max(q.y,q.x)
    w = q
    if k<0.0:
        w = vec2(b.y-p.x,-k)
    return sign(k)*(max(w,0.0)).norm() + r

@ti.func
def s( v0,  d,  x) :
    return ti.math.smoothstep(v0, v0+d, x)

@ti.func
def c2p( uv) :
    return vec2(ti.atan2(uv.x, uv.y), (uv).norm())

@ti.func
def flower( uv, size, rpetals, npetals):
    uv = c2p(uv);        
    uv.x +=  uv.y
    m = (fract(uv.x / 3.1415926 / 2. * npetals) - 0.5) * rpetals
    m = min(m, -m)
    return  s(size, -0.01, uv.y + m)




@ti.func
def carpet(p, offset_x, offset_y, offset_z):
    t = abs(0.025* (p.x - offset_x)*(p.x - offset_x)+ 0.025 * (p.z - offset_z)*(p.z - offset_z)-offset_y+p.y)
    if t > 1.0:
        t = 10.0
    else:
        t = 0.0
    #return abs(0.1* (p.x - offset_x)*(p.x - offset_x)-offset_y+p.y)
    #return (ti.log(p.x)-offset_y+p.y)
    return t

@ti.func
def sdf_rabit(p):

    color = vec3(0.7,0.7,0.7)
    d     = 10.0

  
    #EAR
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(5,55,0), vec3(3,6,3))
    d2 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(5,49,0), vec3(2,5,2)) 
    d3 = sdf_ellipsoid(symetric_x(p)-vec3(5,53,2), vec3(0.1,5,2)) 
    d  = op_minus(op_union(d1,d2),d3)

    # EYE
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,41,3.5), vec3(2,2,2))
    if d1 < 0.0:
        color = vec3(0.0,0.0,0.0)
        d  = op_union(d1,d)

    # MUS
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,37,6.5),   vec3(4,7,1))
    d2 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(3,38.5,6.5), vec3(4,7,1))
    if op_minus(d1,d2) < 0.0:
        color = vec3(0.0,0.0,0.0)
        d  = op_union(op_minus(d1,d2),d)

    # NOSE
    d1 = sdf_ellipsoid(vec3(abs(p.x),p.y,p.z)-vec3(0,36,5.5), vec3(1,1,2))
    if d1 < 0.0:
        color = vec3(1.0,0.0,1.0)
        d  = op_union(d1,d)

    #HEAD
    d1 = sdf_ellipsoid(p-vec3(0,41,0), vec3(7,7,5))
    d2 = sdf_ellipsoid(p-vec3(0,33,0), vec3(8,9,7))
    d  = op_union(op_union(d1,d2),d)

    #BODY
    d1 = sdf_ellipsoid(p-vec3(0,17,0), vec3(8, 12,5))
    d2 = sdf_ellipsoid(p-vec3(0, 8,0), vec3(10, 14,5))
    d3 = sdf_ellipsoid(p-vec3(0, 0,0), vec3(8,16,5))
    if op_union(op_union(d1,d2),d3) < 0.0:
        color = vec3(int(p.x/2)%2,0.0,0.0)
        d  = op_union(op_union(op_union(d1,d2),d3),d)

    #ARM
    d1 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(8,17,0),  vec3(6,0,5), 3)
    d2 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(6,0,5), vec3(4,0,10), 2.5)
    d3 = sdf_ellipsoid((abs(p.x),p.y,p.z)-vec3(4,0,10), vec3(3,3,3))
    d  = op_union(op_union(op_union(d1,d2),d3),d)

    #LEG
    d1 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(3,-14,0), vec3(5,-33,1), 3.5)
    d2 = sdf_line(vec3(abs(p.x),p.y,p.z), vec3(5,-33,1), vec3(7,-48,2), 3)
    d3 = sdf_ellipsoid((abs(p.x),p.y,p.z-2.0)-vec3(7,-48,2), vec3(3,3,6))
    if op_union(op_union(d1,d2),d3) < 0.0:
        color = vec3(0.0,0.0,int(p.z/2)%2)
        d  = op_union(op_union(op_union(d1,d2),d3),d)


    #CARPET
    if abs(p.x) < 20.0  and p.z > 6.0 and p.z < 36.0 and p.y<10.0 and p.y > -20.0:
        d1 = carpet(vec3(abs(p.x),p.y,p.z),5,10,4)
        d  = op_union(d1,d)
        #color = vec3(int(p.x/2)%2,int(p.z/2)%2,0.0)

        dd2 = sdf_ellipsoid(vec3(p.x/20.0, 1.0, (p.z-21.0) / 15.0), vec3(1.0,1.0,1.0))

        #dd = sdStar(vec2(p.x/20.0, (p.z-21.0) / 15.0),1.0,12,8)
        dd = flower(vec2(p.x/40.0, (p.z-21.0) / 30.0),0.3,0.3,7.0)
        if dd2 < 0.1:
            color = vec3(0.5,0.5,0.1)
        elif dd < 0.1:
            color = vec3(0.5,0.1,0.1)
        else:
            color = vec3(0.5,0.5,0.5)
  
        '''
        dd = sdCross(vec2(p.x/17.0, (p.z-11.5) / 5.5), vec2(1.0,0.3), 0.2)
        if dd < 0.0:
            color = vec3(1.0,0.0,0.0)
        else:
            color = vec3(0.8,0.8,0.6) 
        '''
  
    if  p.y == -60.0:
        dd = flower(vec2(p.x/128.0, p.z/128.0),0.3,0.3,3.0)
        if dd < 0.1:
            color = vec3(0.5,0.1,0.1)
        else:
            color = vec3(0.5,0.5,0.5)
        d = 0.0
    return color, d


@ti.kernel
def initialize_voxels():
    n = 64
    for X in ti.grouped(ti.ndrange((-n, n), (-n,n), (-n, n))):
        light_dir = vec3(1, 1, 1)

        eps = 0.01
        color,d = sdf_rabit(X)
        #normal= vec3( sdf_rabit(X+vec3(eps,0,0)) - d,sdf_rabit(X+vec3(0,eps,0)) - d,sdf_rabit(X+vec3(0,0,eps)) - d).normalized()
        if d < 0.5:
            scene.set_voxel(X, 1, color)
            #scene.set_voxel(X, 1, vec3(ti.random(),ti.random(),ti.random()))
initialize_voxels()
scene.finish()

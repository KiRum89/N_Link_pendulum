from sympy.physics.mechanics import *
import matplotlib.pyplot as plt
from matplotlib import animation
from sympy.utilities import lambdify
from scipy.integrate import odeint
import numpy as np
from sympy import simplify,trigsimp,symbols,atan,sin,cos, Matrix,Dummy

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter



num_links =3
q = dynamicsymbols('q1:'+str(num_links+1))
qd=dynamicsymbols('q1:'+str(num_links+1),level=1)
v = symbols('v:'+str(num_links))
x = symbols('x:'+str(num_links))
y = symbols('y:'+str(num_links))
m = symbols('m:'+str(num_links))
g = symbols('g')
l = symbols('l:'+str(num_links))
KE = 0#symbols('KE')
U = 0
vy = 0
vx = 0
y = 0
for i in range(0,num_links):
    vx+=-l[i]*sin(q[i])*qd[i]
    vy+=l[i]*cos(q[i])*qd[i]
    KE += m[i]*(vx**2+vy**2)/2
    y += l[i]*sin(q[i])
    U += m[i]*g*y

    L = KE - U

dynamic = q+qd

dummy_symbols = [Dummy() for i in dynamic]
dummy_dict=dict(zip(dynamic,dummy_symbols))


LM = LagrangesMethod(L,q)
eq=LM.form_lagranges_equations()

print('simplifying')
rhs=LM.rhs()
rhs=rhs.subs(dummy_dict)
print('start simulation')
lens = dict(zip(l,np.ones(num_links)))
masses = dict(zip(m,np.ones(num_links)))
rhs=rhs.subs(lens)
rhs = rhs.subs(masses)
rhs = rhs.subs({g:10})
F_func = lambdify(dummy_symbols,rhs,'numpy')

def F_func2(x,t):

    return np.asarray(F_func(*x).T[0])

t = np.linspace(0,10,250)
x0 = np.array([np.pi/4,np.pi/4,np.pi/4,0,0,0])
sol = odeint(F_func2,x0,t,printmessg=1)
def plot(sol,num_links,t):
    fig = plt.figure()
    ax = plt.axes(xlim=(-3,3),ylim=(-3,3))
    line,=ax.plot([],[],lw=2,marker='o',markersize=6)

    def animate(i):
        x = np.zeros(num_links+1)
        y = np.zeros(num_links+1)
        for j in range(1,num_links+1):
            x[j]=x[j-1]+np.cos(sol[i,j-1])
            y[j]=y[j-1]+np.sin(sol[i,j-1])
        line.set_data(x,y)
        #ax.scatter(x,y)
        return line,

    ani = FuncAnimation(fig, animate,frames=250)

    ani.save("n_link_pend.gif", dpi=300, writer=PillowWriter(fps=25))    

plot(sol,num_links,t)
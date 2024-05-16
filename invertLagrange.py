# inverts the lagrange numerically
from scipy.optimize import fsolve

def forward(x,*ptvecin):
    # forward lagrange interpolation
    p,q = x
    ptvec= ptvecin[0]
    n1 = 0.25*(1-p)*(1-q)
    n2 = 0.25*(1+p)*(1-q)
    n3 = 0.25*(1+p)*(1+q)
    n4 = 0.25*(1-p)*(1+q)
    val1 = ptvec[0][0]*n1+ptvec[1][0]*n2+ptvec[2][0]*n3+ptvec[3][0]*n4
    val2 = ptvec[0][1]*n1+ptvec[1][1]*n2+ptvec[2][1]*n3+ptvec[3][1]*n4
    #print(p,q,val1,val2)
    return [val1,val2]


def reverse_error(x,*args):
    # error function for finding p,q roots
    #print(args)
    rs = args[0][0]
    ptvec = args[0][1]
    x1 = forward(x,ptvec)
    err = [(x1[0]-rs[0])*(x1[0]-rs[0]) ,  (x1[1]-rs[1])*(x1[1]-rs[1]) ]
    print(x,err,x1,rs)
    return err

def reverse(r,s,*ptvec):
    data = [[r,s],ptvec[0]]
    x0 = [0.25,0.25]
    p = fsolve(reverse_error,x0,data)
    return p

def map(r,s,rspts,xypts):
    #returns x, y for a given rs. 
    # domains are given in the rspts and xypts
    p = reverse(r,s,rspts)
    print("inverse",p)
    x = forward(p,xypts)
    print("forward",x)
    return x

#Tests
# simple reverse interpolation
ptvec = [ [0,0],[1,0],[1,1],[0,1]]
r = 0.25
s = 0.5
pq = reverse(r,s,ptvec)
print(pq)
# test it with forward
print(forward(pq,ptvec))

#  map test
ptvec = [ [100,200],[300,210],[400,600],[105,590]]
xyvec = [ [0,0],[600,0],[640,480],[10,500]]
print(map(200,300,ptvec,xyvec))

# do the middle of the map
rtest = (100+300+400+105)/4.0
stest = (200+210+600+590)/4.0
print(map(rtest,stest,ptvec,xyvec))
#middle of xy 
xmid  = (600+640+10)/4
ymid = (480+500)/4
print(xmid,ymid)

#Works great. 
# inverse [-3.59654900e-16  1.05549671e-16]
# forward [312.4999999999999, 245.0]
# [312.4999999999999, 245.0]
# 312.5 245.0

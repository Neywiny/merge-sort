def writer(arr: list):
    ret = ''
    for val in arr:
	ret += str(val) + ','
    return ret[:-1]
def bLen(*args):
    """bLen(*args) -> len(args)
    a better version of len, which takes args because in python3 everything is a generator"""
    return len(args)
true = True
false = False
alphabet = [chr(i) for i in range(97, 97+26)]
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z = alphabet
ALPHABET = [chr(i) for i in range(65, 65+26)]
A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z = ALPHABET
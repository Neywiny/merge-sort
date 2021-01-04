
def bLen(*args):
	"""bLen(*args) -> len(args).
	
	A better version of len, which takes args because in python3 everything is a generator"""
	return len(args)
true = True
false = False
alphabet = [chr(i) for i in range(97, 97+26)]
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z = alphabet
ALPHABET = [chr(i) for i in range(65, 65+26)]
A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z = ALPHABET

def bettorbar(ax, *args, **kwargs):
	"""A better error bar function.
	Adds kwargs: elinestyle, ecolor
	Attempts to set zorder to be the same for all lines"""
	mplkwargs = kwargs.copy()
	mplkwargs.pop('ecolor', None)
	mplkwargs.pop('elinestyle', None)
	err = ax.errorbar(*args, **mplkwargs)
	color = kwargs.get('ecolor', kwargs['c'])
	zorder = err[0].zorder
	err[2][0].set(linestyle = kwargs.get('elinestyle', kwargs.get('ls', kwargs.get('linestyle'))), color=color, zorder=zorder)
	err[1][0].set(color=color, zorder=zorder)
	err[1][1].set(color=color, zorder=zorder)
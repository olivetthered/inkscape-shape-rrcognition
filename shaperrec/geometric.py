import numpy
import sys
numpy.set_printoptions(precision=3)

# *************************************************************
# debugging 
def void(*l):
    pass
def debug_on(*l):
    sys.stderr.write(' '.join(str(i) for i in l) +'\n') 
debug = void
#debug = debug_on

# *************************************************************
# a list of geometric helper functions 
def toArray(parsedList):
    """Interprets a list of [(command, args),...]
    where command is a letter coding for a svg path command
          args are the argument of the command
    """
    interpretCommand = {
        'C': lambda x, prevL : x[-2:], # bezier curve. Ignore the curve.
        'L': lambda x, prevL : x[0:2],
        'M': lambda x, prevL : x[0:2],
        'Z': lambda x, prevL : prevL[0],
        }

    points =[]
    for i, (c, arg) in enumerate(parsedList):
        #debug('toArray ', i, c , arg)
        if c == 'H' or c == 'V' : continue
        newp = interpretCommand[c](arg, points)
        points.append( newp)
    a=numpy.array( points )

    # Some times we have points *very* close to each other
    # these do not bring any meaning full info, so we remove them
    #
    x, y, w, h = computeBox(a)
    sizeC = 0.5*(w+h)
    #deltas = numpy.zeros((len(a),2) )
    deltas = a[1:] - a[:-1] 
    #deltas[-1] = a[0] - a[-1]
    deltaD = numpy.sqrt(numpy.sum( deltas**2, 1 ))
    sortedDind = numpy.argsort(deltaD)
    # expand longuest segments
    nexp = int(len(deltaD)*0.9)
    newpoints=[ None ]*len(a)
    medDelta = deltaD[sortedDind[int(len(deltaD)/2)] ]
    for i, ind in enumerate(sortedDind):
        if deltaD[ind]/sizeC<0.005: continue
        if i>nexp:
            np = int(deltaD[ind]/medDelta)
            pL = [a[ind]]
            #print i,'=',ind,'adding ', np,'  _ ', deltaD[ind], a[ind], a[ind+1]
            for j in range(np-1):
                f = float(j+1)/np
                #print '------> ', (1-f)*a[ind]+f*a[ind+1]
                pL.append( (1-f)*a[ind]+f*a[ind+1] )
            newpoints[ind] = pL
        else:
            newpoints[ind]=[a[ind]]
    if(D(a[0], a[-1])/sizeC > 0.005 ) :
        newpoints[-1]=[a[-1]]

    points = numpy.concatenate([p for p in newpoints if p!=None] )
    ## print ' medDelta ', medDelta, deltaD[sortedDind[-1]]
    ## print len(a) ,' ------> ', len(points)

    rel_norms = numpy.sqrt(numpy.sum( deltas**2, 1 )) / sizeC
    keep = numpy.concatenate([numpy.where( rel_norms >0.005 )[0], numpy.array([len(a)-1])])

    #return a[keep] , [ parsedList[i] for i in keep]
    #print len(a),' ',len(points)
    return points, []

rotMat = numpy.matrix( [[1, -1], [1, 1]] )/numpy.sqrt(2)
unrotMat = numpy.matrix( [[1, 1], [-1, 1]] )/numpy.sqrt(2)

def setupKnownAngles():
    pi = numpy.pi
    #l = [ i*pi/8 for i in range(0, 9)] +[ i*pi/6 for i in [1,2,4,5,] ]
    l = [ i*pi/8 for i in range(0, 9)] +[ i*pi/6 for i in [1, 2, 4, 5,] ] + [i*pi/12 for i in (1, 5, 7, 11)]
    knownAngle = numpy.array( l )
    return numpy.concatenate( [-knownAngle[:0:-1], knownAngle ])
knownAngle = setupKnownAngles()

_twopi =  2*numpy.pi
_pi = numpy.pi

def deltaAngle(a1, a2):
    d = a1 - a2 
    return d if d > -_pi else d+_twopi

def closeAngleAbs(a1, a2):
    d = abs(a1 - a2 )
    return min( abs(d-_pi), abs( _twopi - d), d)

def deltaAngleAbs(a1, a2):
    return abs(in_mPi_pPi(a1 - a2 ))

def in_mPi_pPi(a):
    if(a>_pi): return a-_twopi
    if(a<-_pi): return a+_twopi
    return a
vec_in_mPi_pPi = numpy.vectorize(in_mPi_pPi)

def D2(p1, p2):
    return ((p1-p2)**2).sum()

def D(p1, p2):
    return numpy.sqrt(D2(p1, p2) )

def norm(p):
    return numpy.sqrt( (p**2).sum() )

def computeBox(a):
    """returns the bounding box enclosing the array of points a
    in the form (x,y, width, height) """
    xmin, ymin = a[:, 0].min(), a[:, 1].min()
    xmax, ymax = a[:, 0].max(), a[:, 1].max()

    return xmin, ymin, xmax-xmin, ymax-ymin

def dirAndLength(p1, p2):
    #l = max(D(p1, p2) ,1e-4)
    l = D(p1, p2)
    uv = (p1-p2)/l
    return l, uv

def length(p1, p2):
    return numpy.sqrt( D2(p1, p2) )

def barycenter(points):
    """
    """
    return points.sum(axis=0)/len(points)
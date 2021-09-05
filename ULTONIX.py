from queue import PriorityQueue
import numpy as np
import math
import cv2
import gym
import time
import pybullet as p
import pybullet_data
import cv2.aruco as aruco
import pix_main_arena
import os


#black -1
#pink 0
#green 1
#yellow 2        u  d   l r
#61,62,63,6blue 3={oneway- 4 covidhospital=7 noncovid=8}
#white 4
#red 5

'''
0  1  2  3  4  5
6  7  8  9  10 11
12 13 14 15 16 17
18 19 20 21 22 23
24 25 26 27 28 29
30 31 32 33 34 35
'''

##please take all the roi from the same corner 


inf =1000

n=12
m=12

graph={}

covidhospital=(-1,-1)
noncovidhospital=(-1,-1)
patientspos=[]

colmat=np.zeros((n,m))
cost=np.zeros((n,m))
p.startStateLogging( p.STATE_LOGGING_VIDEO_MP4, "output.mp4" )

def creategraph():
	d=[-1,0,1]
	for i in range(n):
		for j in range(m):
			v=i*n+j
			graph[v]=[]
			if(colmat[i][j]==-1):
				continue
			if colmat[i][j]>60:
				if(colmat[i][j]%10==1):
					if(i-1>=0 and colmat[i-1][j]!=-1):
						if(colmat[i-1][j]<60):
							graph[v].append((i-1)*n+j)
						elif(colmat[i-1][j]%10!=2):
							graph[v].append((i-1)*n+j)
					if(j+1<m and colmat[i][j+1]!=-1):
						if(colmat[i][j+1]<60):
							graph[v].append((i)*n+j+1)
						elif(colmat[i][j+1]%10!=3):
							graph[v].append(i*n+j+1)
					if(j-1>=0 and colmat[i][j-1]!=-1):
						if(colmat[i][j-1]<60):
							graph[v].append((i)*n+j-1)
						elif(colmat[i][j-1]%10!=4):
							graph[v].append(i*n+j-1)

				elif colmat[i][j]%10==2:
					if(i+1<n and colmat[i+1][j]!=-1):
						if(colmat[i+1][j]<60):
							graph[v].append((i+1)*n+j)
						elif(colmat[i+1][j]%10!=1):
							graph[v].append((i+1)*n+j)
					if(j-1>=0 and colmat[i][j-1]!=-1):
						if(colmat[i][j-1]<60):
							graph[v].append(i*n+j-1)
						elif(colmat[i][j-1]%10!=4):
							graph[v].append(i*n+j-1)
					if(j+1<m and colmat[i][j+1]!=-1):
						if(colmat[i][j+1]<60):
							graph[v].append(i*n+j+1)
						elif(colmat[i][j+1]%10!=3):
							graph[v].append(i*n+j+1)
				elif colmat[i][j]%10==3:
					if(j-1>=0 and colmat[i][j-1]!=-1):
						if(colmat[i][j-1]<60):
							graph[v].append(i*n+j-1)
						elif(colmat[i][j-1]%10!=4):
							graph[v].append(i*n+j-1)
					if(i-1>=0 and colmat[i-1][j]!=-1):
						if(colmat[i-1][j]<60):
							graph[v].append((i-1)*n+j)
						elif(colmat[i-1][j]%10!=2):
							graph[v].append((i-1)*n+j)
					if(i+1<m and colmat[i+1][j]!=-1):
						if(colmat[i+1][j]<60):
							graph[v].append((i+1)*n+j)
						elif(colmat[i+1][j]%10!=1):
							graph[v].append((i+1)*n+j)					
				elif colmat[i][j]%10==4:
					if(j+1<m and colmat[i][j+1]!=-1):
						if(colmat[i][j+1]<60):
							graph[v].append(i*n+j+1)
						elif(colmat[i][j+1]%10!=3):
							graph[v].append(i*n+j+1);
					if(i-1>=0 and colmat[i-1][j]!=-1):
						if(colmat[i-1][j]<60):
							graph[v].append((i-1)*n+j)
						elif(colmat[i-1][j]%10!=2):
							graph[v].append((i-1)*n+j);
					if(i+1<n and colmat[i+1][j]!=-1):
						if(colmat[i+1][j]<60):
							graph[v].append((i+1)*n+j)
						elif(colmat[i+1][j]%10!=1):
							graph[v].append((i+1)*n+j);	
				continue
			for k in range(3):
				for p in range(3):
					if(abs(d[k]-d[p])==1):
						x=i+d[k]
						y=j+d[p]
						if(x>=0 and x<n and y>=0 and y<m):
							if(colmat[x][y]==-1):
								continue
							if(colmat[x][y]<60):
								graph[v].append(x*n+y)
							elif(colmat[x][y]%10==1):
								if d[k]!=1 or d[p]!=0:
									graph[v].append(x*n+y)
							elif(colmat[x][y]%10==2):
								if d[k]!=-1 or d[p]!=0:
									graph[v].append(x*n+y)
							elif(colmat[x][y]%10==3):
								if d[k]!=0 or d[p]!=1:
									graph[v].append(x*n+y)
							elif(colmat[x][y]%10==4):
								if d[k]!=0 or d[p]!=-1:
									graph[v].append(x*n+y)








def direction(approx):
	x1=approx[0][0]
	y1=approx[0][1]
	x2=approx[1][0]
	y2=approx[1][1]
	x3=approx[2][0]
	y3=approx[2][1]
	thresh=5
	if(abs(x1-x2)<=thresh):
		if(x3<x1):
			return("left")
		else :
			return("right")
	elif(abs(x1-x3)<=thresh):
		if(x2<x1):
			return("left")
		else:
			return("right")
	elif(abs(x2-x3)<=thresh):
		if(x1<x2):
			return("left")
		else:
			return("right")
	elif(abs(y1-y2)<=thresh):
		if(y3<y1):
			return("up")
		else:
			return("down")
	elif(abs(y2-y3)<=thresh):
		if(y1<y3):
			return("up")
		else:
			return("down")
	elif(abs(y1-y3)<=thresh):
		if(y2<y3):
			return("up")
		else:
			return("down")


def predictshape(mask):
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.erode(mask, kernel,iterations=1)
    mask=cv2.dilate(mask, kernel,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    #cv2.imshow(str(k),mask)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cntr in contours:
        area=cv2.contourArea(cntr)
        if area>=300:
            per=cv2.arcLength(cntr, closed=True)
            approx=cv2.approxPolyDP(cntr, epsilon=0.036*per, closed=True)
            #print(len(approx))
            return(approx)


def predictcol(box):
	# cv2.imshow("pd",box)
	# cv2.waitKey()
	r,b,g=box[50,50]
	##print(r,g,b)
	if(r>=200 and b>=200 and g>=200):
		return 4#white
	elif(r>=200 and b>=100 and g>=200):
		return 0#pink
	elif(r<=20 and b>=200 and g<=20):
		return 1#green
	elif(r<=5 and b<=5 and g<=5):
		return -1#black
	elif(r<=10 and b>=200 and g>=200):
		return 2#yellow
	elif(r>=200 and b<=10 and g<=10):
		return 3#blue
	elif(r<=10 and b<=10 and g>=100):
		return 5#red
	else:
		return(10)


def cornercolor(box):
	
	# cv2.imshow("my", box)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	r,b,g=box[30,30]
	if(r>=200 and b>=200 and g>=200):
		return 4#white
	elif(r>=200 and b>=100 and g>=200):
		return 0#pink
	elif(r<=20 and b>=200 and g<=20):
		return 1#green
	elif(r<=5 and b<=5 and g<=5):
		return -1#black
	elif(r<=10 and b>=200 and g>=200):
		return 2#yellow
	elif(r>=200 and b<=10 and g<=10):
		return 3#blue
	elif(r<=10 and b<=10 and g>=100):
		return 5#red
	else:
		return(10)



def colormatrix(img):
	global covidhospital
	global noncovidhospital
	for i in range(n):
		for j in range(m):
			box=img[i*100:(i+1)*100,j*100:(j+1)*100]
			# cv2.imshow("winname+{}".format(i*n+j), box)
			# cv2.waitKey(100)
			col=predictcol(box)
			if(col==3):
				lowerb=np.array([164,0,0])
				upperb=np.array([255,164,111])

				mask=cv2.inRange(box, lowerb, upperb)
				approx=predictshape(mask)
				approx=approx.reshape(-1,2)
				# cv2.imshow("mmm",mask)
				# cv2.waitKey()
				#print(len(approx))
				if len(approx) is 3:
					dir=direction(approx)
					if(dir=='up'):
						#print("up")
						cornercol=cornercolor(box)
						if(cornercol==4):
							colmat[i][j]=61
						elif cornercol==1:
							colmat[i][j]=71
						elif cornercol==2:
							colmat[i][j]=81
						else:
							colmat[i][j]=91
					elif(dir=='down'):
						#print("down")
						cornercol=cornercolor(box)
						if(cornercol==4):
							colmat[i][j]=62
						elif cornercol==1:
							colmat[i][j]=72
						elif cornercol==2:
							colmat[i][j]=82
						else:
							colmat[i][j]=92
					elif(dir=="left"):
						#print("left")
						cornercol=cornercolor(box)
						if(cornercol==4):
							colmat[i][j]=63
						elif cornercol==1:
							colmat[i][j]=73
						elif cornercol==2:
							colmat[i][j]=83
						else:
							colmat[i][j]=93
					else:
						#print("right")
						cornercol=cornercolor(box)
						if(cornercol==4):
							colmat[i][j]=64
						elif cornercol==1:
							colmat[i][j]=74
						elif cornercol==2:
							colmat[i][j]=84
						else:
							colmat[i][j]=94
				elif len(approx) is 4:
					covidhospital=(i,j)
					colmat[i][j]=7
				else:
					noncovidhospital=(i,j)
					colmat[i][j]=8
			else :
				if(col==0):
					patientspos.append((i,j))
				colmat[i][j]=col  



def dijktras(src,dest,cost):
	q=PriorityQueue()
	par={}
	par[src]=src
	(srcx,srcy)=src
	dist=np.zeros((n,m))
	for i in range(n):
		for j in range(m):
			dist[i][j]=inf
	dist[srcx][srcy]=0
	destx,desty=dest
	q.put((0,src))
	d=[1,-1,0]
	while not q.empty():
		wt,(x,y)=q.get()
		for i in range(len(graph[x*n+y])):
			v=graph[x*n+y][i]
			dx=v//n
			dy=v%n
			if(dist[dx][dy]>dist[x][y]+cost[dx][dy]):
				dist[dx][dy]=dist[x][y]+cost[dx][dy]
				par[(dx,dy)]=(x,y)
				q.put((dist[dx][dy],(dx,dy)))


	path=[]
	(destx,desty)=dest
	path.append(dest)
	#print(dest)
	while destx!=srcx or desty!=srcy:
		(destx,desty)=par[(destx,desty)]
		#print((destx,desty))
		path.append((destx,desty))

	path.reverse()
	for i in range(len(path)):
		x,y=path[i]
		if(i!=len(path)-1):
			print("({} ,{})->".format(x,y),end='')
		else:
			print("({} ,{})".format(x,y))

	return path




def movement(corner,dest):
	x=(corner[0][0]+corner[2][0])//2
	y=(corner[0][1]+corner[2][1])//2
	#print(x,y)
	vxreq=(dest[0]-x)
	vyreq=(dest[1]-y)
	modv=np.sqrt(vxreq**2+vyreq**2)
	vxreq=vxreq/modv
	vyreq=vyreq/modv
	botvx=(corner[0][0]-corner[3][0])
	botvy=(corner[0][1]-corner[3][1])
	mod=np.sqrt(botvx**2+botvy**2)
	botvx=botvx/mod
	botvy=botvy/mod
	botvec=complex(botvx,botvy)
	vec=complex(vxreq,vyreq)
	angle=np.angle(botvec/vec,deg=True)
	#print("angle :",angle)
	if(-7.2<=angle and angle<=7.2):
		return("straight")
	elif(angle>=7.2):
		return("left")
	else:
		return("right")


def posdetect(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ARUCO_PARAMETERS = aruco.DetectorParameters_create()
	ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
	img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))
	if ids is None:
		return([])
	corners=np.array(corners)
	corner=corners[0]
	corner=corner.reshape((corner.shape[1],corner.shape[2]))
	return(corner)

def dist(img,dest):
	corner=posdetect(img)
	if(len(corner)==0):
		return -1
	y=(corner[0][0]+corner[2][0])//2
	x=(corner[0][1]+corner[2][1])//2
	return(np.sqrt((x-dest[1])**2+(y-dest[0])**2))
'''
for i in range(n):
	for j in range(m):
		v=i*n+j;
		print(v)
		for k in range(len(graph[v])):
			print("{},{}".format(graph[v][k]//n,graph[v][k]%n),end=" ")


		print("next")
'''

def getpatientstatus(box):
	lowerb=np.array([164,0,0])
	upperb=np.array([255,164,111])
	mask=cv2.inRange(box, lowerb, upperb)
	'''
	cv2.imshow("box",mask)
	cv2.waitKey(10)
	cv2.destroyAllWindows()
	'''
	approx=predictshape(mask)
	if(len(approx)==4):
		return("covid")
	else:
		return("noncovid")


'''
img=cv2.imread("new.jpg")
img=cv2.resize(img, (100*n,100*m))
for i in range(n):
	for j in range(m):
		box=img[i*100:(i+1)*100,j*100:(j+1)*100]
		cv2.imshow("{},{}".format(i,j), box)
		col=predictcol(box)
		print("{},{}={}".format(i,j,col))

cv2.waitKey(0)
'''


print(colmat)


def createcostmatrix():
	global cost
	for i in range(n):
		for j in range(m):
			if colmat[i][j]==-1:
				cost[i][j]=inf
			elif colmat[i][j]==0:
				cost[i][j]=1
			elif colmat[i][j]==4:
				cost[i][j]=1
			elif colmat[i][j]==1:
				cost[i][j]=2
			elif colmat[i][j]==5:
				cost[i][j]=4
			elif colmat[i][j]==2:
				cost[i][j]=3
			else :
				if colmat[i][j]//10==6:
					cost[i][j]=1
				elif colmat[i][j]//10==7:
					cost[i][j]=2
				elif colmat[i][j]//10==8:
					cost[i][j]=3
				else:
					cost[i][j]=4


def printgraph():
	for i in range(n):
		for j in range(m):
			v=i*n+j;
			print(v)
			for k in range(len(graph[v])):
				print("({},{})".format(graph[v][k]//n,graph[v][k]%n),end=" ")

			print("next")




def imgcrop(img,r):
	img=img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
	img=cv2.resize(img,(n*100,m*100))
	return(img)



def stop():
	g=0
	while g<20:
		g+=1
		p.stepSimulation()
		env.move_husky(0.0,0.0,0.0,0.0)


def moveforward():
	cnt=0
	while cnt<=100:
		p.stepSimulation()
		if cnt%35==0:
			img=env.camera_feed()
			img=imgcrop(img, r)
		cnt+=1
		env.move_husky(7.0,7.0,7.0,7.0)
        #cv2.waitKey(5)

def moveright():
	cnt=0
	while(cnt<=100):
		if cnt%35==0:
			img=env.camera_feed()
			img=imgcrop(img, r)
		p.stepSimulation()
		env.move_husky(7.0,-7.0,7.0,-7.0)
		cnt+=1


def moveleft():
	cnt=0
	while(cnt<=100):
		if(cnt%35==0):
			img=env.camera_feed()
			img=imgcrop(img, r)
		p.stepSimulation()
		env.move_husky(-7.0,7.0,-7.0,7.0)
		cnt+=1


def adjust():
	cnt=0
	while(cnt<=40):
		if(cnt%35==0):
			img=env.camera_feed()
			img=imgcrop(img, r)
		p.stepSimulation()
		env.move_husky(-6.2,6.2,-6.2,6.2)
		cnt+=1


def go(dest,r):
	while True:
		img=env.camera_feed()
		img=imgcrop(img,r)
		distance=dist(img,dest)
		#print(distance)
		if(distance==-1):
			adjust()
			stop()
			continue
		if(distance<27.0):
			break
		corner=posdetect(img)
		if(len(corner)==0):
			adjust()
			stop()
			continue
		move=movement(corner, dest)
		#print(move)
		if(move=="right"):
			moveright()
			stop()
		elif(move=="left"):
			moveleft()
			stop()
		else:
			moveforward()
			stop()

if __name__=="__main__":
	xs=100
	ys=100
	xm=91
	ym=92
	env = gym.make("pix_main_arena-v0")
	# result = cv2.VideoWriter('filename.avi', 
    #                      cv2.VideoWriter_fourcc(*'MJPG'),fps = 20)
	# result.write(env.camera_feed())
	time.sleep(1)
	env.remove_car()
	img =env.camera_feed()
	r=cv2.selectROI("outer-TAKE WHOLE IMAGE from left corner",img)
	img=imgcrop(img,r)
	img=env.camera_feed()
	r2=cv2.selectROI("inne-TAKE ONLY WITH COLORED BOXES NO BLACk from left corner",img)
	cv2.destroyAllWindows()
	img=imgcrop(img, r2)
	colormatrix(img)
	createcostmatrix()
	x,y=covidhospital
	print(patientspos)
	if(x!=-1):
		cost[x][y]=500
	x,y=noncovidhospital
	if(x!=-1):
		cost[x][y]=500
	for i in range(len(patientspos)):
		x,y=patientspos[i]
		cost[x][y]=500
	print(cost)
	creategraph()
	printgraph()
	print(colmat)
	print(noncovidhospital)
	print(covidhospital)
	env.respawn_car()
	while True:
		mnid=-1
		mn=inf
		src=(11,11)
		for i in range(len(patientspos)):
			path=dijktras(src,patientspos[i], cost)
			if(mn>len(path)):
				mn=len(path)
				mnid=i

		print(patientspos[mnid])
		path=dijktras(src,patientspos[mnid], cost)
		k=1
		while k<len(path)-1:
			x,y=path[k]
			dest=[y*ym+ys,x*xm+xs]
			go(dest,r)
			#print("reacheddest")
			k+=1


		x,y=path[len(path)-1]
		print(x,y)
		env.remove_cover_plate(x,y)
		time.sleep(1)
		img =env.camera_feed()
		img=imgcrop(img,r2)
		patientspos.remove(patientspos[mnid])
		u=getpatientstatus(img[x*100:(x+1)*100,y*100:(y+1)*100])
		dest=[y*ym+ys,x*xm+xs]
		go(dest,r)
		print(u)
		px=x
		py=y
		if(u=='noncovid'):
			dest=noncovidhospital
			path=dijktras((px,py), dest, cost)
			k=1
			while k<len(path)-1:
				x,y=path[k]
				dest=[y*ym+ys,x*xm+xs]
				go(dest,r)
				k+=1
			time.sleep(0.5)
			x,y=noncovidhospital
			dest=[y*ym+ys,x*xm+xs]
			go(dest,r)
			stop();
		else:
			dest=covidhospital
			path=dijktras((px,py), dest, cost)
			k=1
			while k<len(path)-1:
				x,y=path[k]
				dest=[y*ym+ys,x*xm+xs]
				go(dest,r)
				k+=1
			x,y=covidhospital
			time.sleep(0.5)
			dest=[y*ym+ys,x*xm+xs]
			go(dest,r)


		mnid=-1
		for i in range(len(patientspos)):
			mnid=i

		if(mnid==-1):
			break

		prevdest=path[len(path)-1]
		dest=patientspos[mnid]
		path=dijktras(prevdest,dest,cost)
		k=1
		while k<len(path)-1:
			x,y=path[k]
			dest=[y*ym+ys,x*xm+xs]
			go(dest,r)
			k+=1

		dest=patientspos[mnid]
		x,y=patientspos[mnid]
		env.remove_cover_plate(x,y)
		time.sleep(0.75)
		img =env.camera_feed()
		img=imgcrop(img,r2)
		print("{},{}".format(x,y))
		patientspos.remove(patientspos[mnid])
		u=getpatientstatus(img[x*100:(x+1)*100,y*100:(y+1)*100])
		print(u)
		dest=[y*ym+ys,x*xm+xs]
		go(dest,r)
		if(u=='noncovid'):
			prevdest=(x,y)
			dest=noncovidhospital
			path=dijktras(prevdest, dest, cost)
			k=1
			while k<len(path)-1:
				x,y=path[k]
				dest=[y*ym+ys,x*xm+xs]
				go(dest,r)
				k+=1
			x,y=noncovidhospital
			dest=[y*ym+ys,x*xm+xs]
			time.sleep(0.5)
			go(dest,r)
		else:
			prevdest=(x,y)
			dest=covidhospital
			path=dijktras(prevdest, dest, cost)
			k=1
			while k<len(path)-1:
				x,y=path[k]
				dest=[y*ym+ys,x*xm+xs]
				go(dest,r)
				k+=1
			x,y=covidhospital
			time.sleep(0.5)
			dest=[y*ym+ys,x*xm+xs]
			go(dest,r)

		break

	for i in range(10):
		print("COVID GONE")

	print("|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-| IIT BHU REOPENED |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|")
	time.sleep(2)



#print(cost)
'''src=(5,5)
dest=(0,0)
dijktras(src, dest,cost)
'''


import numpy   as np
import nibabel as nib # to read NII files
import matplotlib.pyplot as plt
from numpy.lib import histograms
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product

class CTscan:
    def __init__(self,index): #initialization
        self.index=index
        return
    def read_nii(self,filepath): 
        index=self.index  
        ct_scan = nib.load(filepath)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        Nr,Nc,Nimages=array.shape# Nr=512,Nc=512,Nimages=301
        self.Nr=Nr
        self.Nc=Nc
        self.Nimages=Nimages
        sct=array[...,index]
        self.sct=sct
        return(array)  
    def hist(self):
        sct=self.sct
        index=self.index
        a=np.histogram(sct,200,density=True)
        plt.figure()
        plt.plot(a[1][0:200],a[0])
        plt.title('Histogram of CT values in slice '+str(index))
        plt.grid()
        plt.xlabel('value')
        #plt.show()    
        return
    def Kmeans(self,Ncluster,ifind):
        sct=self.sct
        Nr=self.Nr
        Nc=self.Nc
        kmeans = KMeans(n_clusters=Ncluster,random_state=0)# instantiate Kmeans
        self.kmeans=kmeans
        A=sct.reshape(-1,1)# Ndarray, Nr*Nc rows, 1 column
        self.A=A
        kmeans.fit(A)# run Kmeans on A
        self.kmeans_centroids=kmeans.cluster_centers_.flatten()#  centroids/quantized colors
        for k in range(Ncluster):
            ind=(kmeans.labels_==k)# indexes for which the label is equal to k
            A[ind]=self.kmeans_centroids[k]# set the quantized color
        sctq=A.reshape(Nr,Nc)# quantized image
        vm=sct.min()
        vM=sct.max()

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(sct, cmap='bone',interpolation="nearest")
        ax1.set_title('Original image')
        ax2.imshow(sctq,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
        ax2.set_title('Quantized image')

        ii=self.kmeans_centroids.argsort()# sort centroids from lowest to highest
        ind_clust=ii[ifind]# get the index of the desired cluster 
        ind=(kmeans.labels_==ind_clust)# get the indexes of the pixels having the desired color
        D=A*np.nan
        D[ind]=1# set the corresponding values of D  to 1
        D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping
        self.D=D
        plt.figure()
        plt.imshow(D,interpolation="nearest")
        plt.title('Image used to identify lungs')
        plt.show()
        return    
    def plotSample(array_list, color_map = 'nipy_spectral'):
        '''
        Plots a slice with all available annotations
        '''
        plt.figure(figsize=(18,15))

        plt.subplot(1,4,1)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.title('Original Image')

        plt.subplot(1,4,2)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
        plt.title('Lung Mask')

        plt.subplot(1,4,3)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[2], alpha=0.5, cmap=color_map)
        plt.title('Infection Mask')

        plt.subplot(1,4,4)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
        plt.title('Lung and Infection Mask')

        plt.show()
        return
    def filterImage(self,D,NN):
        """D = image (matrix) to be filtered, Nr rows, N columns, scalar values (no RGB color image)
        The image is filtered using a square kernel/impulse response with side 2*NN+1"""
        E=D.copy()
        E[np.isnan(E)]=0
        Df=E*0
        Nr,Nc=D.shape
        rang=np.arange(-NN,NN+1)
        square=np.array([x for x in product(rang, rang)])
        #square=np.array([[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]])
        for kr in range(NN,Nr-NN):
            for kc in range(NN,Nc-NN):
                ir=kr+square[:,0]
                ic=kc+square[:,1]
                Df[kr,kc]=np.sum(E[ir,ic])# Df will have higher values where ones are close to each other in D
        return Df/square.size

    def useDBSCAN(self,D,z,epsv,min_samplesv):
            """D is the image to process, z is the list of image coordinates to be
            clustered"""
            Nr,Nc=D.shape
            clusters =DBSCAN(eps=epsv,min_samples=min_samplesv,metric='euclidean').fit(z)
            a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
            Nclust_DBSCAN=len(a)-1
            Npoints_per_cluster=Npoints_per_cluster[1:]# remove numb. of outliers (-1)
            ii=np.argsort(-Npoints_per_cluster)# from the most to the less populated clusters
            Npoints_per_cluster=Npoints_per_cluster[ii]
            C=np.zeros((Nr,Nc,Nclust_DBSCAN))*np.nan # one image for each cluster
            info=np.zeros((Nclust_DBSCAN,5),dtype=float)
            for k in range(Nclust_DBSCAN):
                i1=ii[k] 
                index=(clusters.labels_==i1)
                jj=z[index,:] # image coordinates of cluster k
                C[jj[:,0],jj[:,1],k]=1 # Ndarray with third coord k stores cluster k
                a=np.mean(jj,axis=0).tolist()
                b=np.var(jj,axis=0).tolist()
                info[k,0:2]=a #  store coordinates of the centroid
                info[k,2:4]=b # store variance
                info[k,4]=Npoints_per_cluster[k] # store points in cluster
            return C,info,clusters
    def find_lungs(self,eps,min_samples):
        D=self.D
        C,centroids,clust=self.useDBSCAN(D,np.argwhere(D==1),eps,min_samples)
        self.centroids=centroids
        # we want left lung first. If the images are already ordered
        # then the center along the y-axis (horizontal axis) of C[:,:,0] is smaller
        if centroids[1,1]<centroids[0,1]:# swap the two subimages
            print('swap')
            tmp = C[:,:,0]*1
            C[:,:,0] = C[:,:,1]*1
            C[:,:,1] = tmp
            tmp=centroids[0,:]*1
            centroids[0,:]=centroids[1,:]*1
            centroids[1,:]=tmp
        LLung = C[:,:,0].copy()  # left lung
        RLung = C[:,:,1].copy()  # right lung

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLung,interpolation="nearest")
        ax1.set_title('Left lung mask - initial')
        ax2.imshow(RLung,interpolation="nearest")
        ax2.set_title('Right lung mask - initial')
        plt.show()
        return
    def quantized(self):
        D=self.D
        Nr=self.Nr
        Nc=self.Nc
        A=self.A
        centroids=self.centroids
        kmeans=self.kmeans
        kmeans_centroids=self.kmeans_centroids

        D=A*np.nan
        ii=kmeans_centroids.argsort()# sort centroids from lowest to highest
        ind=(kmeans.labels_==ii[0])# get the indexes of the pixels with the darkest color
        D[ind]=1# set the corresponding values of D  to 1
        ind=(kmeans.labels_==ii[1])# get the indexes of the pixels with the 2nd darkest  color
        D[ind]=1# set the corresponding values of D  to 1
        D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping

        C,centers2,clust=self.useDBSCAN(D,np.argwhere(D==1),2,5)
        ind=np.argwhere(centers2[:,4]<1000) # remove small clusters
        centers2=np.delete(centers2,ind,axis=0)
        distL=np.sum((centroids[0,0:2]-centers2[:,0:2])**2,axis=1)    
        distR=np.sum((centroids[1,0:2]-centers2[:,0:2])**2,axis=1)    
        iL=distL.argmin()
        iR=distR.argmin() 
        LLungMask=C[:,:,iL].copy()
        RLungMask=C[:,:,iR].copy()
        self.LLungMask=LLungMask
        self.RLungMask=RLungMask
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLungMask,interpolation="nearest")
        ax1.set_title('Left lung mask - improvement')
        ax2.imshow(RLungMask,interpolation="nearest")
        ax2.set_title('Right lung mask - improvement')
        plt.show()
        return
    def final_lung_masks(self):
         
        #%% Final lung masks
        Nr=self.Nr
        Nc=self.Nc
        LLungMask=self.LLungMask
        RLungMask=self.RLungMask
        sct=self.sct
        vm=sct.min()
        vM=sct.max()
        sct=self.sct
        C,centers3,clust=self.useDBSCAN(LLungMask,np.argwhere(np.isnan(LLungMask)),1,5)
        LLungMask=np.ones((Nr,Nc))
        LLungMask[C[:,:,0]==1]=np.nan
        C,centers3,clust=self.useDBSCAN(RLungMask,np.argwhere(np.isnan(RLungMask)),1,5)
        RLungMask=np.ones((Nr,Nc))
        RLungMask[C[:,:,0]==1]=np.nan
        #plt.close('all')
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLungMask,interpolation="nearest")
        ax1.set_title('Left lung mask')
        ax2.imshow(RLungMask,interpolation="nearest")
        ax2.set_title('Right lung mask')
        plt.show()

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLungMask*sct,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
        ax1.set_title('Left lung')
        ax2.imshow(RLungMask*sct,vmin=vm,vmax=vM, cmap='bone',interpolation="nearest")
        ax2.set_title('Right lung')
        plt.show()

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLungMask*sct,interpolation="nearest")
        ax1.set_title('Left lung')
        ax2.imshow(RLungMask*sct,interpolation="nearest")
        ax2.set_title('Right lung')
        plt.show()
        return
    def find_GGO(self):
        LLungMask=self.LLungMask
        RLungMask=self.RLungMask
        sct=self.sct
        LLungMask[np.isnan(LLungMask)]=0
        RLungMask[np.isnan(RLungMask)]=0
        LungsMask=LLungMask+RLungMask
        vm=sct.min()
        vM=sct.max()
        B=LungsMask*sct
        inf_mask=1*(B>-750)&(B<-400)
        InfectionMask=self.filterImage(inf_mask,2)
        InfectionMask=1.0*(InfectionMask>0.25)# threshold to declare opacity
        a=InfectionMask[InfectionMask!=0]
        b=InfectionMask[InfectionMask==0]
        InfectionMask[InfectionMask==0]=np.nan
        plt.close('all') 
        plt.figure()
        plt.imshow(InfectionMask,interpolation="nearest")
        plt.title('infection mask')
        plt.show()
        color_map ='spring'
        plt.figure()
        plt.imshow(sct,alpha=0.8,vmin=vm,vmax=vM, cmap='bone')
        plt.imshow(InfectionMask*255,alpha=1,vmin=0,vmax=255, cmap=color_map,interpolation="nearest")
        plt.title('Original image with ground glass opacities in yellow')
        plt.show()
        return(a,b)
    def infection_meas(self):    
        LLungMask=self.LLungMask
        RLungMask=self.RLungMask
        sct=self.sct
        LLungMask[np.isnan(LLungMask)]=0
        RLungMask[np.isnan(RLungMask)]=0
        L_totall=LLungMask[LLungMask!=0]
        L_totall=L_totall.shape
        L_totall=L_totall[0]#totall pixels of left lung
        R_totall=RLungMask[RLungMask!=0]
        R_totall=R_totall.shape
        R_totall=R_totall[0]#totall pixels of right lung
        BL=LLungMask*sct
        BR=RLungMask*sct
        inf_mask_L=1*(BL>-750)&(BL<-400)
        inf_mask_R=1*(BR>-750)&(BR<-400)
        InfectionMask_L=self.filterImage(inf_mask_L,2)
        InfectionMask_R=self.filterImage(inf_mask_R,2)
        InfectionMask_L=1.0*(InfectionMask_L>0.25)
        InfectionMask_R=1.0*(InfectionMask_R>0.25)# threshold to declare opacity
        L_infected=InfectionMask_L[InfectionMask_L!=0]
        L_infected=L_infected.shape
        L_infected=L_infected[0]
        R_infected=InfectionMask_R[InfectionMask_R!=0]
        R_infected=R_infected.shape
        R_infected=R_infected[0]
        print('left infection')
        print(L_infected)
        print('left totall')
        print(L_totall)
        print('right infection')
        print(R_infected)
        print('right totall')
        print(R_totall)
        L_meas=round((L_infected/L_totall)*100,2)
        R_meas=round((R_infected/R_totall)*100,2)
        return(L_meas,R_meas)
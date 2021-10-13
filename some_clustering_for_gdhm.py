import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from s_dbw import S_Dbw

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering	
import skfuzzy as fuzz

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from numpy import linalg as LA
from sklearn.mixture import GaussianMixture

def getAffinityMatrix(coordinates, k = 7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.
    
    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigenDecomposition(A, topK = 5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = LA.eig(L)
    
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return len(nb_clusters)

def elbow(df,highest):
	
	range_n_clusters = np.arange(2,highest)
	coeff, Sum_of_squared_distances, SDbw = [],[],[]

	scaler = MinMaxScaler()
	df = scaler.fit_transform(df)

	for n_clusters in range_n_clusters:
        
		clusterer = KMeans(n_clusters=n_clusters,tol=0.01, n_init=500)#,random_state=42)
		cluster_labels = clusterer.fit_predict(df)       
        
		silhouette_avg = silhouette_score(df, cluster_labels)  
		score = S_Dbw(df, 
                      cluster_labels,
                      centers_id=None,
                      method='Halkidi',
                      alg_noise='bind',
                      centr='mean', 
                      nearest_centr=True,                       
                      metric='euclidean')
        #print(score)
		coeff.append(silhouette_avg)
		Sum_of_squared_distances.append(clusterer.inertia_)
		SDbw.append(score)  
        
    
	y1 = coeff
	y2 = Sum_of_squared_distances
	y3 = SDbw
    
	fig = go.Figure()
	modes = 'markers+lines'
    
	fig.add_trace(
		go.Scatter(
            x = range_n_clusters,
            y = y1,
            #showlegend = False,
            name = 'Silhouette Score',
            mode = modes,
            marker = dict(color = 'red'),
            yaxis='y1'
        )
    )
    
	fig.add_trace(
		go.Scatter(
            x = range_n_clusters,
            y = y2,
            #showlegend = False,
            name = 'Sum of squared distances',
            mode = modes,
            marker = dict(color = 'green'),
            yaxis='y2'
        )
    )
	fig.add_trace(
		go.Scatter(
            x = range_n_clusters,
            y = y3,
            #showlegend = False,
            name = 'SD validity index',
            mode = modes,
            marker = dict(color = 'blue'),
            yaxis='y3'
        )
    ) 
    
	fig.update_layout(
		yaxis = dict(
            #title="yaxis title",
            titlefont=dict(color="red"),
            tickfont=dict(color="red")
        ),
		yaxis2 = dict(
            #title="yaxis2 title",
            titlefont=dict(color="green"),
            tickfont=dict(color="green"),
            anchor="free",
            overlaying="y",
            side="right",
            position=1
        ),        
		yaxis3 = dict(
            #title="yaxis3 title",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            anchor="free",
            overlaying="y",
            side="right",
            position=0
        ),
    )
	fig.update_layout(
		legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
	return fig.show()

   
def km(dataNorm, id_, n_clusters, **kwds):

	km = KMeans(n_clusters = n_clusters,tol=0.01, n_init=500, random_state=42).fit(dataNorm)
	closest = []
	for i in range(n_clusters):
		c, _ = pairwise_distances_argmin_min(km.cluster_centers_[i].reshape(1, -1), dataNorm[km.labels_ == i])
		closest.append(np.argmax(np.bincount(np.where(dataNorm == dataNorm[km.labels_ == i][c[0]])[0])))
	labels = km.labels_ + 1

	return list(closest), labels


def agg(dataNorm, id_, n_clusters, metric='euclidean', **kwds):
	
	agg = AgglomerativeClustering(n_clusters = n_clusters, affinity=metric, linkage='complete').fit(dataNorm)
	closest = [] 
	labels = agg.labels_ + 1

	return closest, labels


def gmm(dataNorm, id_, n_clusters, **kwds):

    gmm = GaussianMixture(n_components=n_clusters, max_iter=50).fit_predict(dataNorm)
    closest = [] 
    labels = gmm + 1

    return closest, labels


def spec(dataNorm, id_, n_clusters, **kwds):
	
	spec = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42).fit(dataNorm)   
	closest = []
	labels = spec.labels_ + 1

	return closest, labels


def cm(dataNorm, id_, n_clusters, m=1.5, **kwds):




	cntr, u_orig, _, _, _, _, fpc = fuzz.cluster.cmeans(df.T, n_clusters, m=m, error=0.1, maxiter=10000, metric='cosine')
	closest = []
	labels = u_orig.argmax(axis=0)+1
	try:
		for j in range(n_clusters):
			closest.append(pd.DataFrame(df)[u_orig.argmax(axis=0) == j].index[pairwise_distances_argmin_min([cntr[j]], df[u_orig.argmax(axis=0) == j])[0]])
		closest = np.stack(closest).reshape(-1)
		return closest, labels
	except:
		return [], labels

clustering_type = {
	'k-means': km,
	'agglomerative': agg,
	'spectral': spec,
	'c-means': cm,
    'gmm': gmm
}

N = 10
color = list(sns.color_palette(n_colors = N).as_hex())
color_palette = pd.DataFrame([color]).T
color_palette.index += 1
   

def clustering(df, id_, c_type='k-means'):

	"""
	Допустимые методы кластеризации:
	'k-means',
	'agglomerative',
	'spectral',
	'c-means',
	'gmm'
	"""	

	scaler =  MinMaxScaler()
	dataNorm = scaler.fit_transform(df)
	affinity_matrix = getAffinityMatrix(dataNorm, k = 10)
	n_clusters_auto = eigenDecomposition(affinity_matrix)

	print('Спектральный метод предлагает', n_clusters_auto, 'кластеров\n')
	elbow(df,10)

	n_clusters = int(input('Введите количество кластеров\n\n'))

	closest, labels = clustering_type[c_type](dataNorm, id_, n_clusters)

	models = pd.DataFrame(id_,columns=['UWI'])
	output = pd.concat([df,pd.DataFrame(labels,columns=['label']),models],axis=1)
    
	if c_type in ['agglomerative', 'spectral', 'gmm']:
		for cl in range(n_clusters):
        
			df_where_only_i_cluster = output[output.label == cl+1].drop(columns=output.columns[-1])
			centroid = df_where_only_i_cluster.mean()
	        
			centroid = np.array(centroid).reshape(1, -1)
			centroid, _ = pairwise_distances_argmin_min(centroid,df_where_only_i_cluster)
			closest.append(df_where_only_i_cluster.iloc[centroid].index[0])	

	cl = ['Кластер #'+ str(x+1) for x in range(len(closest))]    
    
	fig1 = px.pie(output,
                  names='label',
                  color = 'label',
                  color_discrete_map = color_palette.loc[:n_clusters].to_dict()[0])
                 
	fig1.update_layout(title='% из общей ('+ str(len(df)) +') выборки в каждом кластере')
	fig1.show()
    
	# divided = []     
	# for j,jtem in enumerate(np.unique(labels)):
	# 	divided.append(output[output.label == jtem])  
    
	# for i in range(output.shape[1]-1):        
	# 	feature = output.columns[i]  
        
	# 	fig = make_subplots(rows=1, cols=2) 
        
	# 	for k in range(len(divided)):
	# 		ecdf = ECDF(divided[k].iloc[:,i].values)           
	# 		fig.add_trace(
	# 			go.Scatter(
 #                    x=ecdf.x,
 #                    y=ecdf.y,
 #                    mode='markers',
 #                    legendgroup="group",
 #                    marker = dict(size = 4,color = color_palette.loc[:n_clusters].to_dict()[0][k+1] ),
 #                ),
 #                row = 1, col = 1
 #            )
	# 		fig.add_trace(
 #                go.Scatter(
 #                    x = [divided[k].loc[closest[k]][feature]],
 #                    y = [ecdf.y[list(ecdf.x).index(divided[k].loc[closest[k]][feature])]],
 #                    mode='markers',
 #                    showlegend = False,
 #                    marker_symbol='x',
 #                    marker = dict(size = 15,                                  
 #                                  color = color_palette.loc[:n_clusters].to_dict()[0][k+1] ),
 #                    ),
 #                    row = 1, col = 1
 #                )             
              
	# 		fig.add_trace(
 #                go.Box(
 #                    name = cl[k],
 #                    y = ecdf.x,
 #                    boxpoints='suspectedoutliers',
 #                    legendgroup="group",   
 #                    showlegend=False,
 #                    marker_color = color_palette.loc[:n_clusters].to_dict()[0][k+1],
 #                ),
 #                row = 1, col =2
 #            )
	# 		fig.add_trace(
 #                go.Scatter(
 #                    x = [cl[k]],
 #                    y = [divided[k].loc[closest[k]][feature]],
 #                    mode='markers',
                    
 #                    showlegend = False,
 #                    marker_symbol='x',
 #                    marker = dict(size = 15,                                  
 #                                  color = color_palette.loc[:n_clusters].to_dict()[0][k+1] ),
 #                    ),
 #                    row = 1, col = 2
 #                )             

	# 		fig.update_layout(
 #                xaxis1=dict(title = "Значение параметра"),
 #                xaxis2=dict(title = "Группа данных"),
 #                yaxis1=dict(title="Доля данных"),
 #                yaxis2=dict(title="Значение параметра"))
	# 		fig.update_layout(title = 'Empirical Cumulative Distribution Function and Box-plots for each cluster ' + str(divided[k].columns[i]))
	# 	fig.show()
        
	return output,closest

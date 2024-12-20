import torch
from torch import Tensor
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import networkx as nx 
from scipy.spatial import distance_matrix
import mdtraj as mdt 
import os 
from glob import glob
from einops import rearrange
from typing import *
from tqdm import tqdm
import torch_geometric.utils as tg_utils
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Data , Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

class Batch:
    x:Tensor #trajectory frames. (T*N, d)
    edge_index:Tensor #(2, Ne)
    edge_attr:Tensor
    y:Tensor #frame index in the original simulation data (T,)
    repl:Tensor #replicate index in the original simulation data (T,)
    batch:Tensor # batch index of node (N,)

def read_xtc_trajectory(folder, replicates=None) -> tuple[list[mdt.Trajectory], list[str]]:
    topo = glob(os.path.join(folder,'*.pdb'))
    topo +=  glob(os.path.join(folder,'*.gro'))
    if len(topo)>1:
        raise ValueError(f'Ambiguous pdb/gro files at ', folder)
    elif len(topo) == 1 :
        topo = topo[0]
    else :
        raise ValueError('No valid topology file at '+ folder)
    prot = mdt.load(topo)
    prot = prot.atom_slice(prot.topology.select('protein'))
    #prot = prot.atom_slice(prot.topology.select('protein and name Y00'))#filter out water and membrane molecules if needed
    files =  glob(os.path.join(folder,"*.dcd"))#[:replicates]
    print('FUCKIT',files)
    #files =  glob(os.path.join(folder,"*.xtc"))[:replicates]
    trajs = [mdt.load(xtc,top=prot) for xtc in files]
    trajs = [i.atom_slice(i.topology.select('protein')) for i in trajs]
    return trajs, files

def read_pdb_trajectory(folder, replicates=None)-> tuple[list[mdt.Trajectory],  list[str]]:
    files = glob(os.path.join(folder, '*.pdb'))[:replicates]
    trajs = [mdt.load_pdb(f) for f in files]
    return trajs, files

def apply_to_all_simulation(MDFolder,replicates,
                            timesteps,first_frame=None, last_frame=None, use_displacement=True, 
                            use_dihedral=True, use_cbs = True,stride=10):
    '''
    returns:
        list of data as numpy arrays
        indices of frames
        lengths of replicates
    '''
    out = []
    if glob(os.path.join(MDFolder, '*.dcd'),):
        trajs, src_files = read_xtc_trajectory(MDFolder, replicates)
    else :
        trajs, src_files = read_pdb_trajectory(MDFolder, replicates)
    frame_i = []
    replicate_n = []
    traj0=trajs[0]
    for i_traj, traj in enumerate(trajs):
        if first_frame is None: # deal with the no-cutoff case
            cutoff_inf = 0
        else :
            cutoff_inf = first_frame
        if last_frame is None: # deal with the no-cutoff case
            cutoff_sup = traj.time.shape[0]
        else :
            cutoff_sup = last_frame
        traj = mdt.Trajectory.superpose(traj,traj0,frame=0)
        traj = traj.center_coordinates(traj)
        print('max coordinates',np.max(traj.xyz[:,:,:]))
        residues = [atom.index for atom in traj.topology.atoms if atom.name=='CA'] #map residue -> CA
        res_inv = np.ones((residues[-1]+1,), dtype=int)*-1 #inverse map atom -> residue. Beware, non-CA indices map to -1
        n_res = len(residues)
        res_inv[residues] = np.arange(n_res)
        #xyz = traj.xyz[cutoff_inf:cutoff_sup:timesteps,residues,:] #(n_frames, N, 3)
        xyz = traj.xyz[cutoff_inf:cutoff_sup:timesteps,residues,:] #ATTENTION
        ###EDIT 16082023: Align every frame to

        # xyz = traj.xyz[650:850,residues,:]
        # xyz = xyz[0::timesteps,:,:]
        frame_i.append([*range(cutoff_inf,cutoff_sup, timesteps)])
        replicate_n.append((cutoff_sup - cutoff_inf) // timesteps)
        displacements = np.empty((xyz.shape[0], n_res, 0)) #dummy arrays for later stacking
        angles = np.empty((xyz.shape[0], n_res, 0))
        cbs = np.empty((xyz.shape[0], n_res, 0))
        if use_displacement:
            displacements = []
            displacements.append(np.zeros((xyz.shape[1],3))) #Padding for first frame
            for i in range(1,xyz.shape[0]):
                displacements.append(xyz[i,:,:]-xyz[i-1,:,:])
            displacements = np.stack(displacements,axis=0)
        if use_dihedral:
            print('Using diherdral angles')
            #Create empty array
            angles = np.zeros((xyz.shape[0],n_res,4))
            #Compute angles 
            phi_i, phi = mdt.compute_phi(traj)
            psi_i, psi = mdt.compute_psi(traj)
            #Compute sine and cosine
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)

            #Create full array
            phi_nonpad = np.stack([cos_phi,sin_phi],axis=-1) #(T, N-nchains, 2)
            psi_nonpad = np.stack([cos_psi,sin_psi],axis=-1)
            angles[:,res_inv[phi_i[:,2]],:2] = phi_nonpad[cutoff_inf:cutoff_sup:timesteps,:,:]
            angles[:,res_inv[psi_i[:,1]],2:] = psi_nonpad[cutoff_inf:cutoff_sup:timesteps,:,:]
        if use_cbs:
            print("Using CBs")
            atoms = [atom.index for atom in traj.topology.atoms if atom.name in ['CB', 'HA3']] #HA3 for glycines H
            cbs = traj.xyz[cutoff_inf:cutoff_sup:timesteps,atoms,:] 
        #sample = create_timeseries(xyz,timesteps)
        print(xyz.shape,displacements.shape,angles.shape,cbs.shape)
        xyz = np.concatenate([xyz, displacements, angles, cbs ],axis=-1)
        if use_displacement: #drop first frame with undefined displacement
            xyz = xyz[1:,]
            frame_i[-1] = frame_i[-1][1:]
            replicate_n[-1]-=1
            
        print(f'{src_files[i_traj]} : shape without rot embeddings', xyz.shape)
        out.append(xyz)

    # sort according to length
    # so that test sets, if any, are the shortest
    idx = np.argsort(replicate_n)[::-1]
    out = [out[i] for i in idx]
    frame_i = [frame_i[i] for i in idx]
    src_files = [src_files[i] for i in idx] 
    replicate_n = [replicate_n[i] for i in idx] 
    frame_i = [i for a in frame_i for i in a ] #flatten

    data = np.concatenate(out,axis=0) #Stack over all simulation replicates
    return data, frame_i, replicate_n, src_files

def create_timeseries(xyz,timesteps):
    """
    Input: xyz: [shape B,N,3]
    Output: [batch,timesteps,N,3]  
    """
    out = []

    for j in range(xyz.shape[0]-timesteps-1): 
        sample = xyz[j:j+timesteps,:,:]
        out.append(sample)
    return np.stack(sample,axis=0)

def get_graph(data,threshold=1.5,verbose=True):
    '''
    Make a graph out of frames of x,y,z coords
    threshold in nm
    Return :
        out_coords : list of coordinate tensors
        out_edges : list of edge list tensors
        out_weight : list of weight of edges tensors
    '''
    n_nodes,timesteps,dim = data.shape

    data = rearrange(data,'... n d -> (...) n d')

    out_coords = []
    out_edges = []
    out_weight = []
    for coords in tqdm(data,disable=not verbose, desc='Extracting edges from coordinate matrix'):
        contact = distance_matrix(coords[:,:3],coords[:,:3]) #Select only coordinates and not displacement vectors 
        index = np.where(contact>threshold) #Matthieu 26/04 -> test long range
        contact[index]=0

        graph = nx.from_numpy_array(contact, create_using=nx.DiGraph)  #DiGraph so that edges are listed in both directions
        edges = np.array(graph.edges()).T

        #edges = np.vstack([edges[:,0],edges[:,1]])
        edge_attr = []
        for key,val in nx.get_edge_attributes(graph,"weight").items():
            edge_attr.append(val)

        edge_attr = np.vstack(edge_attr)
        out_weight.append(torch.FloatTensor(edge_attr))
        out_edges.append(torch.LongTensor(edges))
        out_coords.append(torch.FloatTensor(coords))

    return out_coords,out_edges,out_weight

class Normalizer(): 
    def __init__(self, data : List[Data],  blank=False) -> None:
        '''
        Keeps mean and std of data to restore it
        if blank, self.restore is a no-op
        '''
        xyz = []
        for d in data : 
            x = d.x
            if isinstance(d, Batch):
                x = unbatch(x, d.batch) # [(N, d)]
                x = torch.stack(x) # B,N,d
            else :
                x = x.unsqueeze(0) # 1, N , D
            xyz.append(x)
        xyz = torch.cat(xyz, dim=0) #B*Nb, N, d
        #time wise normalization (m_{N,k})
        # self.xyz_m = xyz.mean(dim=0) #N, d
        # self.xyz_s = xyz.std(dim=0)
# residue and time wise normalization (m_k)
        self.xyz_m = xyz.mean(dim=(0,1)) # d
        self.xyz_s = xyz.std(dim=(0,1))
 
        w = torch.cat([d.edge_attr for d in data], dim=0)
        self.w_m = w.mean(dim=0)
        self.w_s = w.std(dim=0)
        self.blank = blank

    def normalize(self, batch: Union[Batch, Data]):
        '''Normalize  data in-place'''
        x = batch.x
        batch.x = (x-self.xyz_m)/self.xyz_s #
        batch.edge_attr = (batch.edge_attr-self.w_m)/self.w_s

    def restore(self, batch: Union[Batch, Data]=None, x : Tensor = None, clip = False) -> Tensor:
        '''Return rescaled batch to original scale 
        If batch is not None, unbatch it and ignore x
        If x is passed, it should have the shape (T,N,d)
        If self.blank, do nothing except unbatch and stack x 
        If clip, leaves out supplementary dimensions of m and s
        '''
        match (batch, x):
            case None, None:
                raise ValueError("Pass either batch or x")
            case _, None:
                x = torch.stack(unbatch(batch.x, batch.batch), dim=0)
            case None, _:
                pass
            case _,_:
                raise ValueError("Pass either batch or x")
        if self.blank:
            return x
        if clip :
            k = x.size(-1)
        else :
            k = None
        x = x*self.xyz_s[...,:k]+self.xyz_m[...,:k]
        # edge_attr = batch.edge_attr*self.w_s+self.w_m
        return x

def get_persistence_edge(batch : Data|Batch, threshold = 0.5):
    x = torch.stack(tg_utils.unbatch(batch.x, batch.batch))
    x = x[:,:,:3] # drop positional encodings and displacements
    edges = tg_utils.unbatch_edge_index(batch.edge_index, batch.batch)
    B,N, _ = x.shape
    A = torch.zeros((B,N,N), device=edges[0].device, dtype = torch.float16)
    for b in range(B):
        A[b][edges[b][0], edges[b][1]] = 1 # update adj matrix
    A = (A.mean(dim=0) > threshold).int() # take only persistent edges
    return  torch.argwhere(A).T
    
def make_dataset(
    coords:List[torch.Tensor],
    edges:List[torch.Tensor],
    weights:List[torch.Tensor],
    frame_idx,
    repl_idx
)-> List[Data]:
    '''
    Make a list of data to give to a dataloader
    input : lists of tensors of shapes (n_edge, 1), (2, n_edge), (N, 3),
    as returned by get_graph
    output : List[Data]
    '''
    data = [Data(x=c,edge_index=e,edge_attr=w,y=b, repl=r ) for c,e,w,b,r in zip(coords,edges,weights,frame_idx, repl_idx)]
    return data


class OverlapLoader(DataLoader): 
    '''DataLoader that loads data with overlaps between batches'''
    def __init__(
            self, dataset: List[Data], repl_len, batch_size: int = 1, offset:int=1, shuffle: bool = True, stride:int=1,
            **kwargs):
        '''
        Refer to DataLoader class for documentation
        offset : number of steps between the start of a batch and the start of the following
        If offset is None, do not overlap
        '''
        assert offset is None or offset > 0, "Offset must be positive"
        print(f"offset {offset}, nb of examples {len(dataset)}, batch_size {batch_size}, stride {stride}")
        sampler = TimeBatchSampler(repl_len, batch_size=batch_size, offset=offset, shuffle=shuffle, stride=stride)
        super().__init__(dataset, follow_batch = ['edge_attr'], batch_sampler=sampler, **kwargs)
        self.avg_edges:int = np.mean([d.edge_index.shape[1] for d in dataset])


class LazyPersistenceLoader(OverlapLoader):
    '''OverlapLoader that computes persistent edges at iter time'''
    def __init__(self, dataset: List[Data], repl_len, batch_size: int = 1, offset: int = 1, shuffle: bool = False, 
                 threshold= 0.5, n_replicates=1, **kwargs):
        super().__init__(dataset, repl_len, batch_size, offset, shuffle, **kwargs)
        self.threshold = threshold
        self.n_replicates = n_replicates #useful for later exploitation

    def __iter__(self) :
        for batch in super().__iter__():
            batch.persistence_edges = get_persistence_edge(batch, self.threshold) #TODO compute once at data loading time. 
            yield batch

class PersistenceLoader(OverlapLoader):
    '''OverlapLoader that computes persistent edges only once'''
    def __init__(self, dataset: List[Data], repl_len, batch_size: int = 1, offset: int = 1, shuffle: bool = False, 
                 threshold= 0.5, n_replicates=1, stride=1, **kwargs):
        super().__init__(dataset, repl_len, batch_size, offset, shuffle, stride=stride, **kwargs)
        self.threshold = threshold
        self.n_replicates = n_replicates #useful for later exploitation
        self.persistence = {}

    def __iter__(self) :
        for batch in super().__iter__():
            key = (batch.y[0].item(), batch.repl[0].item())
            edges =  self.persistence.get(key)
            if edges is None:
                edges = get_persistence_edge(batch, self.threshold)
                self.persistence[key] = edges
            batch.persistence_edges = edges 
            yield batch

class TimeBatchSampler(Sampler):
    def __init__(self,repl_len:List[int], batch_size: int, offset:Optional[int], stride:int =1, shuffle : bool = True) -> None:
        '''
        repl_len : list of lengths of replicates
        offset : offset between batches. If none, is equal to batch len
        stride : number of steps between frames in a batch
        '''
        self.offset = offset if offset is not None else batch_size
        self.sizes = repl_len #sizes of underlying replicates
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = stride
        
    def __iter__(self) -> Iterator[List[int]]:
        d = 0
        index = []
        for N in self.sizes:
            index.extend(range(0+d,N-self.batch_size*self.stride+d, self.offset))
            d+=N
        if self.shuffle:
            np.random.shuffle(index)
        for i in index:
            batch = []
            for j in range(i, i+self.batch_size*self.stride, self.stride):
                batch.append(j)
            yield batch

    def __len__(self):
        return sum([(N-self.batch_size*self.stride)//self.offset for N in self.sizes])


# def overlap_iterator(dataset :List[Data],batch_size:int, offset:int):
#     assert (len(dataset)-batch_size)%offset==0, "To ensure all batches are the same size, batch size and offset must be congruent with length of dataset"
#     i = 0
#     j = 0
#     while i+batch_size <= len(dataset):
#         for j in range(i, i+batch_size):
#             yield dataset[j]
#         i+=offset


def split_batch(src : Tensor, batch : Tensor,i :int, dim : int = 0):
    '''
    Split src in two according to batch vector at graph i along dimension dim.
    Return [s1, s2] where s1 contains the data for the i first graphs, and s2 the rest.
    '''
    sizes = tg_utils.degree(batch, dtype=torch.long)
    s1 = sizes[:i].sum()
    sizes = [s1.cpu().item(), (sizes.sum() - s1).cpu().item()]
    return src.split(sizes, dim)

def split_batch_edge_index(edge_index : Tensor, batch : Tensor,i :int, dim : int = 0):
    '''
    Split edge_index in two according to batch vector at graph i along dimension dim.
    Return [s1, s2] where s1 contains the edge indices for the i first graphs, and s2 the rest.
    '''
    deg = tg_utils.degree(batch, dtype=torch.int64)
    ptr = torch.zeros_like(deg)
    ptr[i:] = deg[:i].sum() #sum of numbers of nodes in previous batch

    edge_batch = batch[edge_index[0]] # two nodes of one edge are in the same batch
    edge_index = edge_index - ptr[edge_batch] # subtract number of nodes in previous graphs
    sizes = tg_utils.degree(edge_batch, dtype=torch.int64)
    s1 = sizes[:i].sum()
    sizes = [s1.cpu().item(), (sizes.sum() - s1).cpu().item()]
    return edge_index.split(sizes, dim=1)



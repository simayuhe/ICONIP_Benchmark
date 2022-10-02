from mpi4py import MPI
from RANDOMACT.env_builder import build_env,build_single_env
import multiprocessing 
from Common.misc_util import set_global_seeds
from Common.vec_env import SubprocVecEnv
def build_vec_env(args):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    print("mpi_ranknvidia",mpi_rank)
    seed = args.seed + 10000 * mpi_rank if args.seed is not None else None
    #print("seed",seed)
#     seed = seed + 10000 * mpi_rank if seed is not None else None
    start_index = 0 # 为啥加这个？
    initializer = None # ??
    def make_thunk(rank,initializer=None):
        return lambda: build_single_env(args,rank,seed)
    set_global_seeds(seed)
    return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(args.num_env)])

   
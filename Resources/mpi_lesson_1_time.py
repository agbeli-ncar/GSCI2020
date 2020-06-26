from mpi4py import MPI
import numpy as np
import time


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
init_time = time.time()

N = 50000000
array = np.ndarray.tolist(np.arange(N+1))

start_index = int(N * rank / size + 1)
end_index   = int(N * (rank +1)  / size) + 1

partial_sum = sum(array[start_index:end_index])

if rank!= 0:
	comm.send(partial_sum,dest=0,tag=rank)
	comm.send((time.time() - init_time), dest = 0, tag = size+rank)

else:
	data_received = np.ndarray.tolist(np.arange(1,size+1))
	time_received = np.ndarray.tolist(np.arange(1,size+1))

	for i in list(range(1,size)):
		data_received[i] = comm.recv(source = i, tag = i)
		time_received[i] = comm.recv(source = i, tag = size+i)
	data_received[0] = partial_sum
	time_received[0] = time.time()-init_time

	print('Your array is ', data_received) 
	print('The sum of your array is ', sum(data_received))
	print('The total time to calculate the sum was', (time.time()-init_time))

	for i in list(range(0,size)):
		print('Node', i, 'took ', time_received[i], ' seconds') 

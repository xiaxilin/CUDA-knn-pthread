/* 
* INPUT:
* m: total num of points
* n: n dimensions
* k: num of nearest points
* V: point coordinates
* OUTPUT:
* out: k nearest neighbors
*/

#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>

//#define DEBUG
#ifdef DEBUG
#define TA_PRINT printf
#else
#define TA_PRINT(...) ;
#endif

#define INIT_MAX 10000000
#define TILE_WIDTH 32
#define TILE_DEPTH 128
#define MAX_BLOCK_SIZE 256
#define MAX_PTRNUM_IN_SMEM 1024 

// compute the square of distance of the ith point and jth point
__global__ void computeDist(int id, int m, int n, int *V, int *D)
{
	__shared__ int rowVector[TILE_WIDTH][TILE_DEPTH];
	__shared__ int colVector[TILE_DEPTH][TILE_WIDTH];
	__shared__ int dist[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
   	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row;
	int col;
	int px;
	int py;	

	for(py=ty; py<TILE_WIDTH; py+=blockDim.y)
	{
		for(px=tx; px<TILE_WIDTH; px+=blockDim.x)
		{
			row = by*TILE_WIDTH+py;
			col = bx*TILE_WIDTH+px;
			dist[py][px] = 0;
			__syncthreads();
		
			if(row >= id*(m/2) && row < (id+1)*(m/2))
			{

				for(int i=0; i<(int)(ceil((float)n/TILE_DEPTH)); i++)
				{
					for(int j=tx; j<TILE_DEPTH; j+=blockDim.x)
					{
						rowVector[py][j] = V[row*n+i*TILE_DEPTH+j];
					}
					for(int j=ty; j<TILE_DEPTH; j+=blockDim.y)
					{		
						colVector[j][px] = V[col*n+i*TILE_DEPTH+j];
					}
					__syncthreads();
			
					for(int j=0; j<TILE_DEPTH; j++)
					{
						dist[py][px] += (rowVector[py][j]-colVector[j][px])*(rowVector[py][j]-colVector[j][px]);
					}
					__syncthreads();
				}
				
				if(row >= (m/2))
				{
					row -= (m/2);
				}

				D[row*m+col] = dist[py][px];
			}
		}
	}
}

extern __shared__ int SMem[];

//find the min value and index in the count^th loop
__device__ int findMin(int id, int m, int k, int count, int *D, int *out)
{
	int i = blockIdx.x;
  	int tid = threadIdx.x;

	int s = blockDim.x/2;
	int resultValue = INIT_MAX;
	int resultIndex = INIT_MAX;
	int indexBase = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	
	for(int num=0; num<m; num+=MAX_PTRNUM_IN_SMEM)
	{
		for(int j=tid; j<indexBase; j+=blockDim.x)
		{
			if(j+num == i+(m/2)*id )
			{
				SMem[j] = INIT_MAX;
			}
			else
			{
//				SMem[j] = D[(i + (m/2)*id) *m+num+j];
				SMem[j] = D[i*m+num+j];
			}
			//index
			SMem[indexBase+j] = j+num;
			__syncthreads();
		}
/*
		if(tid < count)
		{
			if(out[i*k+tid]-num>=0 && out[i*k+tid]-num<indexBase)
			{
				SMem[ out[i*k+tid]-num ] = INIT_MAX;
			}
			__syncthreads();
		}
		__syncthreads();
*/
		for(int j=0; j<count; j++)
		{
			if(out[i*k+j]-num>=0 && out[i*k+j]-num<indexBase)
			{
				SMem[ out[i*k+j]-num ] = INIT_MAX;
			}
			__syncthreads();
		}
		__syncthreads();
//		for(s=indexBase/2; s>0; s>>=1) 
		for(s=indexBase/2; s>32; s>>=1) 
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(j < s) 
				{
					if(SMem[j] == SMem[j+s])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+s])
						{
							SMem[indexBase+j] = SMem[indexBase+j+s];
						}
					}
					else if(SMem[j] > SMem[j+s])
					{
						SMem[j] = SMem[j+s];
						SMem[indexBase+j] = SMem[indexBase+j+s];
					}
				}
				__syncthreads();
			}
		}
		if(tid < 32)
		{
			#pragma unroll 5
			for(s=32; s>0; s>>=1)
			{ 
				if(SMem[tid] == SMem[tid+s])
				{
					if(SMem[indexBase+tid] > SMem[indexBase+tid+s])
					{
						SMem[indexBase+tid] = SMem[indexBase+tid+s];
					}
				}
				else if(SMem[tid] > SMem[tid+s])
				{
					SMem[tid] = SMem[tid+s];
					SMem[indexBase+tid] = SMem[indexBase+tid+s];
				}
			}
		}
	
		__syncthreads();
		if(resultValue == SMem[0])
		{
			if(resultIndex > SMem[indexBase])
			{
				resultIndex = SMem[indexBase];
			}
		} 
		else if (resultValue > SMem[0])
		{
			resultValue = SMem[0];
			resultIndex = SMem[indexBase];
		}
		__syncthreads();
	}
	return resultIndex;

}

// compute the k nearest neighbors
__global__ void knn(int id, int m, int k, int *V, int *D, int *out)
{
	int i;
	int count;

	i = blockIdx.x;
	__syncthreads();
	for(count=0; count<k; count++)
	{
		out[i*k+count] = findMin(id, m, k, count, D, out);
		__syncthreads();
	}
}
//extern "C"

void showResult(int m, int k, int *out)
{
	int i,j;
	for(i=0; i<m; i++)
	{
		for(j=0; j<k; j++)
		{
			printf("%d ", out[i*k+j]);
			if(j == k-1)
			{
				printf("\n");
			}	
		}    	
	}        	
}            	


pthread_barrier_t barr;

struct HYBctx{
	int id;
	int m;
	int n;
	int k;
	int *V;
	int *d_V;
	int *out;
	int *d_out;
	int *D;
};

void launch(struct HYBctx* ctx){

	int m = ctx->m;
	int n = ctx->n;
	int k = ctx->k;

	// copy host values to devices copies
	cudaMemcpy(ctx->d_V, ctx->V, m*n*sizeof(int), cudaMemcpyHostToDevice);

	int gridDimX = (int)(ceil((float)m/TILE_WIDTH));
	int gridDimY = (int)(ceil((float)m/TILE_WIDTH));

	dim3 grid(gridDimX, gridDimY);
	dim3 block(TILE_WIDTH, TILE_WIDTH);

	// launch knn() kernel on GPU
	computeDist<<<grid, block>>>(ctx->id, m, n, ctx->d_V, ctx->D);
	cudaDeviceSynchronize();

	int threadNum = (m<MAX_BLOCK_SIZE)? m: MAX_BLOCK_SIZE;
	int ptrNumInSMEM = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	knn<<<(m/2), threadNum, 2*ptrNumInSMEM*sizeof(int)>>>(ctx->id, m, k, ctx->d_V, ctx->D, ctx->d_out);

	// copy result back to host
	cudaMemcpy(ctx->out+(m/2)*k*(ctx->id), ctx->d_out, (m/2)*k*sizeof(int), cudaMemcpyDeviceToHost);

}

int cudaInit(int rank, struct HYBctx* ctx){
	if(cudaSetDevice(rank) == cudaSuccess){
		int m = ctx->m;
		int n = ctx->n;
		int k = ctx->k;
		// allocate space for devices copies
		cudaMalloc((void **)&ctx->d_V, m*n*sizeof(int));
		cudaMalloc((void **)&ctx->D, (m/2)*m*sizeof(int));
		cudaMalloc((void **)&ctx->d_out, (m/2)*k*sizeof(int));

		return 0;
	}
	return 1;
}

void cudaDown(struct HYBctx* ctx){
	// cleanup
	cudaFree(ctx->d_V);
	cudaFree(ctx->d_out);
	cudaFree(ctx->D);
}

void beforeStart(struct HYBctx* ctx){
	float *dA;
	int i = 0;
	while(1){
		cudaSetDevice(ctx->id);
		if(cudaMalloc((void**)&dA, 1024*sizeof(float))){
			continue;
		}
		break;
	}
	cudaFree(dA);
	TA_PRINT("you get device %d\n",ctx->id);
}

double comtime;

void* GPUthread(void* arg){
	struct HYBctx* ctx = (struct HYBctx*)arg;
	struct timespec start, end;
	int i;
	if(ctx->id == 0) beforeStart(ctx);
	pthread_barrier_wait(&barr);
	if(ctx->id == 1) beforeStart(ctx);
	if(!cudaInit(ctx->id, ctx)) TA_PRINT("GPU thread %d\n", ctx->id);
	pthread_barrier_wait(&barr);
	clock_gettime(CLOCK_REALTIME,&start);
	pthread_barrier_wait(&barr);

	launch(ctx);
	
	pthread_barrier_wait(&barr);
	clock_gettime(CLOCK_REALTIME,&end);
	if(ctx->id == 0)
		comtime = (double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/(double)1000000000L;
//	TA_PRINT("GPU thread %d result: %d\n",ctx->id, A[0]);
	cudaDown(ctx);
	return NULL;
}


int main(int argc, char* argv[]){
	int i;
	int GPU_num = 2;

	int m,n,k;
	int *V, *out;				// host copies
	int *D;						
	FILE *fp;
	if(argc != 2)
	{
		printf("Usage: knn <inputfile>\n");
		exit(1);
	}
	if((fp = fopen(argv[1], "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
	fscanf(fp, "%d %d %d", &m, &n, &k);

	V = (int *) malloc(m*n*sizeof(int));
	out = (int *) malloc(m*k*sizeof(int));

	for(i=0; i<m*n; i++)
	{
		fscanf(fp, "%d", &V[i]);
	}


	struct HYBctx *ctx = (struct HYBctx*)malloc(sizeof(struct HYBctx)*(GPU_num));
	beforeStart(ctx);
	void *pt;
	pt = malloc(sizeof(pthread_t)*(GPU_num));
	pthread_barrier_init(&barr, NULL, 2);
/*	
	// compute the execution time
	cudaEvent_t start, stop;
	// create event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// record event
	cudaEventRecord(start);
*/
	for(i = 0; i< GPU_num; i++){
		ctx[i].id = i;
		ctx[i].m = m;
		ctx[i].n = n;
		ctx[i].k = k;
		ctx[i].V = V;
		ctx[i].out = out;
		pthread_create(&((pthread_t*)pt)[i], NULL, GPUthread, (void*)&ctx[i]);
	}
	for(i = 0; i< GPU_num; i++)
		pthread_join(((pthread_t*)pt)[i], NULL);
        TA_PRINT("\t%f\n",comtime);
	pthread_barrier_destroy(&barr);

/*
	// record event and synchronize
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	// get event elapsed time
	cudaEventElapsedTime(&time, start, stop);
*/
	cudaDeviceReset();

	showResult(m, k, out);
	printf("%f\n", comtime);

	free(V);
	free(out);
	fclose(fp);

	return 0;
}

#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#define N 32
__global__ void kernel(int *dA){
	int id = threadIdx.x;
	int i;
	for(i = 1; i< N; i *= 2){
		if(id+i < N) dA[id] += dA[id+i];
	} __syncthreads();
}

struct HYBctx{
	int id;
	int *dA;
	pthread_barrier_t barr;
};
void launch(struct HYBctx* ctx, int *A){
	cudaMemcpy(ctx->dA, A, sizeof(int)*N, cudaMemcpyHostToDevice);
	kernel<<<1, N>>>(ctx->dA);
	cudaMemcpy(A, ctx->dA, sizeof(int)*N, cudaMemcpyDeviceToHost);
}

int cudaInit(int rank, struct HYBctx* ctx){
	if(cudaSetDevice(rank) == cudaSuccess){
		cudaMalloc((void**)&ctx->dA, sizeof(int)*N);
		return 0;
	}
	return 1;
}
void cudaDown(struct HYBctx* ctx){
	cudaFree(ctx->dA);
}

void* GPUthread(void* arg){
	struct HYBctx* ctx = (struct HYBctx*)arg;
	int A[N];
	int i;
	for(i = 0; i< N; i++) A[i] = ctx->id;
	if(!cudaInit(ctx->id, ctx)) printf("GPU thread %d\n", ctx->id);
	launch(ctx, A);
	printf("GPU thread %d result: %d\n",ctx->id, A[0]);
	cudaDown(ctx);
	return NULL;
}

void beforeStart(struct HYBctx* ctx){
	float *dA;
	int i = 0;
	while(1){
		i = (i+1)%2;
		cudaSetDevice(2*i);
		if(cudaMalloc((void**)&dA, 1024*sizeof(float))){
			continue;
		}
		else{
			cudaGetDevice(&ctx[0].id);
			if(ctx[0].id%2) continue;
		}
		break;
	}
	cudaGetDevice(&ctx[0].id);
	cudaFree(dA);
	ctx[1].id = ctx[0].id+1;
	printf("you get device %d and %d\n",ctx[0].id, ctx[1].id);
}

int main(int argc, char* argv[]){
	int i;
	int GPU_num = 2;
	struct HYBctx *ctx = (struct HYBctx*)malloc(sizeof(struct HYBctx)*(GPU_num));
	beforeStart(ctx);
	void *pt;
	pt = malloc(sizeof(pthread_t)*(GPU_num));
	for(i = 0; i< GPU_num; i++){
		//ctx[i].id = i;
		pthread_create(&((pthread_t*)pt)[i], NULL, GPUthread, (void*)&ctx[i]);
	}
	for(i = 0; i< GPU_num; i++)
		pthread_join(((pthread_t*)pt)[i], NULL);
	cudaDeviceReset();
	return 0;
}

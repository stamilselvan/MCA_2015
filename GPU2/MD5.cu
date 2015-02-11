#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <lock.h>
#include <cuda_runtime.h>


const unsigned int s_table[] = {
7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22 ,
5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20 ,
4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23 ,
6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 };

const unsigned int k_table[] = {
0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee ,
0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501 ,
0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be ,
0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821 ,
0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa ,
0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8 ,
0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed ,
0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a ,
0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c ,
0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70 ,
0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05 ,
0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665 ,
0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039 ,
0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1 ,
0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1 ,
0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };

const int digests_3letters[] = {
0x98500190 , 0xb04fd23c , 0x7d3f96d6 , 0x727fe128 ,			// abc
0x6fb36fd1 , 0x78f81109 , 0x61138c99 , 0x5e70af91 , 			// xyz
0x54cf8e12 , 0x52ac352a , 0xc77da870 , 0x04849140 ,			
0xc7e5bc47 , 0x489f584f , 0x7ed5db67 , 0x08f8a99c , 			
0x6bb8abf3 , 0xd5f44cd3 , 0x4cf19826 , 0x60dca10d , 
0x93593c6d , 0x0f7d01ca , 0x25b469f1 , 0x023f19d3 , 
0xb5253fba , 0x708f5ac8 , 0xaf238429 , 0x5d209bce , 
0x2d7e74d9 , 0xb9bd42a3 , 0x9538f695 , 0x3d1aad33 , 
0x2abea00a , 0xd9116486 , 0x525103ff , 0x47494527 , 
0xba2c5f2c , 0x29f9f79a , 0x1eded386 , 0xaa3f3f75 
};

const int digests_4letters[] = {
0x1cc91d1a , 0xc6257390 , 0xf0dd7192 , 0x72bc44c9 ,				
0xd0122096 , 0xd970819b , 0x9f66f012 , 0x079d7d6d ,				
0xcd6b8f09 , 0x73d32146 , 0x834edeca , 0xf6b42726 ,				
0x5921d648 , 0x62f5df03 , 0x882ee538 , 0x8f0c3891 ,				
0xb9285f01 , 0x36dd1bdf , 0x76d97d42 , 0x9db273fb , 				
0xa5f9f4b1 , 0xd96fe323 , 0x3e57f469 , 0x4045af25 ,				
0x352cd3fb , 0x468d4cbb , 0xa202fcf8 , 0xeff1ab74 ,				
0x8572cb61 , 0xb923e58b , 0x3dcc6e92 , 0xc6d0a57d ,				
0x5dd92fa8 , 0x5df20fb1 , 0x079fd3fa , 0x37be2e37 ,				
0xa51a803d , 0xc3cec132 , 0x7ad882ee , 0x3ff6fd99}; 	

const int digests_6letters[] = {
0x753213d1 , 0xbe1821ee , 0xaf77a563 , 0x52c09f75 ,
0xdf8e57d8 , 0x06ce5884 , 0x76bbc5fb , 0xa45c8ca5 , 
0x097d100d , 0x0ce4bbf5 , 0x5cdee3ad , 0xb7e9e971 ,
0x9ad1e705 , 0x1821006d , 0xd270efde , 0x6e22f41f 
};


#define THREE_LETTER

#ifdef THREE_LETTER
	#define MAX_DG (10)
	#define no_of_char 3
	#define NO_BLOCKS_X 8	
	#define NO_BLOCKS_Y 9	
	#define NO_THREADS_X 16  
	#define NO_THREADS_Y 16
	#define THREAD_LIMIT 17576 // 26*26*26
#endif

#ifdef FOUR_LETTER
	#define MAX_DG (10)
	#define no_of_char 4
	#define NO_BLOCKS_X 32
	#define NO_BLOCKS_Y 14
	#define NO_THREADS_X 32
	#define NO_THREADS_Y 32
	#define THREAD_LIMIT 456976 // 26*26*26*26
#endif

#ifdef SIX_LETTER
	#define MAX_DG (4)
	#define no_of_char 6
	#define NO_BLOCKS_X 512
	#define NO_BLOCKS_Y 920
	#define NO_THREADS_X 32
	#define NO_THREADS_Y 32
	#define THREAD_LIMIT 481890304
#endif

// Simplified for max. 8 letters "__forceinline__"
__forceinline__  __device__ void md5(char* message,int length, unsigned int* digest, int *dev_s_table, int *dev_k_table)
{
   unsigned int a0 = 0x67452301;
	unsigned int b0 = 0xefcdab89; 
   unsigned int c0 = 0x98badcfe; 
   unsigned int d0 = 0x10325476; 
	unsigned int A=a0;
	unsigned int B=b0;
	unsigned int C=c0;
	unsigned int D=d0;
	unsigned int M[16]  = {0,0,0,0, 0,0,0,0, 0,0,0,0 , 0,0,0,0};
	//memcpy(M,message,length);

	#ifdef FOUR_LETTER
		  M[0] = M[0] | ((int) message[3] << 24)
					| ((int) message[2] << 16)
					| ((int) message[1] << 8)
					| ((int) message[0]);
	#endif

	#ifdef THREE_LETTER
		  M[0] = M[0] | ((int) message[2] << 16)
					| ((int) message[1] << 8)
					| ((int) message[0]);
	#endif

	#ifdef SIX_LETTER
			M[0] = M[0] | ((int) message[3] << 24)
					| ((int) message[2] << 16)
					| ((int) message[1] << 8)
					| ((int) message[0]);

			M[1] = M[1] | ((int) message[5] << 8)
						| ((int) message[4]);
	#endif
	
	((char*)M)[length]=0x80;
	M[14]=length*8;

	#pragma unroll
	for (int i=0;i<64;i++) 
	{
		unsigned int F = (B & C) | ((~B) & D);
		unsigned int G = (D & B) | ((~D) & C);
		unsigned int H = B ^ C ^ D;
		unsigned int I = C ^ (B | (~D));
		unsigned int tempD = D;
		D = C;
		C = B;
		unsigned int X=I;
		unsigned int g=(7*i) & 15;
		if (i < 48) { X = H; g=(3*i+5) & 15; }
		if (i < 32) { X = G; g=(5*i+1) & 15; }
		if (i < 16) { X = F; g=i; }

		unsigned int tmp = A+X+dev_k_table[i]+M[g];
		B = B + ((tmp << dev_s_table[i]) | ((tmp & 0xffffffff) >> (32-dev_s_table[i])));
		A = tempD;
	}

   digest[0] = a0 + A;
   digest[1] = b0 + B;
   digest[2] = c0 + C;
   digest[3] = d0 + D;
}


__global__ void check_password(char *dev_result, const int *dev_digest, 
	int *dev_s_table, int *dev_k_table, int *dev_pw_count)
{
	unsigned int dg[4];
	char passwd[no_of_char +1];
	int num_digests = 10;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int index = blockId * (blockDim.x * blockDim.y) 
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	
	#ifdef THREE_LETTER
		passwd[2] = 'a' + (index / 676) % 26;	// ((index / 26) / 26) % 26;
		passwd[1] = 'a' + (index / 26) % 26;
		passwd[0] = 'a' + index % 26;
		passwd[3] = 0;
	#endif

	#ifdef FOUR_LETTER
		passwd[3] = 'a' + (index / 17576) % 26; // (((index / 26) / 26) /26) % 26;
		passwd[2] = 'a' + (index / 676) % 26;
		passwd[1] = 'a' + (index / 26) % 26;
		passwd[0] = 'a' + index % 26;
		passwd[4] = 0;
	#endif

	#ifdef SIX_LETTER
		passwd[5] = 'a' + (index / 11881376 ) % 26;
		passwd[4] = 'a' + (index / 456976 ) % 26;
		passwd[3] = 'a' + (index / 17576) % 26; // (((index / 26) / 26) /26) % 26;
		passwd[2] = 'a' + (index / 676) % 26;
		passwd[1] = 'a' + (index / 26) % 26;
		passwd[0] = 'a' + index % 26;
		passwd[6] = 0;
	#endif

	// md5(passwd,strlen(passwd),dg);
	if(index < THREAD_LIMIT){
		md5(passwd, no_of_char, dg, dev_s_table, dev_k_table);

		bool leaveLoop = false;
		int locks;

		#pragma unroll
		for (int i=0;i< num_digests; i++)
		{
			if (( dg[0] == dev_digest[i*4] ) && ( dg[1] == dev_digest[i*4+1] ) 
				&& ( dg[2] == dev_digest[i*4+2] ) && ( dg[3] == dev_digest[i*4+3] )) {
				//&& dev_pw_count[0] < 40

			    while (!leaveLoop) {
					if (atomicExch(&locks, 1u) == 0u) {
						//critical section

						int t = dev_pw_count[0];
						dev_result[ t ] = passwd[0];
						dev_result[ t +1] = passwd[1];
						dev_result[ t +2] = passwd[2];

						#ifdef THREE_LETTER
							dev_result[ t +3] = ' ';
							dev_pw_count[0] += 4;
						#endif

						#ifdef FOUR_LETTER
							dev_result[ t +3] = passwd[3];
							dev_result[ t +4] = ' ';
							dev_pw_count[0] += 5;
						#endif

						#ifdef SIX_LETTER
							dev_result[ t +3] = passwd[3];
							dev_result[ t +4] = passwd[4];
							dev_result[ t +5] = passwd[5];
							dev_result[ t +6] = ' ';
							dev_pw_count[0] += 7;
						#endif
						
			    		leaveLoop = true;
			    		atomicExch(&(locks),0u);
			        }
			    }

			}
		}
	}
	
}

int main(int argc, char** args) 
{
	char *result, *dev_result, **passwords;
	int *dev_digest, *dev_s_table, *dev_k_table, *dev_pw_count;
	const int *temp_digest;
	int res_size, digest_size, s_table_size, k_table_size, dig_res_size, pw_count[1];

	res_size = sizeof(char) * MAX_DG * (no_of_char + 1);
	digest_size = sizeof(const int) * MAX_DG * 4;
	s_table_size = sizeof(unsigned int) * 64;
	k_table_size = sizeof(unsigned int) * 64;

	result = (char *) malloc(res_size);
	cudaMalloc((void**)&dev_result, res_size);
	cudaMalloc((void**)&dev_digest, digest_size);
	cudaMalloc((void**)&dev_s_table, s_table_size);
	cudaMalloc((void**)&dev_k_table, k_table_size);
	cudaMalloc((void**)&dev_pw_count, sizeof(int));

	/*int *dev_dig_res, *dig_res;
	dig_res_size = sizeof(int) * 4;
	dig_res = (int *) malloc(dig_res_size);
	cudaMalloc((void**)&dev_dig_res, dig_res_size);*/

	assert(result != NULL);
	assert(dev_result != NULL);

	dim3  dimBlock(NO_THREADS_X, NO_THREADS_Y);	/* # threads per Block */
	dim3 dimGrid(NO_BLOCKS_X, NO_BLOCKS_Y);		/* # Blocks per Grid */

	#ifdef THREE_LETTER
		temp_digest = digests_3letters;
	#endif

	#ifdef FOUR_LETTER
		temp_digest = digests_4letters;
	#endif

	#ifdef SIX_LETTER
		temp_digest = digests_6letters;
	#endif

	pw_count[0] = 0;
	cudaMemcpy(dev_digest, temp_digest, digest_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_s_table, s_table, s_table_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_k_table, k_table, k_table_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pw_count, pw_count, sizeof(int), cudaMemcpyHostToDevice);

	check_password<<< dimGrid, dimBlock >>>(dev_result, dev_digest, dev_s_table, dev_k_table, dev_pw_count);

	cudaMemcpy(result, dev_result, res_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(dig_res, dev_dig_res, dig_res_size, cudaMemcpyDeviceToHost);

	char delim[] = " ";
	char* token;
	int i = 0;

	for (token = strtok(result, delim); token && i<MAX_DG; token = strtok(NULL, delim))
	{
	    printf("%d. %s\n",i+1, token);
	    ++i;
	}

	free(result);
	cudaFree(dev_digest);
	cudaFree(dev_result);

	return 0;

}
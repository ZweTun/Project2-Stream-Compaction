#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        #define blockSize 128
        int* obuffer;
        int* ibuffer;


        __global__ void upSweep(int n, int* idata, int layer) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

			int skip = 1 << (layer + 1); // powf(2, layer + 1);

            int i = (index) * skip;
            if (i + skip - 1 >= n) {
                return;
            }
          
           
            idata[int(i + skip - 1)] += idata[int(i + (skip >> 1) - 1)];

        
        }

        __global__ void downSweep(int n, int* idata, int layer) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
          
   
			int skip = 1 << (layer + 1); // powf(2, layer + 1);
            int i = index * skip;
            if (i + skip - 1 >= n) {
                return;
			}   
          


            int t = idata[int(i + (skip >> 1) - 1)];
            
        
            idata[int(i + (skip >> 1) - 1)] = idata[int(i + skip - 1)];
            idata[int(i + skip - 1)] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int size = 1 << ilog2ceil(n); 
       
            int* padded = new int[size];

            for (int i = 0; i < size; i++) {
                if (i < n) {
                    padded[i] = idata[i];
                }
                else {
					padded[i] = 0;
                }
                
            }

            int* obuffer;
            int* ibuffer;
            cudaMalloc((void**)&obuffer, size * sizeof(int));
            cudaMalloc((void**)&ibuffer, size * sizeof(int));

      
            cudaMemcpy(obuffer, padded, size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(ibuffer, padded, size * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

           
          
        
          
			// up sweep
	
            int numBlocks;


            for (int layer = 0; layer <= ilog2ceil(size) - 1; layer++) {
                int numThreads = size / int(powf(2, layer + 1));
                if (numThreads == 0) continue;

				numBlocks = (numThreads + blockSize - 1) / blockSize;
                upSweep<<<numBlocks, blockSize>>>(size, ibuffer, layer);
                cudaDeviceSynchronize();
               
             
            }

	        
			cudaMemset(ibuffer + size - 1, 0, sizeof(int));
         
         

           // down sweep 
   
         
            for (int layer = ilog2ceil(size) - 1; layer >= 0; layer--) {
                int numThreads = size / int(powf(2, layer + 1));
                if (numThreads == 0) continue;

                numBlocks = (numThreads + blockSize - 1) / blockSize;
                downSweep<<<numBlocks, blockSize>>>(size, ibuffer, layer);
                cudaDeviceSynchronize();
               

			}
            timer().endGpuTimer();



   
            cudaMemcpy(odata, ibuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
         
         
			delete[] padded;
			padded = nullptr;
			cudaFree(ibuffer);
			cudaFree(obuffer);
            
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
          //  timer().startGpuTimer();
            // TODO
            int count = 0;
            int* flags = new int[n];
            int* scanned = new int[n];
            int* temp = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    flags[i] = 1;
                }
                else {
                    flags[i] = 0;
                }
            }

			scan(n, scanned, flags);


            for (int i = 0; i < n; i++) {
                if (flags[i] == 1) {
                    temp[scanned[i]] = idata[i];
					count++;
                }

			}
            for (int i = 0; i < count; i++) {
                odata[i] = temp[i];
            }


         //   timer().endGpuTimer();
            return count;
        }
    }
}

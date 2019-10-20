# PC-Benchmark-Embeddings
A  simple benchmark that I wrote for personal usage. I wanted to test if upgrading the RAM modules of my PC I would speed up some of the workloads that I usually do with my PC, so I adapted some lines of code to run a little benchmark and get some measures. This is not an objective benchmark to compare hardware, just a benchmark for personal usage, probably useless to 99.99% of people. I am uploading this to store the code somewhere to be able to use it another day if I need to. 

## Usage:
```
python3 benchmark.py [-cuda]
```
-cuda flags uses the GPU for matrix operations. 

## Requeriments:

```
-tensoflow
-sklearn
-pandas
-regex
-numpy
```

## Example:
CPU: i7 8700K (Cores and cache overclocked to 5Ghz)
 
```
----------------[2 X 16gb 2400Mhz CL15]---------------------------
>>> Loading embedding from Disk to RAM step: 	13.782382 seconds
>>> Embedding length normalization step (CPU): 	0.11546 seconds
>>> Searching for vocabulary step (CPU): 	4.399283 seconds
>>> Matrix dot product step (CPU): 		41.890443 seconds
>>> Searching for nearest neighbors step (CPU): 42.723331 seconds
>>> Exporting embedding from RAM to Disk step: 	11.459138 seconds

----------------[2 X 8gb 3600Mhz CL16]---------------------------
>>> Loading embedding from Disk to RAM step: 	14.201078 seconds
>>> Embedding length normalization step (CPU): 	0.107213 seconds
>>> Searching for vocabulary step (CPU): 	4.399393 seconds
>>> Matrix dot product step (CPU): 		42.363847 seconds
>>> Searching for nearest neighbors step (CPU): 43.447828 seconds
>>> Exporting embedding from RAM to Disk step: 	11.702238 seconds
```

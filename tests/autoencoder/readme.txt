Running the program on CPU with the testConfig.json parameters gives the following output in the console:
Epoch: 0
   Batch: 0 / 9
   Batch: 1 / 9
   Batch: 2 / 9
   Batch: 3 / 9
   Batch: 4 / 9
   Batch: 5 / 9
   Batch: 6 / 9
   Batch: 7 / 9
   Batch: 8 / 9
   Batch: 9 / 9
   MSE: 0.213384

As long as the batch size is forced to be fixed (by not computing if the end of batch exceeds the dataset), the output is the following:
Epoch: 0
   Batch: 0 / 9
   Batch: 1 / 9
   Batch: 2 / 9
   Batch: 3 / 9
   Batch: 4 / 9
   Batch: 5 / 9
   Batch: 6 / 9
   Batch: 7 / 9
   Batch: 8 / 9
   MSE: 0.205711

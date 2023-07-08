# Data

This project's data represents precipitation in Brazil. Its sources are CPTEC/INPE's MERGE data product and ECMWF's singles levels seasonal forecast. Both can be download using file `../neuralnetwork/download/__main__.py`. Please notice that CDS' API is commented out, as more work should be done to consider the parameters that were given.

Moreover, data is then combined (interpolated into seasonal's spatial and time resolutions) by `../neuralnetwork/prepare/combine.py`, which is read by `../neuralnetwork/prepare/split_scale.py` to be split and saved into different files; and scaled for the Neural Network training, and also saved to `*_scaled`.
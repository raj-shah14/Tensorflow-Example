# Tensorflow-Example
Creating data using Tf record, passing it the model and inferencing

## Download the Dataset
[Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and put the picture in a structure where you have
```
./PetImages/Cat/*.jpg
./PetImages/Dog/*.jpg
```
## Creating Record Files
Run the `createtfrecord.py` script
`python createtfrecord.py` . It will generate `train.record`, `val.record`, `test.record` files

## Training the Model
If you are training on a CPU use `train_tfrecord.py` script. It will save the model in `output` directory. You can set number of Epochs by changing to the appropriate value.
If you are training on a GPU use `train_tfrecord_gpu.py` script. The model will be saved in `output_gpu` directory. You can set the number of GPU to train on by putting correct value in `num_gpus` parameter. You can set number of Epochs by changing to the appropriate value.

## Create a Frozen Graph Protobuf
You can use `freeze_graph.py` to generate frozen.pb file if you have trained on a CPU.
You can use `freeze_graph_gpu.py` to generate frozen_model.pb if you have trained using multiple GPUs.


# Tune TensorFlow Serving MKL-DNN Settings via Grid Search

Designed for use with the [TensorFlow Serving tutorials in the Intel Model Zoo](https://github.com/IntelAI/models/tree/master/docs),
this script runs an Intel-optimized deep learning model for inference in a TensorFlow Serving Docker container and loops through a set of MKL-DNN parameters so that you can exhaustively test a large search space automatically.

Presently, this script only supports the Wide and Deep model, but it can easily be extended for more models from the Intel Model Zoo. 

1. Clone the Model Zoo for Intel Architecture (use the develop branch of the internal repo if Model Zoo 1.5 is not yet publicly available)
2. Go to the tutorial folder for TensorFlow Serving Wide and Deep: `cd models/docs/recommendation/tensorflow_serving`
3. Clone this repo: `git clone https://github.com/mhbuehler/tfserving-mkl-grid-search.git`
4. Follow the Wide and Deep tutorial steps such that you have a `tensorflow/serving:mkl` docker image, the Criteo Kaggle dataset, a virtual environment, pre-trained model, and saved_model.
Once you are able to run the model server and test inference using the `run_wide_deep.py` script, you are ready to go to step 5.
5. Open the `tune_settings.sh` script for editing (`vi tfserving-mkl-grid-search/tune_settings.sh`) and edit:
     * The `docker_run()` function to point to the location of your machine's saved_model.pb directory
     * The `wide_deep()` function to point to the location of your preprocessed Criteo dataset in TFRecords format
     * The array variables `batch_sizes`, `omp_num_threads`, `intra_threads`, and `inter_threads` to express the range of values you want to search
6. Save and exit the file.
7. Now run the bash script with MODEL_NAME=wide_deep and your desired log location: `MODEL_NAME=wide_deep OUTPUT_DIR=/home/<user> bash tune_settings.sh`

The script will repeatedly start/stop the TensorFlow Serving model server and run the inference script for every combination of values you entered for `batch_sizes`, `omp_num_threads`, `intra_threads`, and `inter_threads`.
When it completes, it will print a table of result metrics from which you can select the optimal set.

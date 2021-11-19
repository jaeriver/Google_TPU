# Train and Inference CNN Models with Google_TPU
## Model Types
- MobileNet
- ResNet50
- VGG16

## Datasets
- ImageNet2012 (type : tfrecord)

### tpu_info.json <br/>
write your GCP information to use TPU

### tpu_utils
- tpu_create.sh <br/>
  create tpu_v2 using gcloud
- tpu_delete.sh <br/>
  delete tpu using gcloud
- tpu_utilization.sh <br/>
  check tpu utilization using cloud-tpu-profiler
  - to use
  ```pip install cloud-tpu-profiler```

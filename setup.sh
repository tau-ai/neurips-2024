PROJECT_ID="neurips-424615"
# gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="serviceAccount:643783993848-compute@developer.gserviceaccount.com" \
#     --role="roles/tpu.admin"
# gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="serviceAccount:643783993848-compute@developer.gserviceaccount.com" \
#     --role="roles/compute.admin"
# gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="serviceAccount:643783993848-compute@developer.gserviceaccount.com" \
#     --role="roles/storage.objectAdmin"
# gcloud iam service-accounts keys create tpu-key.json \
#     --iam-account=643783993848-compute@developer.gserviceaccount.com
    # --iam-account=tpu-service-account@$PROJECT_ID.iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS="/home/duckb/neurips/tpu-key.json"
export TPU_NAME="tpu-instance"
export NEXT_PLUGGABLE_DEVICE_USE_C_API=true
export TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so
echo "TPU_NAME=$TPU_NAME"
python3 tpu-test.py
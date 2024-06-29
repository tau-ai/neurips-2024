PROJECT_ID="neurips-424615"
SERVICE_ACCOUNT="643783993848-compute@developer.gserviceaccount.com"

# List all keys
EXISTING_KEYS=$(gcloud iam service-accounts keys list --iam-account=$SERVICE_ACCOUNT --format="value(name)")

# Delete each key
for KEY in $EXISTING_KEYS; do
    gcloud iam service-accounts keys delete $KEY --iam-account=$SERVICE_ACCOUNT --quiet
done

echo "All keys for service account $SERVICE_ACCOUNT have been deleted."
export PROJECT_ID=<YOUR_UNIQUE_LOWER_CASE_PROJECT_ID>
export BILLING_ACCOUNT_ID=<YOUR_BILLING_ACCOUNT_ID>
export APP=myapp 
export PORT=1234
export REGION="europe-west3"
export TAG="gcr.io/$PROJECT_ID/$APP"

gcloud projects create $PROJECT_ID --name="My FastAPI App"

# Set Default Project (all later commands will use it) 
gcloud config set project $PROJECT_ID

# Add Billing Account (no free lunch^^)
gcloud beta billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID
gcloud services enable cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    run.googleapis.com
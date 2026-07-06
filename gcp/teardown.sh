#!/usr/bin/env bash
# Delete every billable GCP resource created for the Agent Runtime/Registry/Gateway
# demo. Safe to re-run (ignores "not found"). APIs stay enabled (free).
set -uo pipefail
PROJECT=save-the-hibiscus
LOC=us-central1
RID=2005454783237849088

echo "== authz policy =="
gcloud beta network-security authz-policies delete hibiscus-authz-policy --location=$LOC --project=$PROJECT --quiet 2>&1 | tail -1
echo "== authz extension =="
gcloud beta service-extensions authz-extensions delete hibiscus-iap-authz --location=$LOC --project=$PROJECT --quiet 2>&1 | tail -1
echo "== registry services (endpoint + agent) =="
gcloud agent-registry services delete anthropic-api --location=$LOC --project=$PROJECT --quiet 2>&1 | tail -1
gcloud agent-registry services delete hibiscus-copilot --location=$LOC --project=$PROJECT --quiet 2>&1 | tail -1
echo "== agent gateway =="
gcloud network-services agent-gateways delete hibiscus-gateway --location=$LOC --project=$PROJECT --quiet 2>&1 | tail -1
echo "== capture candidate bucket =="
gcloud storage rm -r gs://save-the-hibiscus-captures --project=$PROJECT --quiet 2>&1 | tail -1
echo "== reasoning engine (Agent Runtime) =="
TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
curl -s -X DELETE -H "Authorization: Bearer $TOKEN" \
  "https://$LOC-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$LOC/reasoningEngines/$RID?force=true" \
  | tail -c 300; echo
echo "== done =="

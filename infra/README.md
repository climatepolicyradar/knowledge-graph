# AWS S3 bucket infra-as-code

The code in this repo manages the bucket `s3://cpr-kg-feather-files` which exists in production and has cross-account access to labs. Labs cross-account access is needed as the feather files stored in this bucket are used by processes related to model training (AWS production) and the vibe checker (AWS labs).

Vibe checker infra on AWS labs is managed separately in `vibe-checker/infra`.

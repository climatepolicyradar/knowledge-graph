import { createS3Client, getBucketName } from "@/lib/s3";
import { errorResponse, successResponse } from "@/lib/api-response";

import { ClassifierData } from "@/types/classifiers";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import cache from "@/lib/cache";

export async function GET(
  _request: Request,
  {
    params,
  }: { params: Promise<{ concept_id: string; classifier_id: string }> },
) {
  try {
    const { concept_id, classifier_id } = await params;

    // Check cache first
    const cacheKey = `classifier-${concept_id}-${classifier_id}`;
    const cachedData = cache.get(cacheKey);

    if (cachedData) {
      console.log(`Cache hit for classifier: ${concept_id}/${classifier_id}`);
      return successResponse(cachedData);
    }

    console.log(
      `Cache miss for classifier: ${concept_id}/${classifier_id}, fetching from S3...`,
    );
    const s3Client = createS3Client();
    const bucket = await getBucketName();

    const key = `${concept_id}/${classifier_id}/classifier.json`;

    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const response = await s3Client.send(command);
    const body = response.Body;

    if (!body) {
      throw new Error("No body in S3 response");
    }

    const text = await body.transformToString();
    const classifierData: ClassifierData = JSON.parse(text);

    // Store in cache
    cache.set(cacheKey, classifierData);
    console.log(
      `Classifier data cached successfully for: ${concept_id}/${classifier_id}`,
    );

    return successResponse(classifierData);
  } catch (error) {
    return errorResponse(error, 500);
  }
}

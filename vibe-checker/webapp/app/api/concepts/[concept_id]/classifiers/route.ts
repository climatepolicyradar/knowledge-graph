import { GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { createS3Client, getBucketName } from "@/lib/s3";
import { errorResponse, successResponse } from "@/lib/api-response";

import cache from "@/lib/cache";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ concept_id: string }> },
) {
  try {
    const { concept_id } = await params;

    // Check cache first
    const cacheKey = `classifiers-${concept_id}`;
    const cachedData = cache.get(cacheKey);

    if (cachedData) {
      console.log(`Cache hit for classifiers: ${concept_id}`);
      return successResponse(cachedData);
    }

    console.log(
      `Cache miss for classifiers: ${concept_id}, fetching from S3...`,
    );
    const s3Client = createS3Client();
    const bucket = await getBucketName();

    // List all objects under the concept_id prefix
    const command = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: `${concept_id}/`,
      Delimiter: "/",
    });

    const response = await s3Client.send(command);

    // Extract classifier IDs from the common prefixes
    const classifierIds =
      response.CommonPrefixes?.map((prefix) => {
        // Remove the concept_id/ prefix and trailing slash
        return prefix.Prefix?.replace(`${concept_id}/`, "").replace("/", "");
      }).filter(Boolean) || [];

    // Fetch classifier metadata for sorting
    const classifierDetails = await Promise.all(
      classifierIds.map(async (classifierId) => {
        try {
          const classifierKey = `${concept_id}/${classifierId}/classifier.json`;
          const classifierCommand = new GetObjectCommand({
            Bucket: bucket,
            Key: classifierKey,
          });
          const classifierResponse = await s3Client.send(classifierCommand);
          const classifierText =
            await classifierResponse.Body?.transformToString();

          if (classifierText) {
            const classifierData = JSON.parse(classifierText);
            return {
              id: classifierId,
              date: classifierData.date,
            };
          }
        } catch (error) {
          console.warn(
            `Could not fetch classifier metadata for ${classifierId}:`,
            error,
          );
        }

        // Return classifier without date if we couldn't fetch metadata
        return {
          id: classifierId,
          date: null,
        };
      }),
    );

    // Sort by date (most recent first), putting null dates at the end
    const sortedClassifiers = classifierDetails.sort((a, b) => {
      if (a.date === null && b.date === null) return 0;
      if (a.date === null) return 1;
      if (b.date === null) return -1;
      return new Date(b.date).getTime() - new Date(a.date).getTime();
    });

    // Return just the IDs in sorted order
    const sortedClassifierIds = sortedClassifiers.map((c) => c.id);

    // Store in cache
    cache.set(cacheKey, sortedClassifierIds);
    console.log(`Classifiers data cached successfully for: ${concept_id}`);

    return successResponse(sortedClassifierIds);
  } catch (error) {
    return errorResponse(error, 500);
  }
}

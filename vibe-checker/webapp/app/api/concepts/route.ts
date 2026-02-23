import { EnhancedConcept } from "@/types/concepts";
import { GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { createS3Client, getBucketName } from "@/lib/s3";
import { errorResponse, successResponse } from "@/lib/api-response";

import cache from "@/lib/cache";

type ClassifierDetail = { prefix: string; date: string | null };

export async function GET() {
  try {
    const cacheKey = "concepts";
    const cachedData = cache.get(cacheKey);

    if (cachedData) {
      console.log("Cache hit for concepts");
      return successResponse(cachedData);
    }

    console.log("Cache miss for concepts, fetching from S3...");
    const s3Client = createS3Client();
    const bucket = await getBucketName();

    // Discover concept IDs from top-level S3 prefixes (e.g. "Q123/")
    const listCommand = new ListObjectsV2Command({
      Bucket: bucket,
      Delimiter: "/",
    });
    const listResponse = await s3Client.send(listCommand);

    const conceptIds = (listResponse.CommonPrefixes ?? [])
      .map((prefix: { Prefix?: string }) => prefix.Prefix?.replace("/", "") ?? "")
      .filter((id: string) => /^Q\d+$/.test(id));

    const enhancedConcepts: EnhancedConcept[] = await Promise.all(
      conceptIds.map(async (conceptId: string) => {
        const fallback: EnhancedConcept = {
          wikibase_id: conceptId,
          preferred_label: conceptId,
          description: `Concept ${conceptId}`,
          n_classifiers: 0,
        };

        try {
          // List classifier subdirectories for this concept
          const classifierListCommand = new ListObjectsV2Command({
            Bucket: bucket,
            Prefix: `${conceptId}/`,
            Delimiter: "/",
          });
          const classifierListResponse = await s3Client.send(
            classifierListCommand,
          );
          const classifierPrefixes = classifierListResponse.CommonPrefixes ?? [];
          const n_classifiers = classifierPrefixes.length;

          if (n_classifiers === 0) return fallback;

          // Find the most recent classifier by date from classifier.json
          const classifierDetails: ClassifierDetail[] = await Promise.all(
            classifierPrefixes.map(async (prefix: { Prefix?: string }) => {
              try {
                const command = new GetObjectCommand({
                  Bucket: bucket,
                  Key: `${prefix.Prefix}classifier.json`,
                });
                const response = await s3Client.send(command);
                const text = await response.Body?.transformToString();
                if (text) {
                  const data = JSON.parse(text);
                  return { prefix: prefix.Prefix!, date: data.date as string | null };
                }
              } catch {
                // ignore missing classifier.json
              }
              return { prefix: prefix.Prefix!, date: null };
            }),
          );

          const latestPrefix = classifierDetails.sort(
            (a: ClassifierDetail, b: ClassifierDetail) => {
              if (!a.date && !b.date) return 0;
              if (!a.date) return 1;
              if (!b.date) return -1;
              return new Date(b.date).getTime() - new Date(a.date).getTime();
            },
          )[0].prefix;

          // Read concept metadata from the latest classifier's concept.json
          try {
            const command = new GetObjectCommand({
              Bucket: bucket,
              Key: `${latestPrefix}concept.json`,
            });
            const response = await s3Client.send(command);
            const text = await response.Body?.transformToString();
            if (text) {
              const conceptMetadata = JSON.parse(text);
              return {
                wikibase_id: conceptId,
                preferred_label: conceptMetadata.preferred_label || conceptId,
                description:
                  conceptMetadata.description || `Concept ${conceptId}`,
                n_classifiers,
              };
            }
          } catch {
            // fall through to fallback with n_classifiers
          }

          return { ...fallback, n_classifiers };
        } catch (error) {
          console.warn(`Failed to fetch concept data for ${conceptId}:`, error);
          return fallback;
        }
      }),
    );

    enhancedConcepts.sort((a, b) =>
      a.preferred_label.localeCompare(b.preferred_label),
    );

    cache.set(cacheKey, enhancedConcepts);
    console.log("Concepts data cached successfully");

    return successResponse(enhancedConcepts);
  } catch (error) {
    return errorResponse(error, 500);
  }
}

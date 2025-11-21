import { EnhancedConcept, YamlConcept } from "@/types/concepts";
import { GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { createS3Client, getBucketName } from "@/lib/s3";
import { errorResponse, successResponse } from "@/lib/api-response";

import cache from "@/lib/cache";
import { getConceptDefaults } from "@/lib/concept-helpers";
import { load } from "js-yaml";

export async function GET() {
  try {
    // Check cache first
    const cacheKey = "concepts";
    const cachedData = cache.get(cacheKey);

    if (cachedData) {
      console.log("Cache hit for concepts");
      return successResponse(cachedData);
    }

    console.log("Cache miss for concepts, fetching from S3...");
    const s3Client = createS3Client();
    const bucket = await getBucketName();

    // Fetch concepts.yml
    const conceptsKey = "concepts.yml";
    const conceptsCommand = new GetObjectCommand({
      Bucket: bucket,
      Key: conceptsKey,
    });
    const conceptsResponse = await s3Client.send(conceptsCommand);
    const conceptsBody = conceptsResponse.Body;

    if (!conceptsBody) {
      throw new Error("No body in concepts.yml S3 response");
    }

    const yaml_content = await conceptsBody.transformToString();
    const concepts = load(yaml_content) as (string | YamlConcept)[];

    // List all objects to find concept.json files
    const listCommand = new ListObjectsV2Command({ Bucket: bucket });
    const listResponse = await s3Client.send(listCommand);
    const objects = listResponse.Contents || [];

    // Filter for concept.json files and group by concept_id
    const conceptJsonPaths: { [conceptId: string]: string[] } = {};
    for (const obj of objects) {
      if (obj.Key && obj.Key.endsWith("/concept.json")) {
        // Extract concept_id from path like "Q123/classifier_id/concept.json"
        const parts = obj.Key.split("/");
        if (parts.length >= 3) {
          const conceptId = parts[0];
          if (!conceptJsonPaths[conceptId]) {
            conceptJsonPaths[conceptId] = [];
          }
          conceptJsonPaths[conceptId].push(obj.Key);
        }
      }
    }

    // For each concept, get the latest classifier's concept.json
    const enhancedConcepts: EnhancedConcept[] = [];
    for (const concept of concepts) {
      const conceptId =
        typeof concept === "string"
          ? concept
          : concept.id || concept.wikibase_id;

      if (!conceptId) continue;

      const conceptJsonKeys = conceptJsonPaths[conceptId];

      if (conceptJsonKeys && conceptJsonKeys.length > 0) {
        // Sort keys to get the latest classifier (using explicit localeCompare)
        conceptJsonKeys.sort((a, b) => a.localeCompare(b));
        const latestConceptJsonKey =
          conceptJsonKeys[conceptJsonKeys.length - 1];

        try {
          const conceptJsonCommand = new GetObjectCommand({
            Bucket: bucket,
            Key: latestConceptJsonKey,
          });
          const conceptJsonResponse = await s3Client.send(conceptJsonCommand);
          const conceptJsonBody = conceptJsonResponse.Body;

          if (conceptJsonBody) {
            const conceptJsonContent =
              await conceptJsonBody.transformToString();
            const conceptMetadata = JSON.parse(conceptJsonContent);
            const defaults = getConceptDefaults(concept, conceptId);

            enhancedConcepts.push({
              ...defaults,
              preferred_label:
                conceptMetadata.preferred_label || defaults.preferred_label,
              description: conceptMetadata.description || defaults.description,
              n_classifiers: conceptJsonKeys.length,
            });
          } else {
            const defaults = getConceptDefaults(concept, conceptId);
            enhancedConcepts.push({
              ...defaults,
              n_classifiers: conceptJsonKeys.length,
            });
          }
        } catch (error) {
          console.warn(`Failed to fetch concept.json for ${conceptId}:`, error);
          const defaults = getConceptDefaults(concept, conceptId);
          enhancedConcepts.push({
            ...defaults,
            n_classifiers: 0,
          });
        }
      } else {
        const defaults = getConceptDefaults(concept, conceptId);
        enhancedConcepts.push({
          ...defaults,
          n_classifiers: 0,
        });
      }
    }

    // Store in cache
    cache.set(cacheKey, enhancedConcepts);
    console.log("Concepts data cached successfully");

    return successResponse(enhancedConcepts);
  } catch (error) {
    return errorResponse(error, 500);
  }
}

import { createS3Client, getBucketName } from "@/lib/s3";

import { FilterParams } from "@/types/filters";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { NextResponse } from "next/server";
import { PAGINATION } from "@/lib/constants";
import { Prediction } from "@/types/predictions";
import cache from "@/lib/cache";
import { errorResponse } from "@/lib/api-response";
import { filterPredictions } from "@/lib/prediction-filters";

export async function GET(
  request: Request,
  {
    params,
  }: { params: Promise<{ wikibase_id: string; classifier_id: string }> },
) {
  try {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const pageSize = PAGINATION.DEFAULT_PAGE_SIZE;

    // Parse filter parameters
    const filters: FilterParams = {
      translated: searchParams.get("translated")
        ? searchParams.get("translated") === "true"
        : undefined,
      corpus_type: searchParams.get("corpus_type") || undefined,
      world_bank_region: searchParams.get("world_bank_region") || undefined,
      publication_year_start: searchParams.get("publication_year_start")
        ? parseInt(searchParams.get("publication_year_start") || "0")
        : undefined,
      publication_year_end: searchParams.get("publication_year_end")
        ? parseInt(searchParams.get("publication_year_end") || "0")
        : undefined,
      document_id: searchParams.get("document_id") || undefined,
      search: searchParams.get("search") || undefined,
      has_predictions: searchParams.get("has_predictions")
        ? searchParams.get("has_predictions") === "true"
        : undefined,
    };

    // Validate pagination parameters
    if (page < PAGINATION.MIN_PAGE) {
      return errorResponse(
        new Error("Invalid pagination parameters. Page must be >= 1"),
        400,
      );
    }

    const { wikibase_id, classifier_id } = await params;
    const key = `${wikibase_id}/${classifier_id}/predictions.jsonl`;

    // Check cache first
    const cacheKey = `predictions-${wikibase_id}-${classifier_id}`;
    let allPredictions = cache.get<Prediction[]>(cacheKey);

    if (!allPredictions) {
      console.log(`Cache miss for predictions: ${key}, fetching from S3...`);

      const s3Client = createS3Client();
      const bucket = await getBucketName();

      const command = new GetObjectCommand({ Bucket: bucket, Key: key });
      const response = await s3Client.send(command);
      const body = response.Body;

      if (!body) {
        throw new Error("No body in S3 response");
      }

      const text = await body.transformToString();

      // Parse JSONL data
      const lines = text.split("\n").filter((line) => line.trim() !== "");
      allPredictions = lines.map((line) => JSON.parse(line)) as Prediction[];

      // Store in cache
      cache.set(cacheKey, allPredictions);
      console.log(`Predictions data cached successfully for: ${key}`);
    } else {
      console.log(`Cache hit for predictions: ${key}`);
    }

    if (!allPredictions) {
      throw new Error("Failed to load predictions data");
    }

    // Store total unfiltered count
    const totalUnfiltered = allPredictions.length;

    // Apply filters using extracted function
    const filteredPredictions = filterPredictions(allPredictions, filters);

    // Apply pagination to filtered results
    const totalFiltered = filteredPredictions.length;
    const totalPages = Math.ceil(totalFiltered / pageSize);
    const startIndex = (page - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const paginatedPassages = filteredPredictions.slice(startIndex, endIndex);

    // Extract unique values for filter options
    const uniqueCorpusTypes = [
      ...new Set(
        allPredictions.map(
          (p) => p.metadata["document_metadata.corpus_type_name"],
        ),
      ),
    ].sort();
    const uniqueRegions = [
      ...new Set(allPredictions.map((p) => p.metadata.world_bank_region)),
    ].sort();

    return NextResponse.json({
      success: true,
      data: paginatedPassages,
      pagination: {
        page,
        pageSize,
        totalFiltered,
        totalUnfiltered,
        totalPages,
        hasNextPage: page < totalPages,
        hasPreviousPage: page > 1,
      },
      filters: {
        availableCorpusTypes: uniqueCorpusTypes,
        availableRegions: uniqueRegions,
        applied: filters,
      },
    });
  } catch (error) {
    return errorResponse(error, 500);
  }
}

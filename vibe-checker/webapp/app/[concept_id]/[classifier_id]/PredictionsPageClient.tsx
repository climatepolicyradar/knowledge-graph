"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import Breadcrumb from "@/components/Breadcrumb";
import { ConceptData } from "@/types/concepts";
import ConceptHeader from "@/components/ConceptHeader";
import { DEBOUNCE } from "@/lib/constants";
import ErrorMessage from "@/components/ErrorMessage";
import { FilterState } from "@/types/filters";
import LabelledPassage from "@/components/LabelledPassage";
import LoadingSpinner from "@/components/LoadingSpinner";
import MaterialIcon from "@/components/MaterialIcon";
import PaginationControls from "@/components/PaginationControls";
import { Prediction } from "@/types/predictions";
import PredictionFilters from "@/components/PredictionFilters";
import SearchBox from "@/components/SearchBox";
import { enrichConceptData } from "@/lib/concept-helpers";

interface PredictionsPageClientProps {
  conceptId: string;
  classifierId: string;
}

export default function PredictionsPageClient({
  conceptId,
  classifierId,
}: PredictionsPageClientProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  // API response state
  const [pageSize, setPageSize] = useState<number>(0); // Will be inferred from results
  const [hasNextPage, setHasNextPage] = useState<boolean>(false);
  const [hasPreviousPage, setHasPreviousPage] = useState<boolean>(false);
  const [totalFiltered, setTotalFiltered] = useState<number>(0);
  const [totalUnfiltered, setTotalUnfiltered] = useState<number>(0);
  const [availableCorpusTypes, setAvailableCorpusTypes] = useState<string[]>(
    [],
  );
  const [availableRegions, setAvailableRegions] = useState<string[]>([]);

  // Concept metadata
  const [conceptData, setConceptData] = useState<ConceptData | null>(null);

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  // Memoize URL-derived state to prevent unnecessary re-renders
  const urlState = useMemo(() => {
    const page = parseInt(searchParams.get("page") || "1");
    const filters: FilterState = {
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
      has_predictions: searchParams.get("has_predictions")
        ? searchParams.get("has_predictions") === "true"
        : undefined,
    };
    const searchTerms = searchParams.get("search") || undefined;
    return { page, filters, searchTerms };
  }, [searchParams]);

  // Memoized URL update function
  const updateURL = useCallback(
    (newFilters: FilterState, newPage: number, searchTerms?: string) => {
      const params = new URLSearchParams();

      // Add page
      if (newPage > 1) {
        params.set("page", newPage.toString());
      }

      // Add filters
      if (newFilters.translated !== undefined) {
        params.set("translated", newFilters.translated.toString());
      }
      if (newFilters.corpus_type) {
        params.set("corpus_type", newFilters.corpus_type);
      }
      if (newFilters.world_bank_region) {
        params.set("world_bank_region", newFilters.world_bank_region);
      }
      if (newFilters.publication_year_start) {
        params.set(
          "publication_year_start",
          newFilters.publication_year_start.toString(),
        );
      }
      if (newFilters.publication_year_end) {
        params.set(
          "publication_year_end",
          newFilters.publication_year_end.toString(),
        );
      }
      if (newFilters.document_id) {
        params.set("document_id", newFilters.document_id);
      }
      if (newFilters.has_predictions !== undefined) {
        params.set("has_predictions", newFilters.has_predictions.toString());
      }
      if (searchTerms) {
        params.set("search", searchTerms);
      }

      const newURL = params.toString()
        ? `/${conceptId}/${classifierId}?${params.toString()}`
        : `/${conceptId}/${classifierId}`;

      router.replace(newURL);
    },
    [conceptId, classifierId, router],
  );

  // Search handler
  const handleSearchChange = useCallback(
    (newSearchTerms: string) => {
      updateURL(urlState.filters, 1, newSearchTerms || undefined); // Reset to page 1 on search
    },
    [updateURL, urlState.filters],
  );

  // Fetch concept metadata first
  useEffect(() => {
    const fetchConceptData = async () => {
      try {
        const response = await fetch(`/api/concepts`);
        const result = await response.json();
        if (result.success && result.data) {
          const concepts: ConceptData[] = Array.isArray(result.data)
            ? result.data.map(enrichConceptData)
            : [];
          const concept = concepts.find((c) => c.wikibase_id === conceptId);
          if (concept) {
            setConceptData(concept);
          }
        }
      } catch (error) {
        console.error("Failed to fetch concept data:", error);
      }
    };

    if (conceptId) {
      fetchConceptData();
    }
  }, [conceptId]);

  // Fetch predictions based on URL state
  useEffect(() => {
    const fetchPredictions = async () => {
      if (!conceptId || !classifierId) return;

      setLoading(true);
      try {
        const { page, filters, searchTerms } = urlState;

        const params = new URLSearchParams();
        params.set("page", page.toString());
        if (searchTerms) {
          params.set("search", searchTerms);
        }

        // Add filter parameters
        Object.entries(filters).forEach(([key, value]) => {
          if (value !== undefined && value !== "") {
            params.set(key, value.toString());
          }
        });

        const queryString = params.toString();
        const response = await fetch(
          `/api/predictions/${conceptId}/${classifierId}?${queryString}`,
        );
        const result = await response.json();

        if (result.success) {
          setPredictions(result.data);

          // Infer page size from results
          if (result.data && result.data.length > 0) {
            setPageSize(result.data.length);
          }

          // Update pagination state
          if (result.pagination) {
            setHasNextPage(result.pagination.hasNextPage);
            setHasPreviousPage(result.pagination.hasPreviousPage);
            setTotalFiltered(result.pagination.totalFiltered);
            setTotalUnfiltered(result.pagination.totalUnfiltered);
          }

          // Update available filter options
          if (result.filters) {
            setAvailableCorpusTypes(result.filters.availableCorpusTypes || []);
            setAvailableRegions(result.filters.availableRegions || []);
          }
        } else {
          throw new Error(result.error || "Failed to fetch predictions");
        }
      } catch (err) {
        console.error("Error fetching predictions:", err);
        setError(
          `Failed to fetch predictions: ${
            err instanceof Error ? err.message : "Unknown error"
          }`,
        );
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [conceptId, classifierId, urlState, pageSize]);

  // Memoized event handlers
  const handleFilterChange = useCallback(
    (newFilters: FilterState) => {
      const newPage = 1; // Reset to first page when filters change
      updateURL(newFilters, newPage, urlState.searchTerms);
    },
    [updateURL, urlState.searchTerms],
  );

  const handlePageChange = useCallback(
    (newPage: number) => {
      updateURL(urlState.filters, newPage, urlState.searchTerms);
    },
    [updateURL, urlState.filters, urlState.searchTerms],
  );

  return (
    <div className="page-container">
      <div className="mx-auto max-w-7xl">
        <Breadcrumb href={`/${conceptId}`}>Back to classifiers</Breadcrumb>

        <ConceptHeader
          conceptId={conceptId}
          classifierId={classifierId}
          conceptData={conceptData ?? undefined}
        />

        <div className="card mb-6 p-6">
          <div className="flex flex-col gap-4">
            {/* Search and Actions */}
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              {/* Search Input */}
              <div className="max-w-md flex-1">
                <SearchBox
                  value={urlState.searchTerms || ""}
                  onChange={handleSearchChange}
                  placeholder="Search in passage text..."
                  debounceMs={DEBOUNCE.FILTERS}
                />
              </div>

              {/* Download Button */}
              <div className="flex items-center gap-4">
                <a
                  href={`/api/predictions/${conceptId}/${classifierId}/download`}
                  title="Download JSON"
                  className="btn-primary flex items-center gap-2 px-4 py-2 text-sm"
                  download
                >
                  <MaterialIcon name="download" size={16} />
                  Download JSON
                </a>
              </div>
            </div>

            {/* Filters */}
            <PredictionFilters
              filters={urlState.filters}
              onFilterChange={handleFilterChange}
              availableCorpusTypes={availableCorpusTypes}
              availableRegions={availableRegions}
              totalFiltered={totalFiltered}
              totalUnfiltered={totalUnfiltered}
              searchTerms={urlState.searchTerms}
            />
          </div>
        </div>

        {/* Pagination controls */}
        {!loading && !error && (
          <PaginationControls
            page={urlState.page}
            totalFiltered={totalFiltered}
            pageSize={pageSize}
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            onPageChange={handlePageChange}
          />
        )}

        {loading && <LoadingSpinner message="Loading predictions..." />}

        {error && <ErrorMessage error={error} />}

        {/* Passages container */}
        {!loading && !error && predictions.length > 0 && (
          <div id="passages-container" className="space-y-4">
            {predictions.map((prediction, index) => (
              <LabelledPassage
                key={index}
                text={prediction.text}
                marked_up_text={prediction.marked_up_text}
                spans={prediction.spans}
                metadata={prediction.metadata}
              />
            ))}
          </div>
        )}

        {/* Bottom pagination controls */}
        {!loading && !error && (
          <div className="mt-6">
            <PaginationControls
              page={urlState.page}
              totalFiltered={totalFiltered}
              pageSize={pageSize}
              hasNextPage={hasNextPage}
              hasPreviousPage={hasPreviousPage}
              onPageChange={handlePageChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";

import Card from "../components/Card";
import { ConceptData } from "@/types/concepts";
import { DEBOUNCE } from "@/lib/constants";
import ErrorMessage from "../components/ErrorMessage";
import LoadingSpinner from "../components/LoadingSpinner";
import SearchBox from "../components/SearchBox";
import { enrichConceptData } from "@/lib/concept-helpers";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  const [concepts, setConcepts] = useState<ConceptData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState<string>("");

  useEffect(() => {
    const fetchConcepts = async () => {
      try {
        const response = await fetch("/api/concepts");
        const result = await response.json();

        if (result.success) {
          const conceptData: ConceptData[] = (result.data as ConceptData[]).map(
            enrichConceptData,
          );
          setConcepts(conceptData);
        } else {
          throw new Error(result.error || "Failed to fetch concepts");
        }
      } catch (err) {
        console.error("Error fetching concepts:", err);
        setError(
          `Failed to fetch concepts: ${
            err instanceof Error ? err.message : "Unknown error"
          }`,
        );
      } finally {
        setLoading(false);
      }
    };

    fetchConcepts();
  }, []);

  // Memoize filtered concepts to prevent unnecessary recalculations
  const filteredConcepts = useMemo(() => {
    if (!searchTerm.trim()) return concepts;
    return concepts.filter(
      (concept) =>
        concept.preferred_label
          ?.toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        concept.wikibase_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        concept.description?.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }, [searchTerm, concepts]);

  return (
    <div className="page-container">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8">
          <div className="mb-4 md:mb-0 md:flex md:items-center md:justify-between">
            <div>
              <h1 className="page-title">Vibe Checker</h1>
              <p className="page-subtitle">
                Check predictions from candidate classifiers
              </p>
            </div>
          </div>
          <div className="mt-4 w-full max-w-sm md:max-w-md">
            <SearchBox
              value={searchTerm}
              onChange={setSearchTerm}
              placeholder="Search for a concept"
              debounceMs={DEBOUNCE.SEARCH}
            />
          </div>
        </div>

        {loading && <LoadingSpinner message="Loading concepts..." />}

        {error && <ErrorMessage error={error} />}

        {!loading && !error && (
          <Card className="overflow-hidden">
            <table className="w-full table-auto text-left">
              <thead>
                <tr>
                  <th className="w-32 border-b border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Wikibase ID
                    </p>
                  </th>
                  <th className="border-b border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Preferred Label
                    </p>
                  </th>
                  <th className="hidden border-b border-neutral-200 bg-neutral-50 p-4 md:table-cell dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Description
                    </p>
                  </th>
                  <th className="hidden w-32 border-b border-neutral-200 bg-neutral-50 p-4 md:table-cell dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Classifiers
                    </p>
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredConcepts.map((concept) => (
                  <tr
                    key={concept.wikibase_id}
                    className="cursor-pointer transition-colors hover:bg-neutral-50 dark:hover:bg-neutral-700"
                    onClick={() => router.push(`/${concept.wikibase_id}`)}
                  >
                    <td className="border-b border-neutral-200 p-4 dark:border-neutral-600">
                      <p className="font-medium break-words text-neutral-900 dark:text-neutral-100">
                        {concept.wikibase_id}
                      </p>
                    </td>
                    <td className="border-b border-neutral-200 p-4 dark:border-neutral-600">
                      <p className="text-sm break-words text-neutral-600 dark:text-neutral-400">
                        {concept.preferred_label}
                      </p>
                    </td>
                    <td className="hidden border-b border-neutral-200 p-4 md:table-cell dark:border-neutral-600">
                      <p className="text-sm break-words text-neutral-600 dark:text-neutral-400">
                        {concept.description}
                      </p>
                    </td>
                    <td className="hidden border-b border-neutral-200 p-4 md:table-cell dark:border-neutral-600">
                      <p className="text-center text-sm text-neutral-600 dark:text-neutral-400">
                        {concept.n_classifiers || 0}
                      </p>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filteredConcepts.length === 0 && concepts.length > 0 && (
              <div className="p-12 text-center">
                <p className="text-neutral-600 dark:text-neutral-400">
                  No concepts found matching &quot;
                  <span className="font-medium">{searchTerm}</span>&quot;
                </p>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";

import Breadcrumb from "@/components/Breadcrumb";
import { ClassifierInfo } from "@/types/classifiers";
import { ConceptData } from "@/types/concepts";
import ConceptHeader from "@/components/ConceptHeader";
import ErrorMessage from "@/components/ErrorMessage";
import LoadingSpinner from "@/components/LoadingSpinner";
import MaterialIcon from "@/components/MaterialIcon";
import { enrichConceptData } from "@/lib/concept-helpers";
import { useRouter } from "next/navigation";

interface ConceptPageClientProps {
  conceptId: string;
}

export default function ConceptPageClient({
  conceptId,
}: ConceptPageClientProps) {
  const router = useRouter();
  const [classifiers, setClassifiers] = useState<ClassifierInfo[]>([]);
  const [conceptData, setConceptData] = useState<ConceptData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const fetchClassifiers = async () => {
      try {
        // Fetch classifiers list and concept data in parallel
        const [classifiersResponse, conceptsResponse] = await Promise.all([
          fetch(`/api/concepts/${conceptId}/classifiers`),
          fetch("/api/concepts"),
        ]);

        const classifiersResult = await classifiersResponse.json();
        const conceptsResult = await conceptsResponse.json();

        if (classifiersResult.success) {
          // Initialize classifiers with loading state
          const classifierInfos: ClassifierInfo[] = classifiersResult.data.map(
            (id: string) => ({
              id,
              loading: true,
            }),
          );

          // Fetch concept metadata from concepts list
          if (conceptsResult.success) {
            const concepts: ConceptData[] = Array.isArray(conceptsResult.data)
              ? conceptsResult.data.map(enrichConceptData)
              : [];
            const concept = concepts.find((c) => c.wikibase_id === conceptId);
            if (concept) {
              setConceptData(concept);
            }
          }

          // Fetch detailed data for each classifier in parallel
          const classifierDetails = await Promise.all(
            classifierInfos.map(async (classifierInfo) => {
              try {
                const classifierResponse = await fetch(
                  `/api/concepts/${conceptId}/classifiers/${classifierInfo.id}`,
                );
                const classifierResult = await classifierResponse.json();
                return classifierResult.success ? classifierResult.data : null;
              } catch (err) {
                console.error(err);
                return null;
              }
            }),
          );

          // Update classifiers with fetched data in a single batch
          setClassifiers(
            classifierInfos.map((info, i) => ({
              ...info,
              data: classifierDetails[i] || undefined,
              loading: false,
              error: classifierDetails[i] ? undefined : "Failed to load",
            })),
          );
        } else {
          throw new Error(
            classifiersResult.error || "Failed to fetch classifiers",
          );
        }
      } catch (err) {
        console.error("Error fetching classifiers:", err);
        setError(
          `Failed to fetch classifiers: ${
            err instanceof Error ? err.message : "Unknown error"
          }`,
        );
      } finally {
        setLoading(false);
      }
    };

    if (conceptId) {
      fetchClassifiers();
    }
  }, [conceptId]);

  return (
    <div className="page-container">
      <div className="mx-auto max-w-7xl">
        <Breadcrumb href="/">Back to all concepts</Breadcrumb>

        <ConceptHeader
          conceptId={conceptId}
          conceptData={conceptData ?? undefined}
        />

        {loading && <LoadingSpinner message="Loading classifiers..." />}

        {error && <ErrorMessage error={error} />}

        {!loading && !error && classifiers.length > 0 && (
          <div className="card overflow-hidden">
            <table className="w-full table-auto text-left">
              <thead>
                <tr>
                  <th className="w-40 border-b border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      ID
                    </p>
                  </th>
                  <th className="hidden border-b border-neutral-200 bg-neutral-50 p-4 md:table-cell dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Name
                    </p>
                  </th>
                  <th className="w-32 border-b border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-600 dark:bg-neutral-700">
                    <p className="text-sm font-medium text-neutral-600 dark:text-neutral-300">
                      Created
                    </p>
                  </th>
                </tr>
              </thead>
              <tbody>
                {classifiers.map((classifier) => (
                  <tr
                    key={classifier.id}
                    className="cursor-pointer transition-colors hover:bg-neutral-50 dark:hover:bg-neutral-700"
                    onClick={() =>
                      router.push(`/${conceptId}/${classifier.id}`)
                    }
                  >
                    <td className="border-b border-neutral-200 p-4 dark:border-neutral-600">
                      <div className="flex items-center gap-2">
                        <p className="font-medium break-words text-neutral-900 dark:text-neutral-100">
                          {classifier.id}
                        </p>
                        {classifier.loading && (
                          <MaterialIcon
                            name="progress_activity"
                            size={16}
                            className="animate-spin text-neutral-400"
                          />
                        )}
                      </div>
                    </td>
                    <td className="hidden border-b border-neutral-200 p-4 md:table-cell dark:border-neutral-600">
                      {classifier.data ? (
                        <p className="text-sm break-words text-neutral-600 dark:text-neutral-400">
                          {classifier.data.name}
                        </p>
                      ) : classifier.error ? (
                        <p className="text-sm text-red-600">Failed to load</p>
                      ) : classifier.loading ? (
                        <p className="text-sm text-neutral-400">Loading...</p>
                      ) : (
                        <p className="text-sm text-neutral-400">—</p>
                      )}
                    </td>
                    <td className="border-b border-neutral-200 p-4 dark:border-neutral-600">
                      {classifier.data ? (
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          {new Date(classifier.data.date).toLocaleDateString()}
                        </p>
                      ) : (
                        <p className="text-sm text-neutral-400">—</p>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!loading && !error && classifiers.length === 0 && (
          <div className="card p-12 text-center">
            <div className="text-secondary">
              <p>
                No classifiers found for concept{" "}
                <span className="text-primary font-medium">{conceptId}</span>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

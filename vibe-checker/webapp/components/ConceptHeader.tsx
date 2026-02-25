import Link from "next/link";
import ExternalLink from "./ExternalLink";
import { buildWikibaseUrl } from "@/lib/urls";

interface ConceptHeaderProps {
  conceptId: string;
  classifierId?: string;
  conceptData?: {
    preferred_label?: string;
    description?: string;
  };
}

export default function ConceptHeader({
  conceptId,
  classifierId,
  conceptData,
}: ConceptHeaderProps) {
  return (
    <div className="card mb-6 p-6">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="page-title">
            {classifierId ? (
              <>
                <Link
                  href={`/${conceptId}`}
                  className="text-accent-primary transition-colors hover:text-accent-secondary"
                >
                  {conceptId}
                </Link>
                <span className="text-secondary"> / </span>
                <span className="text-primary">{classifierId}</span>
              </>
            ) : (
              conceptId
            )}
          </h1>

          {/* Concept metadata */}
          {(conceptData?.preferred_label || conceptData?.description) && (
            <div className="mt-3 space-y-1">
              {conceptData.preferred_label && (
                <p className="text-sm font-medium text-secondary max-w-2xl">
                  {conceptData.preferred_label}
                </p>
              )}
              {conceptData.description && (
                <p className="text-sm text-secondary max-w-2xl">
                  {conceptData.description}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="flex-shrink-0">
          <ExternalLink
            href={buildWikibaseUrl(conceptId)}
            className="text-sm text-secondary hover:text-primary"
          >
            View in Wikibase
          </ExternalLink>
        </div>
      </div>
    </div>
  );
}

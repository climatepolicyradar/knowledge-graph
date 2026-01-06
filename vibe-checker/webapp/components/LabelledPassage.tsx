import ExternalLink from "./ExternalLink";
import { PredictionMetadata } from "@/types/predictions";
import { buildDocumentUrl } from "@/lib/urls";

interface LabelledPassageProps {
  text: string;
  marked_up_text?: string;
  spans?: Array<{ start: number; end: number; label: string }>;
  metadata: PredictionMetadata;
}

export default function LabelledPassage({
  text,
  marked_up_text,
  metadata,
}: LabelledPassageProps) {
  // Build CPR document URL
  const familySlug = metadata["document_metadata.family_slug"] || "";
  const documentSlug = metadata["document_metadata.slug"] || "";
  const pageNumber = Number(metadata["text_block.page_number"]) + 1;
  const documentUrl = buildDocumentUrl(documentSlug, familySlug, pageNumber);

  // Parse publication date
  const pubDate = new Date(metadata["document_metadata.publication_ts"]);

  // Function to render text with highlights
  const renderPassageText = () => {
    // Use marked_up_text directly - pipeline includes proper CSS classes
    if (marked_up_text) {
      return <span dangerouslySetInnerHTML={{ __html: marked_up_text }} />;
    }

    // Fallback to plain text if no marked_up_text available
    return <span>{text}</span>;
  };

  return (
    <div className="passage-card card p-6 transition duration-300 hover:shadow-md">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded bg-bg-tertiary px-2 py-1 text-text-primary">
            {metadata["document_metadata.corpus_type_name"]}
          </span>
          <span className="rounded bg-bg-tertiary px-2 py-1 text-text-primary">
            {metadata.world_bank_region}
          </span>
          <span className="rounded bg-bg-tertiary px-2 py-1 text-text-primary">
            {metadata.translated === "True" ? "Translated" : "Original"}
          </span>
          <span className="rounded bg-bg-tertiary px-2 py-1 text-text-primary">
            {pubDate.getFullYear()}
          </span>
        </div>
        <ExternalLink
          href={documentUrl}
          className="text-secondary hover:text-primary text-xs flex-shrink-0 ml-4"
        >
          View in CPR
        </ExternalLink>
      </div>

      <div className="text-secondary mb-2 text-sm">
        {metadata.document_id} | Page {pageNumber}
      </div>

      <div className="passage-text text-primary leading-relaxed">
        {renderPassageText()}
      </div>
    </div>
  );
}

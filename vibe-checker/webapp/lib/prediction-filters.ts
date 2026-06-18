import { FilterParams } from "@/types/filters";
import { Prediction } from "@/types/predictions";

/**
 * Apply the user-selected filters to a set of predictions, in-memory.
 */
export function filterPredictions(
  predictions: Prediction[],
  filters: FilterParams,
): Prediction[] {
  return predictions.filter((prediction) => {
    const metadata = prediction.metadata;

    if (
      filters.translated !== undefined &&
      (metadata.translated === "True") !== filters.translated
    ) {
      return false;
    }

    if (
      filters.corpus_type &&
      metadata["document_metadata.corpus_type_name"] !== filters.corpus_type
    ) {
      return false;
    }

    if (
      filters.world_bank_region &&
      metadata.world_bank_region !== filters.world_bank_region
    ) {
      return false;
    }

    if (filters.publication_year_start || filters.publication_year_end) {
      const year = new Date(metadata.publication_ts).getFullYear();
      if (
        (filters.publication_year_start &&
          year < filters.publication_year_start) ||
        (filters.publication_year_end && year > filters.publication_year_end)
      ) {
        return false;
      }
    }

    if (
      filters.document_id &&
      !metadata.document_id
        .toLowerCase()
        .includes(filters.document_id.toLowerCase())
    ) {
      return false;
    }

    if (filters.search) {
      try {
        const regex = new RegExp(filters.search.trim(), "i");
        if (!regex.test(prediction.text)) return false;
      } catch {
        const term = filters.search.toLowerCase().trim();
        if (!prediction.text.toLowerCase().includes(term)) return false;
      }
    }

    return (
      filters.has_predictions === undefined ||
      (prediction.spans && prediction.spans.length > 0) ===
        filters.has_predictions
    );
  });
}

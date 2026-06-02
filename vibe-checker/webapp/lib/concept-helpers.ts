import { ConceptData, EnhancedConcept } from "@/types/concepts";

/**
 * Apply default values to a concept record so downstream UI can rely on every
 * field being present.
 */
export function enrichConceptData(concept: ConceptData): EnhancedConcept {
  return {
    wikibase_id: concept.wikibase_id,
    preferred_label: concept.preferred_label || concept.wikibase_id,
    description: concept.description || `Concept ${concept.wikibase_id}`,
    n_classifiers: concept.n_classifiers || 0,
  };
}

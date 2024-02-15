<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

/**
 * Results of a SPARQL query, along with information whether and how they were cached.
 * The results are represented using the
 * {@link https://www.w3.org/TR/sparql11-results-json/ SPARQL 1.1 Query Results JSON Format}.
 *
 * @license GPL-2.0-or-later
 */
class CachedQueryResults extends CachedArray {

	/**
	 * @return array The query results.
	 * For SELECT queries, you typically iterate over ['results']['bindings'],
	 * while for ASK queries, you typically check ['boolean'].
	 */
	public function getArray() {
		return parent::getArray();
	}

}
